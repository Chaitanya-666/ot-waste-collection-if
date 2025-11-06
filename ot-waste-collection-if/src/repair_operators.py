# ot-waste-collection-if/ot-waste-collection-if/src/repair_operators.py
"""
Repair operators for VRP with Intermediate Facilities (VRP-IF).

This module implements a small, clear set of repair operators for use in an
ALNS algorithm:
 - GreedyInsertion
 - RegretInsertion (k-regret)
 - IFAwareRepair (tries to place IF visits when needed)
 - SavingsInsertion (Clarke-Wright style)
 - RepairOperatorManager (simple adaptive selection support)

The implementation is intentionally conservative and defensive: it aims to be
easy to read, integrate with the project's `Solution` and `Route` classes, and
robust for typical small-to-medium problem instances.
"""

from copy import deepcopy
import random
from typing import List, Tuple, Optional

from .solution import Solution, Route
from .problem import Location, ProblemInstance


class RepairOperator:
    """Base class for repair operators."""

    def __init__(self, name: str):
        self.name = name
        self.performance_score = 0.0
        self.usage_count = 0

    def apply(self, partial_solution: Solution) -> Solution:
        """Apply the operator to a partial solution and return a completed solution."""
        raise NotImplementedError

    def update_performance(self, score: float) -> None:
        """Update operator performance (ALNS will call this)."""
        self.performance_score += score
        self.usage_count += 1

    def average_score(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.performance_score / self.usage_count


# ----------------------
# Helper utilities
# ----------------------
def calculate_route_distance(route: Route, problem: ProblemInstance) -> float:
    """Compute total Euclidean distance of a route using the provided problem."""
    if not route.nodes:
        return 0.0
    total = 0.0
    for i in range(len(route.nodes) - 1):
        total += problem.calculate_distance(route.nodes[i], route.nodes[i + 1])
    return total


def recalc_loads(route: Route) -> List[float]:
    """Recalculate cumulative loads after each node along a route.

    Rules:
      - depot (any node.type != 'customer') does not add demand
      - IF node (node.type == 'if') resets load to 0
    """
    loads: List[float] = []
    current_load = 0.0
    # Ensure there is at least an entry per node
    for node in route.nodes:
        if node.type == "customer":
            current_load += float(node.demand)
        elif node.type == "if":
            current_load = 0.0
        loads.append(current_load)
    return loads


def ensure_route_ends_with_depot(route: Route, problem: ProblemInstance) -> None:
    """Make sure the route starts and ends at depot (in-place)."""
    if not route.nodes:
        route.nodes = [problem.depot, problem.depot]
        route.loads = [0.0, 0.0]
        return
    if route.nodes[0] != problem.depot:
        route.nodes.insert(0, problem.depot)
    if route.nodes[-1] != problem.depot:
        route.nodes.append(problem.depot)
    route.loads = recalc_loads(route)


def try_insert_if_before(route: Route, insert_pos: int, if_facility: Location) -> None:
    """Insert IF facility at insert_pos (in-place)."""
    route.nodes.insert(insert_pos, if_facility)
    route.loads = recalc_loads(route)


def find_nearest_if(
    problem: ProblemInstance, ref: Optional[Location]
) -> Optional[Location]:
    """Return nearest IF facility to a reference location (or None if none).
    If ref is None, use the depot as the reference. This is defensive: if the
    depot is also missing, return the first IF (if any) or None.
    """
    if not problem.intermediate_facilities:
        return None

    # Use depot when ref is None
    if ref is None:
        ref = problem.depot

    # If still no reference (very defensive), return the first IF available
    if ref is None:
        return (
            problem.intermediate_facilities[0]
            if problem.intermediate_facilities
            else None
        )

    best = min(
        problem.intermediate_facilities,
        key=lambda ifn: problem.calculate_distance(ref, ifn),
    )
    return best


def enforce_if_visits(route: Route, problem: ProblemInstance) -> bool:
    """Ensure route includes IF visits so vehicle capacity is never exceeded.

    This function modifies the route.nodes list in-place by inserting IFs
    greedily before the point where capacity would be exceeded. Returns True
    if successful, False if infeasible (e.g., no IFs available when needed).
    """
    if not route.nodes:
        return False
    # We work on a mutable copy of nodes to avoid repeated re-scan complexity;
    # but we modify route.nodes in-place for caller to use.
    nodes = list(route.nodes)
    loads: List[float] = []
    current_load = 0.0
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node.type == "customer":
            current_load += float(node.demand)
            if current_load > problem.vehicle_capacity:
                # Need to insert nearest IF before this customer
                nearest_if = find_nearest_if(
                    problem, nodes[i - 1] if i - 1 >= 0 else problem.depot
                )
                if nearest_if is None:
                    return False
                nodes.insert(i, nearest_if)
                current_load = 0.0
                # continue scanning after the inserted IF
                i += 1
                continue
        elif node.type == "if":
            current_load = 0.0
        loads.append(current_load)
        i += 1
    # Ensure depots at start/end
    if nodes[0] != problem.depot:
        nodes.insert(0, problem.depot)
    if nodes[-1] != problem.depot:
        nodes.append(problem.depot)
    # write back
    route.nodes = nodes
    route.loads = recalc_loads(route)
    return True


def route_is_feasible(route: Route, problem: ProblemInstance) -> bool:
    """Basic feasibility check: max load <= vehicle capacity (if nodes present)."""
    if not route.nodes:
        return True
    loads = recalc_loads(route)
    return max(loads) <= problem.vehicle_capacity


# ----------------------
# Greedy insertion
# ----------------------
class GreedyInsertion(RepairOperator):
    """Greedy insertion: insert the next customer that produces the smallest cost increase."""

    def __init__(self):
        super().__init__("greedy_insertion")

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        # Make sure routes use depot endpoints
        for r in sol.routes:
            ensure_route_ends_with_depot(r, problem)

        # Build the unassigned customer list
        unassigned = list(getattr(sol, "unassigned_customers", []))

        # Heuristic ordering: larger demands first often helps
        unassigned.sort(key=lambda c: float(c.demand), reverse=True)

        while unassigned:
            best = None  # tuple (cost_increase, customer, route_idx, pos)
            for customer in unassigned:
                # Try inserting into each existing route
                for ridx, route in enumerate(sol.routes):
                    # feasible insertion positions are 1 .. len(route.nodes)-1
                    for pos in range(1, len(route.nodes)):
                        # simulate
                        route.nodes.insert(pos, customer)
                        route.loads = recalc_loads(route)
                        # if capacity violation, skip
                        if route_is_feasible(route, problem):
                            # compute distance with insertion
                            d_with = calculate_route_distance(route, problem)
                            # compute distance without insertion
                            route.nodes.pop(pos)
                            route.loads = recalc_loads(route)
                            d_without = calculate_route_distance(route, problem)
                            # restore insertion for evaluation consistency
                            route.nodes.insert(pos, customer)
                            route.loads = recalc_loads(route)
                            delta = d_with - d_without
                            if best is None or delta < best[0]:
                                best = (delta, customer, ridx, pos)
                        else:
                            # restore
                            route.nodes.pop(pos)
                            route.loads = recalc_loads(route)
                            continue
                        # restore if we didn't already
                        route.nodes.pop(pos)
                        route.loads = recalc_loads(route)

                # Try inserting as a new route (customer alone between depots)
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                new_route.loads = recalc_loads(new_route)
                if route_is_feasible(new_route, problem):
                    d_new = calculate_route_distance(new_route, problem)
                    if best is None or d_new < best[0]:
                        best = (d_new, customer, None, None)

            if best is None:
                # nothing feasible -> stop
                break

            # perform the best insertion found
            _, customer, ridx, pos = best
            if ridx is None:
                # add new route
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                new_route.loads = recalc_loads(new_route)
                sol.routes.append(new_route)
            else:
                # insert into existing route
                route = sol.routes[ridx]
                route.nodes.insert(pos, customer)
                route.loads = recalc_loads(route)

            # remove from unassigned
            if customer in unassigned:
                unassigned.remove(customer)
            if (
                hasattr(sol, "unassigned_customers")
                and customer in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(customer)

        sol.calculate_metrics()
        return sol


# ----------------------
# Regret insertion
# ----------------------
class RegretInsertion(RepairOperator):
    """Regret-k insertion."""

    def __init__(self, k: int = 2):
        super().__init__("regret_insertion")
        self.k = max(2, int(k))

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        for r in sol.routes:
            ensure_route_ends_with_depot(r, problem)

        unassigned = list(getattr(sol, "unassigned_customers", []))
        # sort large first to stabilize
        unassigned.sort(key=lambda c: float(c.demand), reverse=True)

        while unassigned:
            # For each customer compute best k insertion costs (route, pos)
            candidate_info = []
            for customer in unassigned:
                insertion_costs: List[Tuple[float, Optional[int], Optional[int]]] = []
                # existing routes
                for ridx, route in enumerate(sol.routes):
                    for pos in range(1, len(route.nodes)):
                        route.nodes.insert(pos, customer)
                        route.loads = recalc_loads(route)
                        feasible = route_is_feasible(route, problem)
                        d_with = (
                            calculate_route_distance(route, problem)
                            if feasible
                            else float("inf")
                        )
                        route.nodes.pop(pos)
                        route.loads = recalc_loads(route)
                        if feasible:
                            # cost increase relative to route without insertion:
                            # compute d_without:
                            d_without = calculate_route_distance(route, problem)
                            insertion_costs.append((d_with - d_without, ridx, pos))
                # new route option
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                new_route.loads = recalc_loads(new_route)
                if route_is_feasible(new_route, problem):
                    insertion_costs.append(
                        (calculate_route_distance(new_route, problem), None, None)
                    )

                insertion_costs.sort(key=lambda x: x[0])
                if len(insertion_costs) == 0:
                    continue
                # compute regret value from best k
                k_considered = min(self.k, len(insertion_costs))
                best_cost = insertion_costs[0][0]
                regret = sum(
                    insertion_costs[i][0] - best_cost for i in range(1, k_considered)
                )
                candidate_info.append((regret, customer, insertion_costs[0]))

            if not candidate_info:
                break

            # choose max regret
            candidate_info.sort(key=lambda x: x[0], reverse=True)
            _, chosen_customer, best_insertion = candidate_info[0]
            cost, ridx, pos = best_insertion

            # perform insertion
            if ridx is None:
                new_route = Route()
                new_route.nodes = [problem.depot, chosen_customer, problem.depot]
                new_route.loads = recalc_loads(new_route)
                sol.routes.append(new_route)
            else:
                sol.routes[ridx].nodes.insert(pos, chosen_customer)
                sol.routes[ridx].loads = recalc_loads(sol.routes[ridx])

            if chosen_customer in unassigned:
                unassigned.remove(chosen_customer)
            if (
                hasattr(sol, "unassigned_customers")
                and chosen_customer in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(chosen_customer)

        sol.calculate_metrics()
        return sol


# ----------------------
# IF-aware repair
# ----------------------
class IFAwareRepair(RepairOperator):
    """IF-aware repair: insert customers while ensuring IF visits are placed as needed."""

    def __init__(self):
        super().__init__("if_aware_repair")

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        # Normalize existing routes
        for r in sol.routes:
            ensure_route_ends_with_depot(r, problem)

        unassigned = list(getattr(sol, "unassigned_customers", []))
        # handle larger demands first
        unassigned.sort(key=lambda c: float(c.demand), reverse=True)

        while unassigned:
            best = None  # (cost, customer, ridx, pos)
            for customer in unassigned:
                # try existing routes
                for ridx, route in enumerate(sol.routes):
                    for pos in range(1, len(route.nodes)):
                        route.nodes.insert(pos, customer)
                        # attempt to enforce IFs for this tentative route (locally)
                        ok = enforce_if_visits(route, problem)
                        if ok:
                            d_with = calculate_route_distance(route, problem)
                            # compute distance without the inserted customer:
                            # temporarily remove and compute
                            route.nodes.pop(pos)
                            route.loads = recalc_loads(route)
                            d_without = calculate_route_distance(route, problem)
                            # restore insertion
                            route.nodes.insert(pos, customer)
                            route.loads = recalc_loads(route)
                            delta = d_with - d_without
                            if best is None or delta < best[0]:
                                best = (delta, customer, ridx, pos)
                        # restore original (remove any IFs too)
                        # remove IFs (simple approach: remove all IF nodes and rebuild loads)
                        route.nodes = [
                            n
                            for n in route.nodes
                            if n.type != "if" or n == problem.depot
                        ]
                        ensure_route_ends_with_depot(route, problem)

                # try new route
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                ok = enforce_if_visits(new_route, problem)
                if ok:
                    d_new = calculate_route_distance(new_route, problem)
                    if best is None or d_new < best[0]:
                        best = (d_new, customer, None, None)

            if best is None:
                break

            _, customer, ridx, pos = best
            if ridx is None:
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                enforce_if_visits(new_route, problem)
                new_route.loads = recalc_loads(new_route)
                sol.routes.append(new_route)
            else:
                route = sol.routes[ridx]
                route.nodes.insert(pos, customer)
                enforce_if_visits(route, problem)
                route.loads = recalc_loads(route)

            if customer in unassigned:
                unassigned.remove(customer)
            if (
                hasattr(sol, "unassigned_customers")
                and customer in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(customer)

        sol.calculate_metrics()
        return sol


# ----------------------
# Savings insertion
# ----------------------
class SavingsInsertion(RepairOperator):
    """Clarke-Wright style savings-based repair."""

    def __init__(self):
        super().__init__("savings_insertion")

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        # Build initial single-customer routes for unassigned customers
        unassigned = list(getattr(sol, "unassigned_customers", []))
        for customer in unassigned:
            r = Route()
            r.nodes = [problem.depot, customer, problem.depot]
            r.loads = recalc_loads(r)
            sol.routes.append(r)
            if (
                hasattr(sol, "unassigned_customers")
                and customer in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(customer)

        # compute savings between all route pairs (only single-customer routes initially)
        # savings = d(depot,c1) + d(depot,c2) - d(c1,c2)
        savings_list: List[Tuple[float, int, int]] = []
        for i in range(len(sol.routes)):
            for j in range(i + 1, len(sol.routes)):
                route_i = sol.routes[i]
                route_j = sol.routes[j]
                # customers are route_i.nodes[1] and route_j.nodes[1]
                if len(route_i.nodes) >= 3 and len(route_j.nodes) >= 3:
                    c1 = route_i.nodes[1]
                    c2 = route_j.nodes[1]
                    s = (
                        problem.calculate_distance(problem.depot, c1)
                        + problem.calculate_distance(problem.depot, c2)
                        - problem.calculate_distance(c1, c2)
                    )
                    savings_list.append((s, i, j))

        # sort savings descending
        savings_list.sort(key=lambda x: x[0], reverse=True)

        # Attempt merges
        for s, i, j in savings_list:
            # indices may have shifted; ensure valid
            if i >= len(sol.routes) or j >= len(sol.routes):
                continue
            if i == j:
                continue
            route_i = sol.routes[i]
            route_j = sol.routes[j]
            # Try merge: nodes = depot + route_i[1:-1] + route_j[1:-1] + depot
            merged = Route()
            merged.nodes = (
                [problem.depot]
                + route_i.nodes[1:-1]
                + route_j.nodes[1:-1]
                + [problem.depot]
            )
            # enforce IFs if needed
            ok = enforce_if_visits(merged, problem)
            if not ok:
                continue
            merged.loads = recalc_loads(merged)
            if route_is_feasible(merged, problem):
                # Replace routes i and j with merged (remove higher index first to be safe)
                hi, lo = max(i, j), min(i, j)
                sol.routes[lo] = merged
                sol.routes.pop(hi)

        sol.calculate_metrics()
        return sol


# ----------------------
# Manager
# ----------------------
class RepairOperatorManager:
    """Simple manager for repair operators with roulette selection and weight updates."""

    def __init__(self):
        self.operators: List[RepairOperator] = [
            GreedyInsertion(),
            RegretInsertion(k=2),
            IFAwareRepair(),
            SavingsInsertion(),
        ]
        self.weights: List[float] = [1.0 for _ in self.operators]
        self.iteration = 0
        self.learning_period = 50
        self.reaction = 0.2

    def select(self) -> RepairOperator:
        total = sum(self.weights)
        if total <= 0:
            return random.choice(self.operators)
        r = random.random() * total
        cum = 0.0
        for op, w in zip(self.operators, self.weights):
            cum += w
            if r <= cum:
                return op
        return self.operators[-1]

    def update(self) -> None:
        """Periodic weight update (normalize by average operator score)."""
        self.iteration += 1
        if self.iteration % self.learning_period != 0:
            return
        scores = [max(0.0, op.average_score()) for op in self.operators]
        total = sum(scores)
        if total <= 0:
            # reset to uniform
            self.weights = [1.0 for _ in self.operators]
            return
        # update weights using simple blending
        for i, score in enumerate(scores):
            normalized = score / total
            self.weights[i] = (1.0 - self.reaction) * self.weights[
                i
            ] + self.reaction * normalized

    def info(self):
        return [
            (op.name, self.weights[i], op.average_score())
            for i, op in enumerate(self.operators)
        ]
