# Author: Chaitanya Shinde (231070066)
#
# This file implements the "repair" operators for the ALNS algorithm. These
# operators take a partial solution (with some customers unassigned) and
# re-insert the unassigned customers to create a new, complete solution.
# Different operators use different heuristics to decide where to insert
# customers.
"""
Repair operators for VRP with Intermediate Facilities (VRP-IF).

This module provides a cleaned, consistent implementation of several repair
operators used by the ALNS solver:

 - GreedyInsertion
 - RegretInsertion (k-regret)
 - IFAwareRepair (ensures IF visits are placed to satisfy capacity)
 - SavingsInsertion (simple Clarke-Wright style merge)
 - RepairOperatorManager (simple adaptive manager)

Notes:
 - The project's Solution tracks `unassigned_customers` as a set of customer
   IDs (integers). Repair operators operate with Location objects but update
   the solution's `unassigned_customers` set using IDs.
 - The implementations are defensive: feasibility checks and insertion caps are
   employed to avoid pathological infinite loops.
"""

from copy import deepcopy
import random
from typing import List, Tuple, Optional

from .solution import Solution, Route
from .problem import Location, ProblemInstance


class RepairOperator:
    """Base class for all repair operators."""

    def __init__(self, name: str):
        self.name = name
        self.performance_score = 0.0
        self.usage_count = 0

    def apply(self, partial_solution: Solution) -> Solution:
        """Applies the repair operator to a partial solution, returning a complete one."""
        raise NotImplementedError

    def update_performance(self, score: float) -> None:
        """Updates the operator's performance score based on a recent run."""
        self.performance_score += score
        self.usage_count += 1

    def average_score(self) -> float:
        """Calculates the average performance score of this operator."""
        if self.usage_count == 0:
            return 0.0
        return self.performance_score / self.usage_count


# ----------------------
# Helper utilities
# ----------------------
def calculate_route_distance(route: Route, problem: ProblemInstance) -> float:
    """Computes the total Euclidean distance of a route."""
    if not route.nodes or len(route.nodes) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route.nodes) - 1):
        total += problem.calculate_distance(route.nodes[i], route.nodes[i + 1])
    return total


def recalc_loads(route: Route) -> List[float]:
    """
    Recalculates the cumulative load at each node in a route.
    The load resets to zero after visiting an intermediate facility (IF).
    """
    loads: List[float] = []
    current_load = 0.0
    for node in route.nodes:
        if node.type == "customer":
            current_load += float(node.demand)
        elif node.type == "if":
            current_load = 0.0
        loads.append(current_load)
    return loads


def ensure_route_ends_with_depot(route: Route, problem: ProblemInstance) -> None:
    """Ensures a route starts and ends at the depot (modifies the route in-place)."""
    if not route.nodes:
        route.nodes = [problem.depot, problem.depot]
        route.loads = [0.0, 0.0]
        return
    if route.nodes[0] != problem.depot:
        route.nodes.insert(0, problem.depot)
    if route.nodes[-1] != problem.depot:
        route.nodes.append(problem.depot)
    route.loads = recalc_loads(route)


def find_nearest_if(
    problem: ProblemInstance, ref: Optional[Location]
) -> Optional[Location]:
    """Finds the intermediate facility closest to a given reference location."""
    if not problem.intermediate_facilities:
        return None
    if ref is None:
        ref = problem.depot
    if ref is None:
        return (
            problem.intermediate_facilities[0]
            if problem.intermediate_facilities
            else None
        )
    return min(
        problem.intermediate_facilities,
        key=lambda ifn: problem.calculate_distance(ref, ifn),
    )


def enforce_if_visits(route: Route, problem: ProblemInstance) -> bool:
    """
    Iterates through a route and inserts intermediate facility (IF) visits
    wherever necessary to prevent vehicle capacity from being exceeded.
    Returns True on success, False on failure.
    """
    if not route.nodes:
        return False

    # A single customer's demand cannot exceed vehicle capacity
    for n in route.nodes:
        if n.type == "customer" and float(n.demand) > problem.vehicle_capacity:
            return False

    nodes = list(route.nodes)
    current_load = 0.0
    i = 0

    # Safety cap to avoid infinite loops in pathological cases
    insertion_count = 0
    max_insertions = max(10, len(nodes) * 2)
    last_insert_pos = -1

    while i < len(nodes):
        node = nodes[i]
        if node.type == "customer":
            current_load += float(node.demand)
            # If load is high, insert an IF visit before the current customer
            if current_load >= 0.8 * problem.vehicle_capacity:
                if insertion_count >= max_insertions:
                    return False
                nearest_if = find_nearest_if(
                    problem, nodes[i - 1] if i - 1 >= 0 else problem.depot
                )
                if nearest_if is None:
                    return False
                if i == last_insert_pos:
                    return False
                nodes.insert(i, nearest_if)
                insertion_count += 1
                last_insert_pos = i
                current_load = 0.0
                i += 1
                continue
        elif node.type == "if":
            current_load = 0.0

        i += 1

    # Ensure route starts/ends with depot
    if nodes and nodes[0] != problem.depot:
        nodes.insert(0, problem.depot)
    if nodes and nodes[-1] != problem.depot:
        nodes.append(problem.depot)

    route.nodes = nodes
    route.loads = recalc_loads(route)
    return True


def route_is_feasible(route: Route, problem: ProblemInstance) -> bool:
    """Checks if a route is feasible based on vehicle capacity."""
    if not route.nodes:
        return True
    loads = recalc_loads(route)
    return max(loads) <= problem.vehicle_capacity


# ----------------------
# Repair Operators
# ----------------------
class GreedyInsertion(RepairOperator):
    """
    Inserts unassigned customers one by one into the position that results
    in the smallest increase in total cost (distance).
    """

    def __init__(self):
        super().__init__("greedy_insertion")

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        for r in sol.routes:
            ensure_route_ends_with_depot(r, problem)

        unassigned_ids = set(getattr(sol, "unassigned_customers", set()))
        unassigned = [c for c in problem.customers if c.id in unassigned_ids]
        unassigned.sort(key=lambda c: float(c.demand), reverse=True)

        while unassigned:
            best = None  # Stores (cost_increase, customer, route_idx, position)
            for customer in list(unassigned):
                # Try inserting into existing routes
                for ridx, route in enumerate(sol.routes):
                    for pos in range(1, len(route.nodes)):
                        route.nodes.insert(pos, customer)
                        route.loads = recalc_loads(route)
                        if route_is_feasible(route, problem):
                            d_with = calculate_route_distance(route, problem)
                            route.nodes.pop(pos)
                            route.loads = recalc_loads(route)
                            d_without = calculate_route_distance(route, problem)
                            delta = d_with - d_without
                            if best is None or delta < best[0]:
                                best = (delta, customer, ridx, pos)
                        else:
                            route.nodes.pop(pos)
                            route.loads = recalc_loads(route)

                # Try creating a new route for the customer
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                new_route.loads = recalc_loads(new_route)
                if route_is_feasible(new_route, problem):
                    d_new = calculate_route_distance(new_route, problem)
                    if best is None or d_new < best[0]:
                        best = (d_new, customer, None, None)

            if best is None:
                break  # No feasible insertion found

            _, customer, ridx, pos = best
            if ridx is None: # Create a new route
                nr = Route()
                nr.nodes = [problem.depot, customer, problem.depot]
                nr.loads = recalc_loads(nr)
                sol.routes.append(nr)
            else: # Insert into an existing route
                sol.routes[ridx].nodes.insert(pos, customer)
                sol.routes[ridx].loads = recalc_loads(sol.routes[ridx])

            if customer in unassigned:
                unassigned.remove(customer)
            if (
                hasattr(sol, "unassigned_customers")
                and customer.id in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(customer.id)

        sol.calculate_metrics()
        return sol


class RegretInsertion(RepairOperator):
    """
    Inserts customers based on a "regret" value. The regret is the difference
    in cost between the best insertion position and the second-best (or k-th best).
    This prioritizes customers that have fewer good insertion options.
    """

    def __init__(self, k: int = 2):
        super().__init__("regret_insertion")
        self.k = max(2, int(k))

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        for r in sol.routes:
            ensure_route_ends_with_depot(r, problem)

        unassigned_ids = set(getattr(sol, "unassigned_customers", set()))
        unassigned = [c for c in problem.customers if c.id in unassigned_ids]
        unassigned.sort(key=lambda c: float(c.demand), reverse=True)

        while unassigned:
            candidate_info = []
            for customer in unassigned:
                insertion_costs = []
                # Find all possible insertion positions and their costs
                for ridx, route in enumerate(sol.routes):
                    for pos in range(1, len(route.nodes)):
                        route.nodes.insert(pos, customer)
                        route.loads = recalc_loads(route)
                        feasible = route_is_feasible(route, problem)
                        route.nodes.pop(pos)
                        route.loads = recalc_loads(route)
                        if feasible:
                            temp = Route()
                            temp.nodes = route.nodes.copy()
                            temp.nodes.insert(pos, customer)
                            temp.loads = recalc_loads(temp)
                            d_with = calculate_route_distance(temp, problem)
                            d_without = calculate_route_distance(route, problem)
                            insertion_costs.append((d_with - d_without, ridx, pos))

                # Consider creating a new route
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                new_route.loads = recalc_loads(new_route)
                if route_is_feasible(new_route, problem):
                    insertion_costs.append(
                        (calculate_route_distance(new_route, problem), None, None)
                    )

                insertion_costs.sort(key=lambda x: x[0])
                if not insertion_costs:
                    continue

                # Calculate regret value
                k_considered = min(self.k, len(insertion_costs))
                best_cost = insertion_costs[0][0]
                regret = sum(
                    insertion_costs[i][0] - best_cost for i in range(1, k_considered)
                )
                candidate_info.append((regret, customer, insertion_costs[0]))

            if not candidate_info:
                break

            # Choose the customer with the highest regret to insert next
            candidate_info.sort(key=lambda x: x[0], reverse=True)
            _, chosen_customer, best_insertion = candidate_info[0]
            cost, ridx, pos = best_insertion

            if ridx is None:
                nr = Route()
                nr.nodes = [problem.depot, chosen_customer, problem.depot]
                nr.loads = recalc_loads(nr)
                sol.routes.append(nr)
            else:
                sol.routes[ridx].nodes.insert(pos, chosen_customer)
                sol.routes[ridx].loads = recalc_loads(sol.routes[ridx])

            if chosen_customer in unassigned:
                unassigned.remove(chosen_customer)
            if (
                hasattr(sol, "unassigned_customers")
                and chosen_customer.id in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(chosen_customer.id)

        sol.calculate_metrics()
        return sol


class IFAwareRepair(RepairOperator):
    """
    A greedy insertion operator that is aware of intermediate facilities (IFs).
    When it inserts a customer, it also checks if an IF visit is needed and
    inserts one if necessary, ensuring the route remains feasible.
    """

    def __init__(self):
        super().__init__("if_aware_repair")

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        for r in sol.routes:
            ensure_route_ends_with_depot(r, problem)

        unassigned_ids = set(getattr(sol, "unassigned_customers", set()))
        unassigned = [c for c in problem.customers if c.id in unassigned_ids]
        unassigned.sort(key=lambda c: float(c.demand), reverse=True)

        while unassigned:
            best = None
            for customer in list(unassigned):
                # Try inserting into existing routes
                for ridx, route in enumerate(sol.routes):
                    for pos in range(1, len(route.nodes)):
                        original_nodes = route.nodes.copy()
                        original_loads = route.loads.copy()

                        # Create a temporary route for feasibility checking
                        tentative = Route()
                        tentative.nodes = original_nodes.copy()
                        tentative.nodes.insert(pos, customer)
                        tentative.loads = recalc_loads(tentative)

                        # Enforce IF visits and check feasibility
                        ok = enforce_if_visits(tentative, problem)
                        if ok and route_is_feasible(tentative, problem):
                            d_with = calculate_route_distance(tentative, problem)
                            baseline = Route()
                            baseline.nodes = original_nodes.copy()
                            baseline.loads = recalc_loads(baseline)
                            d_without = calculate_route_distance(baseline, problem)
                            delta = d_with - d_without
                            if best is None or delta < best[0]:
                                best = (delta, customer, ridx, pos)

                        route.nodes = original_nodes
                        route.loads = original_loads

                # Try creating a new route
                new_route = Route()
                new_route.nodes = [problem.depot, customer, problem.depot]
                ok = enforce_if_visits(new_route, problem)
                if ok and route_is_feasible(new_route, problem):
                    d_new = calculate_route_distance(new_route, problem)
                    if best is None or d_new < best[0]:
                        best = (d_new, customer, None, None)

            if best is None:
                break

            _, customer, ridx, pos = best
            if ridx is None:
                nr = Route()
                nr.nodes = [problem.depot, customer, problem.depot]
                enforce_if_visits(nr, problem)
                nr.loads = recalc_loads(nr)
                sol.routes.append(nr)
            else:
                sol.routes[ridx].nodes.insert(pos, customer)
                enforce_if_visits(sol.routes[ridx], problem)
                sol.routes[ridx].loads = recalc_loads(sol.routes[ridx])

            if customer in unassigned:
                unassigned.remove(customer)
            if (
                hasattr(sol, "unassigned_customers")
                and customer.id in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(customer.id)

        sol.calculate_metrics()
        return sol


class SavingsInsertion(RepairOperator):
    """
    A repair operator based on the Clarke and Wright savings heuristic. It first
    creates a separate route for each unassigned customer and then iteratively
    merges routes based on the highest "savings" in distance.
    """

    def __init__(self):
        super().__init__("savings_insertion")

    def apply(self, partial_solution: Solution) -> Solution:
        sol = deepcopy(partial_solution)
        problem: ProblemInstance = sol.problem

        unassigned_ids = set(getattr(sol, "unassigned_customers", set()))
        unassigned = [c for c in problem.customers if c.id in unassigned_ids]

        # Create a new route for each unassigned customer
        for customer in unassigned:
            r = Route()
            r.nodes = [problem.depot, customer, problem.depot]
            r.loads = recalc_loads(r)
            sol.routes.append(r)
            if (
                hasattr(sol, "unassigned_customers")
                and customer.id in sol.unassigned_customers
            ):
                sol.unassigned_customers.remove(customer.id)

        # Calculate savings for merging every pair of routes
        savings_list = []
        for i in range(len(sol.routes)):
            for j in range(i + 1, len(sol.routes)):
                ri = sol.routes[i]
                rj = sol.routes[j]
                if len(ri.nodes) >= 3 and len(rj.nodes) >= 3:
                    c1 = ri.nodes[1]
                    c2 = rj.nodes[1]
                    if (
                        getattr(problem, "depot", None) is None
                        or c1 is None
                        or c2 is None
                    ):
                        continue
                    try:
                        # Savings = (dist(depot, c1) + dist(depot, c2)) - dist(c1, c2)
                        d_depot_c1 = problem.calculate_distance(problem.depot, c1)
                        d_depot_c2 = problem.calculate_distance(problem.depot, c2)
                        d_c1_c2 = problem.calculate_distance(c1, c2)
                        s = d_depot_c1 + d_depot_c2 - d_c1_c2
                    except Exception:
                        continue
                    savings_list.append((s, i, j))

        savings_list.sort(key=lambda x: x[0], reverse=True)

        # Iteratively merge routes with the highest savings
        for s, i, j in savings_list:
            if i >= len(sol.routes) or j >= len(sol.routes) or i == j:
                continue
            ri = sol.routes[i]
            rj = sol.routes[j]
            merged = Route()
            merged.nodes = (
                [problem.depot] + ri.nodes[1:-1] + rj.nodes[1:-1] + [problem.depot]
            )
            ok = enforce_if_visits(merged, problem)
            if not ok:
                continue
            merged.loads = recalc_loads(merged)
            if route_is_feasible(merged, problem):
                hi, lo = max(i, j), min(i, j)
                sol.routes[lo] = merged
                sol.routes.pop(hi)

        sol.calculate_metrics()
        return sol


class RepairOperatorManager:
    """Manages the selection and adaptive weighting of repair operators."""

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
        self.reaction = 0.2 # How much the weights are adjusted based on performance

    def select(self) -> RepairOperator:
        """Selects a repair operator using roulette wheel selection based on weights."""
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
        """Periodically updates the weights of the operators based on their performance."""
        self.iteration += 1
        if self.iteration % self.learning_period != 0:
            return
        scores = [max(0.0, op.average_score()) for op in self.operators]
        total = sum(scores)
        if total <= 0:
            self.weights = [1.0 for _ in self.operators]
            return
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
