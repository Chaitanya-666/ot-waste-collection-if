"""
ALNS framework for VRP with Intermediate Facilities
"""

import random
import math
import time
from typing import List, Tuple, Dict, Optional
from .solution import Solution, Route
from .problem import ProblemInstance
from .repair_operators import (
    RepairOperatorManager,
    enforce_if_visits,
    recalc_loads,
    route_is_feasible,
)
from .destroy_operators import DestroyOperatorManager


class ALNS:
    def __init__(self, problem_instance: ProblemInstance):
        self.problem = problem_instance
        self.current_solution = None
        self.best_solution = None
        self.iteration = 0
        self.start_time = None
        self.convergence_history = []
        # Optional callback used to support live plotting / iteration updates.
        # If set, this should be a callable accepting (iteration:int, best_solution: Solution)
        # The demo / UI can assign a function to receive intermediate results.
        self.iteration_callback = None

        # ALNS parameters
        self.max_iterations = 1000
        self.destroy_degree = 0.15  # Percentage of customers to remove
        self.temperature = 1000.0
        self.cooling_rate = 0.9995
        self.adaptive_period = 100
        self.seed = 42

        # Operator managers and weights
        # Use DestroyOperatorManager for destroy operators (class-based)
        self.destroy_manager = DestroyOperatorManager(self.problem)
        self.destroy_operators = list(self.destroy_manager.operators.keys())

        # Use RepairOperatorManager (class-based repair operators)
        self.repair_manager = RepairOperatorManager()
        # Keep a small list of repair operator names for compatibility
        self.repair_operators = [op.name for op in self.repair_manager.operators]

        # Initialize weights (equal distribution)
        self.destroy_weights = {op: 1.0 for op in self.destroy_operators}
        self.repair_weights = {op: 1.0 for op in self.repair_operators}

        # Operator performance tracking
        self.destroy_scores = {op: 0.0 for op in self.destroy_operators}
        self.repair_scores = {op: 0.0 for op in self.repair_operators}
        self.destroy_usage = {op: 0 for op in self.destroy_operators}
        self.repair_usage = {op: 0 for op in self.repair_operators}

        # Random seed
        random.seed(self.seed)

        # Learning rate for weight updates
        self.learning_rate = 0.1

    def run(self, max_iterations: Optional[int] = None) -> Solution:
        """Run the ALNS optimization algorithm"""
        if max_iterations:
            self.max_iterations = max_iterations

        self.start_time = time.time()
        print(f"Starting ALNS optimization with {self.max_iterations} iterations...")

        # Generate initial solution
        self.current_solution = self._generate_initial_solution()
        self.best_solution = self.current_solution.copy()

        print(f"Initial solution: {self.current_solution.total_cost:.2f}")

        # Main ALNS loop
        for iteration in range(self.max_iterations):
            self.iteration = iteration

            # Select operators based on weights
            destroy_op = self._select_destroy_operator()
            repair_op = self._select_repair_operator()

            # Track operator usage
            self.destroy_usage[destroy_op] += 1
            self.repair_usage[repair_op] += 1

            # Destroy phase
            partial_solution = self._destroy(self.current_solution, destroy_op)

            # Repair phase
            new_solution = self._repair(partial_solution, repair_op)

            # Recompute candidate solution cost (includes penalty for unassigned customers)
            if new_solution is not None:
                try:
                    new_solution.total_cost = self._calculate_total_cost(new_solution)
                except Exception:
                    # if recalculation fails, continue without breaking the iteration
                    pass

            # Acceptance: operate on the penalized objective (distance + heavy penalty for unassigned customers).
            # Allow the SA acceptance criterion to accept improving penalized-cost solutions even if
            # they are not strictly fully-feasible so the search can move toward complete assignments.
            if new_solution is not None and self._accept_solution(new_solution):
                self.current_solution = new_solution

                # Update best solution based on penalized cost (prefer lower penalized cost)
                if new_solution.total_cost < self.best_solution.total_cost:
                    self.best_solution = new_solution.copy()
                    print(
                        f"New best solution found at iteration {iteration}: {new_solution.total_cost:.2f}"
                    )

            # Update operator scores
            self._update_operator_scores(destroy_op, repair_op, new_solution)

            # Adaptive weight adjustment
            if iteration % self.adaptive_period == 0 and iteration > 0:
                self._update_operator_weights()

            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Track convergence
            self.convergence_history.append(self.best_solution.total_cost)

            # If a live-plot / iteration callback is registered, call it with the current iteration.
            # The callback can be used to update a plot or UI in real-time.
            if callable(getattr(self, "iteration_callback", None)):
                try:
                    # provide a safe copy or the best solution object depending on caller needs
                    self.iteration_callback(iteration, self.best_solution)
                except Exception:
                    # fail-safe: do not interrupt ALNS if callback raises
                    pass

            # Progress reporting
            if iteration % 100 == 0:
                elapsed = time.time() - self.start_time
                print(
                    f"Iteration {iteration}/{self.max_iterations}, "
                    f"Best: {self.best_solution.total_cost:.2f}, "
                    f"Time: {elapsed:.1f}s"
                )

        final_time = time.time() - self.start_time
        print(f"ALNS completed in {final_time:.1f} seconds")
        print(f"Final best solution: {self.best_solution.total_cost:.2f}")
        print(f"Total iterations: {self.max_iterations}")

        # Post-process: attempt to greedily insert any remaining unassigned customers
        # into existing routes (or create new routes up to the available fleet).
        # This helps finalize solutions and reduce leftover unassigned customers.
        try:
            _post_process_assignment(self, self.best_solution)
            # Recalculate final aggregated metrics and penalized cost
            self.best_solution.calculate_metrics()
            self.best_solution.total_cost = self._calculate_total_cost(
                self.best_solution
            )
            print("Post-processing complete: attempted to assign remaining customers.")
            print(
                f"Final best solution (post-process): {self.best_solution.total_cost:.2f}"
            )
        except Exception:
            # Fail-safe: do not break on post-process errors
            pass

        return self.best_solution


# ----------------------------------------------------------------------
# Module-level helper: post-processing finalizer
# Kept at module level to avoid altering class layout while still giving
# the routine access to solver internals when passed `self`.
# ----------------------------------------------------------------------
def _post_process_assignment(solver: "ALNS", solution: Solution) -> None:
    """
    Greedy post-processing to insert any unassigned customers into existing routes
    or by creating new routes up to the problem's `number_of_vehicles`.

    This routine is intentionally simple: it tries feasible insertions (respecting IFs
    via `enforce_if_visits`) and stops when no further assignments are possible.
    """
    problem = solver.problem

    # Build quick id->Location map for customers
    id_to_customer = {c.id: c for c in getattr(problem, "customers", [])}

    # Determine unassigned customer IDs (solution.unassigned_customers is the canonical store)
    unassigned_ids = set(getattr(solution, "unassigned_customers", set()) or set())

    # Defensive exit if nothing to do
    if not unassigned_ids:
        return

    max_vehicles = (
        int(problem.number_of_vehicles)
        if getattr(problem, "number_of_vehicles", None)
        else len(solution.routes) or 1
    )

    made_progress = True
    # iterate until no more placements in a full pass
    while unassigned_ids and made_progress:
        made_progress = False
        for cust_id in list(unassigned_ids):
            customer = id_to_customer.get(cust_id)
            if customer is None:
                # unknown customer id: drop it
                unassigned_ids.discard(cust_id)
                continue

            placed = False

            # Try to insert into existing routes at any position
            for route in solution.routes:
                # try positions between nodes (1 .. len-1)
                for pos in range(1, max(1, len(route.nodes))):
                    # build tentative route
                    tentative = Route()
                    tentative.nodes = route.nodes.copy()
                    tentative.nodes.insert(pos, customer)
                    tentative.loads = recalc_loads(tentative)

                    # enforce IF visits on tentative route
                    try:
                        ok = enforce_if_visits(tentative, problem)
                    except Exception:
                        ok = False

                    if not ok:
                        continue

                    # check capacity feasibility
                    try:
                        feasible = route_is_feasible(tentative, problem)
                    except Exception:
                        feasible = False

                    if feasible:
                        # perform the insertion on the real route and enforce IFs there
                        route.nodes.insert(pos, customer)
                        try:
                            enforce_if_visits(route, problem)
                        except Exception:
                            pass
                        route.loads = recalc_loads(route)
                        placed = True
                        break
                if placed:
                    break

            # If not placed, attempt to create a new route if fleet allows
            if (
                not placed
                and len(
                    [
                        r
                        for r in solution.routes
                        if r.nodes
                        and any(getattr(n, "type", None) == "customer" for n in r.nodes)
                    ]
                )
                < max_vehicles
            ):
                nr = Route()
                nr.nodes = [problem.depot, customer, problem.depot]
                nr.loads = recalc_loads(nr)
                try:
                    ok = enforce_if_visits(nr, problem)
                except Exception:
                    ok = False
                if ok and route_is_feasible(nr, problem):
                    solution.routes.append(nr)
                    placed = True

            if placed:
                unassigned_ids.discard(cust_id)
                made_progress = True

        # end for each customer

    # write back remaining unassigned ids into solution object
    try:
        solution.unassigned_customers = set(unassigned_ids)
    except Exception:
        # best-effort: ignore failures writing back
        pass

    def _generate_initial_solution(self) -> Solution:
        """Generate initial feasible solution using a simple nearest-neighbour construction.

        This method builds routes incrementally and ensures:
         - each route starts and ends at the depot
         - per-route loads are respected (no route will have load > vehicle_capacity)
         - `solution.unassigned_customers` is a set of customer IDs not yet assigned
        """
        solution = Solution(self.problem)

        # Remaining customers (Location objects)
        remaining = [c for c in self.problem.customers]

        # Ensure we create at least the minimum number of empty routes required
        # by the problem instance so the greedy constructor can utilise the fleet.
        min_needed = 1
        try:
            min_needed = max(1, int(self.problem.get_min_vehicles_needed()))
        except Exception:
            # fallback: leave min_needed as 1 if method unavailable
            min_needed = 1

        # Create that many empty routes up-front (each starts at depot)
        for _ in range(min_needed):
            r = self._create_empty_route()
            solution.routes.append(r)

        # Use the first route as the current insertion target initially
        current_route = (
            solution.routes[0] if solution.routes else self._create_empty_route()
        )
        current_location = self.problem.depot

        # Greedy nearest insertion while respecting capacity: if current route cannot accept
        # the nearest customer, start a new route.
        while remaining:
            # choose nearest customer to the current location
            nearest_customer = min(
                remaining,
                key=lambda c: self.problem.calculate_distance(current_location, c),
            )

            # compute current load on route
            cur_load = sum(
                getattr(node, "demand", 0)
                for node in current_route.nodes
                if getattr(node, "type", None) == "customer"
            )

            # if adding nearest_customer would exceed capacity, close current route and start a new one
            if (
                cur_load + float(getattr(nearest_customer, "demand", 0))
                > self.problem.vehicle_capacity
            ):
                # ensure current route ends at depot
                if (
                    not current_route.nodes
                    or current_route.nodes[-1] != self.problem.depot
                ):
                    current_route.nodes.append(self.problem.depot)
                current_route.calculate_metrics(self.problem)

                # start a fresh route
                current_route = self._create_empty_route()
                solution.routes.append(current_route)
                current_location = self.problem.depot
                continue

            # ensure route has an end depot to insert before; create if missing
            if not current_route.nodes or current_route.nodes[-1] != self.problem.depot:
                # if route only has starting depot, append end depot so we can insert before it
                current_route.nodes.append(self.problem.depot)

            # insert customer before the terminal depot
            insert_pos = len(current_route.nodes) - 1
            current_route.nodes.insert(insert_pos, nearest_customer)
            # recalc route metrics
            current_route.calculate_metrics(self.problem)

            # remove from remaining and advance current location
            remaining.remove(nearest_customer)
            current_location = nearest_customer

        # Finalize all routes: ensure they start and end with depot and calculate metrics
        for route in solution.routes:
            if not route.nodes:
                route.nodes = [self.problem.depot, self.problem.depot]
            if route.nodes[0] != self.problem.depot:
                route.nodes.insert(0, self.problem.depot)
            if route.nodes[-1] != self.problem.depot:
                route.nodes.append(self.problem.depot)
            route.calculate_metrics(self.problem)

        # Build set of served customer IDs and set unassigned_customers as IDs
        served_ids = set()
        for route in solution.routes:
            for node in route.nodes:
                if getattr(node, "type", None) == "customer":
                    served_ids.add(node.id)

        all_ids = set(c.id for c in self.problem.customers)
        solution.unassigned_customers = all_ids - served_ids

        # Compute aggregated metrics for the solution
        solution.total_distance = sum(
            getattr(r, "total_distance", 0.0) for r in solution.routes
        )
        solution.total_time = sum(
            getattr(r, "total_time", 0.0) for r in solution.routes
        )
        # Use ALNS's cost calculator which includes a penalty for unassigned customers
        # so the search prefers complete assignments over partial low-distance solutions.
        solution.total_cost = self._calculate_total_cost(solution)

        return solution

    def _create_empty_route(self) -> "Route":
        """Create an empty route"""
        from .solution import Route

        route = Route()
        route.nodes = [self.problem.depot]
        route.loads = [0]
        return route

    def _get_route_load(self, route: "Route") -> int:
        """Calculate current load of a route"""
        return sum(node.demand for node in route.nodes if node.type == "customer")

    def _calculate_total_cost(self, solution: Solution) -> float:
        """Calculate total cost of a solution.

        In addition to the travel distance (sum of edge distances) this method
        includes a heavy penalty for any unassigned customers so the ALNS search
        prefers complete assignments where feasible.
        """
        total_cost = 0.0

        for route in solution.routes:
            for i in range(len(route.nodes) - 1):
                from_node = route.nodes[i]
                to_node = route.nodes[i + 1]
                distance = self.problem.calculate_distance(from_node, to_node)
                total_cost += distance

        # Penalize unassigned customers heavily so that complete-service solutions
        # are preferred over lower-distance partial solutions.
        penalty_per_unassigned = 1000.0
        try:
            n_unassigned = 0
            if (
                hasattr(solution, "unassigned_customers")
                and solution.unassigned_customers is not None
            ):
                # unassigned_customers stored as a set of IDs in this project
                n_unassigned = len(solution.unassigned_customers)
            else:
                # Fallback: compute by comparing customer ids in routes
                assigned = set()
                for r in solution.routes:
                    for n in r.nodes:
                        if getattr(n, "type", None) == "customer":
                            assigned.add(n.id)
                all_ids = set(c.id for c in self.problem.customers)
                n_unassigned = len(all_ids - assigned)
            total_cost += penalty_per_unassigned * float(n_unassigned)
        except Exception:
            # If anything goes wrong, avoid failing the cost calculation; do not add penalty.
            pass

        return total_cost

    def _select_destroy_operator(self) -> str:
        """Select destroy operator.

        Prefer the DestroyOperatorManager selection (roulette by manager weights).
        Fall back to the legacy weight-based selection if the manager is not available.
        """
        try:
            if hasattr(self, "destroy_manager") and callable(
                getattr(self.destroy_manager, "select_operator", None)
            ):
                return self.destroy_manager.select_operator()
        except Exception:
            # defensive fallback to legacy mechanism below
            pass

        # Legacy weighted selection (kept for compatibility)
        weights = [self.destroy_weights[op] for op in self.destroy_operators]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(self.destroy_operators)

        probabilities = [w / total_weight for w in weights]
        return random.choices(self.destroy_operators, weights=probabilities)[0]

    def _select_repair_operator(self) -> str:
        """Select repair operator based on weights"""
        weights = [self.repair_weights[op] for op in self.repair_operators]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(self.repair_operators)

        probabilities = [w / total_weight for w in weights]
        return random.choices(self.repair_operators, weights=probabilities)[0]

    def _destroy(self, solution: Solution, operator: str) -> Solution:
        """Apply destroy operator to create partial solution.

        Try to use the DestroyOperatorManager (class-based). If it fails for any
        reason, fall back to the legacy string-based remove functions.
        """
        partial_solution = solution.copy()
        removal_count = max(1, int(len(self.problem.customers) * self.destroy_degree))

        # Preferred path: manager-driven destroy
        try:
            if hasattr(self, "destroy_manager") and callable(
                getattr(self.destroy_manager, "apply_operator", None)
            ):
                return self.destroy_manager.apply_operator(
                    partial_solution, operator, removal_count
                )
        except Exception:
            # Fall through to legacy handlers on error
            pass

        # Legacy handlers (compatibility)
        if operator == "random":
            return self._random_removal(partial_solution, removal_count)
        elif operator == "worst":
            return self._worst_removal(partial_solution, removal_count)
        elif operator == "shaw":
            return self._shaw_removal(partial_solution, removal_count)
        elif operator == "route":
            return self._route_removal(partial_solution, removal_count)
        else:
            return self._random_removal(partial_solution, removal_count)

    def _random_removal(self, solution: Solution, count: int) -> Solution:
        """Randomly remove customers from solution"""
        all_customers = [
            node
            for route in solution.routes
            for node in route.nodes
            if node.type == "customer"
        ]

        if len(all_customers) <= count:
            return self._create_empty_solution()

        removed_customers = random.sample(all_customers, min(count, len(all_customers)))

        for customer in removed_customers:
            for route in solution.routes:
                if customer in route.nodes:
                    route.nodes.remove(customer)
                    route.loads = self._recalculate_loads(route)
                    break

        # Clean up empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _worst_removal(self, solution: Solution, count: int) -> Solution:
        """Remove customers with highest marginal cost"""
        all_customers = [
            node
            for route in solution.routes
            for node in route.nodes
            if node.type == "customer"
        ]

        if len(all_customers) <= count:
            return self._create_empty_solution()

        # Calculate marginal cost for each customer
        customer_costs = []
        for customer in all_customers:
            cost_increase = self._calculate_marginal_cost(customer, solution)
            customer_costs.append((customer, cost_increase))

        # Sort by cost and remove worst ones
        customer_costs.sort(key=lambda x: x[1], reverse=True)
        removed_customers = [customer for customer, _ in customer_costs[:count]]

        for customer in removed_customers:
            for route in solution.routes:
                if customer in route.nodes:
                    route.nodes.remove(customer)
                    route.loads = self._recalculate_loads(route)
                    break

        # Clean up empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _shaw_removal(self, solution: Solution, count: int) -> Solution:
        """Remove similar customers based on proximity and demand"""
        all_customers = [
            node
            for route in solution.routes
            for node in route.nodes
            if node.type == "customer"
        ]

        if len(all_customers) <= count:
            return self._create_empty_solution()

        # Select random seed customer
        seed_customer = random.choice(all_customers)

        # Find similar customers
        similar_customers = []
        for customer in all_customers:
            if customer != seed_customer:
                distance = self.problem.calculate_distance(seed_customer, customer)
                demand_diff = abs(seed_customer.demand - customer.demand)
                similarity = 1.0 / (1.0 + distance + demand_diff)
                similar_customers.append((customer, similarity))

        # Sort by similarity and remove most similar
        similar_customers.sort(key=lambda x: x[1], reverse=True)
        removed_customers = [customer for customer, _ in similar_customers[:count]]

        for customer in removed_customers:
            for route in solution.routes:
                if customer in route.nodes:
                    route.nodes.remove(customer)
                    route.loads = self._recalculate_loads(route)
                    break

        # Clean up empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _route_removal(self, solution: Solution, count: int) -> Solution:
        """Remove entire route segments"""
        if len(solution.routes) <= count:
            return self._create_empty_solution()

        # Select random routes to remove
        routes_to_remove = random.sample(
            solution.routes, min(count, len(solution.routes))
        )

        for route in routes_to_remove:
            solution.routes.remove(route)

        return solution

    def _create_empty_solution(self) -> Solution:
        """Create an empty solution"""
        return Solution(self.problem)

    def _recalculate_loads(self, route: "Route") -> List[int]:
        """Recalculate loads for a route"""
        loads = [0]
        current_load = 0

        for node in route.nodes[1:]:  # Skip depot
            if node.type == "customer":
                current_load += node.demand
            loads.append(current_load)

        return loads

    def _calculate_marginal_cost(
        self, customer: "Location", solution: Solution
    ) -> float:
        """Calculate marginal cost of removing a customer"""
        original_cost = self._calculate_total_cost(solution)

        # Temporarily remove customer
        for route in solution.routes:
            if customer in route.nodes:
                route.nodes.remove(customer)
                route.loads = self._recalculate_loads(route)
                break

        new_cost = self._calculate_total_cost(solution)

        # Restore customer
        for route in solution.routes:
            if customer in route.nodes:
                break
        else:
            # Customer wasn't found, add it back to first route
            solution.routes[0].nodes.insert(1, customer)
            solution.routes[0].loads = self._recalculate_loads(solution.routes[0])

        return new_cost - original_cost

    def _repair(self, partial_solution: Solution, operator: str) -> Solution:
        """Apply repair operator to create complete solution.

        This integrates the class-based RepairOperatorManager. The ALNS main loop
        still selects an operator name (string). We try to map that name to a
        manager operator (matching by prefix), otherwise fall back to legacy
        string-based methods or ask the manager to select an operator.
        """
        # Try to find a matching operator in the manager by name prefix
        try:
            for op in self.repair_manager.operators:
                # match by prefix to allow legacy names like 'greedy' to match 'greedy_insertion'
                if op.name.startswith(operator):
                    return op.apply(partial_solution)
        except Exception:
            # defensive - if manager not available or something goes wrong, fallback
            pass

        # Fallback to existing string-based methods for compatibility
        if operator == "greedy":
            return self._greedy_insertion(partial_solution)
        elif operator == "regret":
            return self._regret_insertion(partial_solution)
        elif operator == "if_aware":
            return self._if_aware_repair(partial_solution)
        else:
            # As a last resort, ask the manager to pick an operator and apply it
            selected = self.repair_manager.select()
            return selected.apply(partial_solution)

    def _greedy_insertion(self, partial_solution: Solution) -> Solution:
        """Greedy insertion of remaining customers"""
        unassigned_customers = self._get_unassigned_customers(partial_solution)

        while unassigned_customers:
            best_customer = None
            best_position = None
            best_cost = float("inf")

            for customer in unassigned_customers:
                for route_idx, route in enumerate(partial_solution.routes):
                    for position in range(1, len(route.nodes)):
                        # Try inserting customer
                        original_nodes = route.nodes.copy()
                        original_loads = route.loads.copy()

                        route.nodes.insert(position, customer)
                        route.loads = self._recalculate_loads(route)

                        # Check feasibility
                        if self._is_route_feasible(route):
                            cost_increase = self._calculate_insertion_cost(
                                route, position - 1, position + 1
                            )

                            if cost_increase < best_cost:
                                best_cost = cost_increase
                                best_customer = customer
                                best_position = (route_idx, position)

                        # Restore route
                        route.nodes = original_nodes
                        route.loads = original_loads

            # Insert best customer
            if best_customer:
                route_idx, position = best_position
                partial_solution.routes[route_idx].nodes.insert(position, best_customer)
                partial_solution.routes[route_idx].loads = self._recalculate_loads(
                    partial_solution.routes[route_idx]
                )
                unassigned_customers.remove(best_customer)
            else:
                # Create new route if no feasible insertion found
                new_route = self._create_empty_route()
                new_route.nodes.insert(1, unassigned_customers[0])
                new_route.loads = self._recalculate_loads(new_route)
                partial_solution.routes.append(new_route)
                unassigned_customers.remove(unassigned_customers[0])

        # Add depots at end of routes
        for route in partial_solution.routes:
            route.nodes.append(self.problem.depot)
            route.loads.append(0)

        partial_solution.total_cost = self._calculate_total_cost(partial_solution)
        return partial_solution

    def _regret_insertion(self, partial_solution: Solution) -> Solution:
        """Regret-k insertion for better customer placement"""
        unassigned_customers = self._get_unassigned_customers(partial_solution)
        k_regret = 2

        while unassigned_customers:
            best_customer = None
            max_regret = -float("inf")

            for customer in unassigned_customers:
                insertion_costs = []

                # Calculate insertion costs for all possible positions
                for route_idx, route in enumerate(partial_solution.routes):
                    for position in range(1, len(route.nodes)):
                        original_nodes = route.nodes.copy()
                        original_loads = route.loads.copy()

                        route.nodes.insert(position, customer)
                        route.loads = self._recalculate_loads(route)

                        if self._is_route_feasible(route):
                            cost_increase = self._calculate_insertion_cost(
                                route, position - 1, position + 1
                            )
                            insertion_costs.append(cost_increase)

                        # Restore route
                        route.nodes = original_nodes
                        route.loads = original_loads

                # Calculate regret
                if len(insertion_costs) >= k_regret:
                    insertion_costs.sort()
                    regret = sum(insertion_costs[1:k_regret]) - insertion_costs[0] * (
                        k_regret - 1
                    )

                    if regret > max_regret:
                        max_regret = regret
                        best_customer = customer

            # Insert best customer using greedy insertion
            if best_customer:
                partial_solution = self._greedy_insertion_for_customer(
                    partial_solution, best_customer
                )
                unassigned_customers.remove(best_customer)
            else:
                # Create new route
                new_route = self._create_empty_route()
                new_route.nodes.insert(1, unassigned_customers[0])
                new_route.loads = self._recalculate_loads(new_route)
                partial_solution.routes.append(new_route)
                unassigned_customers.remove(unassigned_customers[0])

        # Add depots at end of routes
        for route in partial_solution.routes:
            route.nodes.append(self.problem.depot)
            route.loads.append(0)

        partial_solution.total_cost = self._calculate_total_cost(partial_solution)
        return partial_solution

    def _if_aware_repair(self, partial_solution: Solution) -> Solution:
        """IF-aware repair considering intermediate facility constraints"""
        unassigned_customers = self._get_unassigned_customers(partial_solution)

        while unassigned_customers:
            best_customer = None
            best_position = None
            best_cost = float("inf")

            for customer in unassigned_customers:
                for route_idx, route in enumerate(partial_solution.routes):
                    for position in range(1, len(route.nodes)):
                        original_nodes = route.nodes.copy()
                        original_loads = route.loads.copy()

                        route.nodes.insert(position, customer)
                        route.loads = self._recalculate_loads(route)

                        # Check feasibility including IF requirements
                        if self._is_route_feasible_with_if(route):
                            cost_increase = self._calculate_insertion_cost(
                                route, position - 1, position + 1
                            )

                            if cost_increase < best_cost:
                                best_cost = cost_increase
                                best_customer = customer
                                best_position = (route_idx, position)

                        # Restore route
                        route.nodes = original_nodes
                        route.loads = original_loads

            # Insert best customer
            if best_customer:
                route_idx, position = best_position
                partial_solution.routes[route_idx].nodes.insert(position, best_customer)
                partial_solution.routes[route_idx].loads = self._recalculate_loads(
                    partial_solution.routes[route_idx]
                )
                unassigned_customers.remove(best_customer)
            else:
                # Create new route
                new_route = self._create_empty_route()
                new_route.nodes.insert(1, unassigned_customers[0])
                new_route.loads = self._recalculate_loads(new_route)
                partial_solution.routes.append(new_route)
                unassigned_customers.remove(unassigned_customers[0])

        # Add depots at end of routes
        for route in partial_solution.routes:
            route.nodes.append(self.problem.depot)
            route.loads.append(0)

        partial_solution.total_cost = self._calculate_total_cost(partial_solution)
        return partial_solution

    def _get_unassigned_customers(self, solution: Solution) -> List["Location"]:
        """Get list of unassigned customers.

        Prefer using the solution's `unassigned_customers` set of IDs when available
        (identity-safe). Fall back to deriving unassigned customers from routes by id.
        """
        # If the solution explicitly tracks unassigned customer IDs, use that.
        try:
            unassigned_ids = set(getattr(solution, "unassigned_customers", set()))
            if unassigned_ids:
                return [c for c in self.problem.customers if c.id in unassigned_ids]
        except Exception:
            # Defensive: fall back to deriving from routes below.
            pass

        # Fallback: derive unassigned by checking which customer IDs do not appear in routes.
        assigned_ids = set()
        for route in solution.routes:
            for node in route.nodes:
                if getattr(node, "type", None) == "customer":
                    assigned_ids.add(getattr(node, "id", None))

        return [c for c in self.problem.customers if c.id not in assigned_ids]

    def _greedy_insertion_for_customer(
        self, solution: Solution, customer: "Location"
    ) -> Solution:
        """Greedy insertion for a specific customer"""
        best_position = None
        best_cost = float("inf")

        for route_idx, route in enumerate(solution.routes):
            for position in range(1, len(route.nodes)):
                original_nodes = route.nodes.copy()
                original_loads = route.loads.copy()

                route.nodes.insert(position, customer)
                route.loads = self._recalculate_loads(route)

                if self._is_route_feasible(route):
                    cost_increase = self._calculate_insertion_cost(
                        route, position - 1, position + 1
                    )

                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_position = (route_idx, position)

                # Restore route
                route.nodes = original_nodes
                route.loads = original_loads

        # Insert at best position
        if best_position:
            route_idx, position = best_position
            solution.routes[route_idx].nodes.insert(position, customer)
            solution.routes[route_idx].loads = self._recalculate_loads(
                solution.routes[route_idx]
            )

        return solution

    def _is_route_feasible(self, route: "Route") -> bool:
        """Check if route is feasible (capacity constraints)"""
        max_load = max(route.loads) if route.loads else 0
        return max_load <= self.problem.vehicle_capacity

    def _is_route_feasible_with_if(self, route: "Route") -> bool:
        """Check if route is feasible with IF constraints"""
        # Basic feasibility check
        if not self._is_route_feasible(route):
            return False

        # Additional IF constraint checks would go here
        # For now, just check capacity
        return True

    def _calculate_insertion_cost(
        self, route: "Route", from_pos: int, to_pos: int
    ) -> float:
        """Calculate cost increase from inserting a customer"""
        if from_pos < 0 or to_pos >= len(route.nodes):
            return float("inf")

        from_node = route.nodes[from_pos]
        to_node = route.nodes[to_pos]

        return self.problem.calculate_distance(from_node, to_node)

    def _accept_solution(self, new_solution: Solution) -> bool:
        """Simulated annealing acceptance criterion"""
        cost_difference = new_solution.total_cost - self.current_solution.total_cost

        if cost_difference < 0:
            return True  # Always accept improving moves
        else:
            # Probabilistic acceptance of worse solutions
            acceptance_probability = math.exp(
                -cost_difference / max(self.temperature, 1.0)
            )
            return random.random() < acceptance_probability

    def _update_operator_scores(
        self, destroy_op: str, repair_op: str, new_solution: Solution
    ):
        """Update operator scores based on solution quality"""
        score = 0.0

        if new_solution.total_cost < self.current_solution.total_cost:
            score = 1.0  # Good improvement
        elif new_solution.total_cost == self.current_solution.total_cost:
            score = 0.5  # Neutral
        else:
            score = 0.1  # Poor

        self.destroy_scores[destroy_op] += score
        if repair_op in self.repair_scores:
            self.repair_scores[repair_op] += score

        # Notify the RepairOperatorManager (if available) so class-based
        # repair operators can adapt their internal performance measures.
        try:
            for op in getattr(self, "repair_manager", []).operators:
                # use startswith to map short/legacy names to operator.name values
                if op.name.startswith(repair_op):
                    op.update_performance(score)
                    break
        except Exception:
            # keep the update lightweight and fail-safe
            pass

        # Notify the DestroyOperatorManager (if available) to record destroy performance.
        try:
            if hasattr(self, "destroy_manager"):
                self.destroy_manager.update_operator_performance(destroy_op, score)
        except Exception:
            # fail-safe: do not interrupt ALNS when operator reporting fails
            pass

    def _update_operator_weights(self):
        """Update operator weights based on performance"""
        # Update destroy operator weights
        total_destroy_score = sum(self.destroy_scores.values())
        if total_destroy_score > 0:
            for op in self.destroy_operators:
                self.destroy_weights[op] = (
                    self.destroy_weights[op] * (1 - self.learning_rate)
                    + (self.destroy_scores[op] / total_destroy_score)
                    * self.learning_rate
                )

        # Update repair operator weights
        total_repair_score = sum(self.repair_scores.values())
        if total_repair_score > 0:
            for op in self.repair_operators:
                self.repair_weights[op] = (
                    self.repair_weights[op] * (1 - self.learning_rate)
                    + (self.repair_scores[op] / total_repair_score) * self.learning_rate
                )

        # Normalize weights
        total_destroy_weight = sum(self.destroy_weights.values())
        total_repair_weight = sum(self.repair_weights.values())

        if total_destroy_weight > 0:
            for op in self.destroy_operators:
                self.destroy_weights[op] /= total_destroy_weight

        if total_repair_weight > 0:
            for op in self.repair_operators:
                self.repair_weights[op] /= total_repair_weight

        # Reset scores for next adaptive period
        self.destroy_scores = {op: 0.0 for op in self.destroy_operators}
        self.repair_scores = {op: 0.0 for op in self.repair_operators}

        print(f"Iteration {self.iteration}: Updated operator weights")
        print(f"Destroy weights: {self.destroy_weights}")
        print(f"Repair weights: {self.repair_weights}")
