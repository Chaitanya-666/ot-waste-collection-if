"""
ALNS implementation (clean, self-contained) for VRP with Intermediate Facilities.

This implementation is intentionally conservative and focuses on providing the
interfaces and basic behavior the test-suite expects:

- Class `ALNS` with methods:
    - __init__(problem_instance)
    - _generate_initial_solution()
    - run(max_iterations=None)
  and helpers used internally by the tests.

- Uses the existing manager classes from `destroy_operators` and `repair_operators`.
- Produces deterministic behavior when a `seed` is set.

Notes:
- This file is written to be compatible with tests that import modules
  by name (the test harness adds `src/` to PYTHONPATH). Therefore imports
  are top-level (e.g., `from solution import Solution`).
- The emphasis is correct external behavior and stability, not maximal
  algorithmic performance.
"""

import random
import math
import time
from typing import Optional, List, Dict

from solution import Solution, Route
from problem import ProblemInstance, Location
from destroy_operators import DestroyOperatorManager
from repair_operators import (
    RepairOperatorManager,
    recalc_loads,
    enforce_if_visits,
    route_is_feasible,
)


class ALNS:
    def __init__(self, problem_instance: ProblemInstance):
        # Core problem reference
        self.problem: ProblemInstance = problem_instance

        # Current and best solutions
        self.current_solution: Optional[Solution] = None
        self.best_solution: Optional[Solution] = None

        # Iteration control & logging
        self.iteration: int = 0
        self.max_iterations: int = 200
        self.start_time: Optional[float] = None
        self.convergence_history: List[float] = []

        # Simulated annealing / acceptance parameters (simple defaults)
        self.temperature: float = 1000.0
        self.cooling_rate: float = 0.995

        # Randomness
        self.seed: int = 42
        random.seed(self.seed)

        # Managers
        self.destroy_manager: DestroyOperatorManager = DestroyOperatorManager(
            self.problem
        )
        self.repair_manager: RepairOperatorManager = RepairOperatorManager()

        # Expose operator names for backward compatibility with tests
        self.destroy_operators: List[str] = list(self.destroy_manager.operators.keys())
        self.repair_operators: List[str] = [
            op.name for op in self.repair_manager.operators
        ]

        # Weights: simple equal initialization
        self.destroy_weights: Dict[str, float] = {
            name: 1.0 for name in self.destroy_operators
        }
        self.repair_weights: Dict[str, float] = {
            name: 1.0 for name in self.repair_operators
        }

        # Tracking scores & usage
        self.destroy_scores: Dict[str, float] = {
            name: 0.0 for name in self.destroy_operators
        }
        self.repair_scores: Dict[str, float] = {
            name: 0.0 for name in self.repair_operators
        }
        self.destroy_usage: Dict[str, int] = {
            name: 0 for name in self.destroy_operators
        }
        self.repair_usage: Dict[str, int] = {name: 0 for name in self.repair_operators}

        # Optional callback for live plotting (kept for compatibility)
        self.iteration_callback = None

        # Learning / adaptation
        self.learning_rate: float = 0.1
        self.adaptive_period: int = 50

    # -----------------------------
    # Public API used by the tests
    # -----------------------------
    def _generate_initial_solution(self) -> Solution:
        """
        Greedy nearest-neighbour style initializer.

        Creates routes that start and end at the depot and inserts customers
        until capacity would be exceeded, then starts a new route. Ensures
        IFs are inserted when necessary using `enforce_if_visits`.
        """
        sol = Solution(self.problem)

        # quick defensive check
        if not getattr(self.problem, "depot", None):
            # create an empty solution (tests will handle feasibility)
            sol.unassigned_customers = set(c.id for c in self.problem.customers)
            return sol

        # copy list of customers
        remaining = [c for c in self.problem.customers]

        # Start with a single empty route (depot present)
        def make_empty_route() -> Route:
            r = Route()
            r.nodes = [self.problem.depot, self.problem.depot]
            r.loads = [0.0, 0.0]
            return r

        if not remaining:
            sol.routes = []
            sol.unassigned_customers = set()
            sol.calculate_metrics()
            return sol

        sol.routes = [make_empty_route()]

        # Greedy assignment: pick nearest customer to last location
        current_route = sol.routes[0]
        last_loc = self.problem.depot

        while remaining:
            # pick nearest remaining customer
            nearest = min(
                remaining, key=lambda c: self.problem.calculate_distance(last_loc, c)
            )
            # compute current load on route
            cur_load = sum(
                getattr(n, "demand", 0.0)
                for n in current_route.nodes
                if getattr(n, "type", None) == "customer"
            )
            if (
                cur_load + float(getattr(nearest, "demand", 0.0))
                > self.problem.vehicle_capacity
            ):
                # finalize current route and start a new one
                if current_route.nodes[-1] != self.problem.depot:
                    current_route.nodes.append(self.problem.depot)
                current_route.calculate_metrics(self.problem)
                current_route = make_empty_route()
                sol.routes.append(current_route)
                last_loc = self.problem.depot
                continue

            # insert nearest before final depot
            insert_pos = len(current_route.nodes) - 1
            current_route.nodes.insert(insert_pos, nearest)
            # update loads and enforce IF visits if needed
            current_route.loads = recalc_loads(current_route)
            # attempt to enforce IFs; if enforcement fails, roll back and start new route
            ok = True
            try:
                ok = enforce_if_visits(current_route, self.problem)
            except Exception:
                ok = False

            if not ok:
                # rollback insertion
                current_route.nodes.pop(insert_pos)
                current_route.loads = recalc_loads(current_route)
                # close route and start a new one
                if current_route.nodes[-1] != self.problem.depot:
                    current_route.nodes.append(self.problem.depot)
                current_route.calculate_metrics(self.problem)
                current_route = make_empty_route()
                sol.routes.append(current_route)
                last_loc = self.problem.depot
                continue

            # success: finalize route recalculation
            current_route.calculate_metrics(self.problem)
            last_loc = nearest
            remaining.remove(nearest)

        # finalize routes
        for r in sol.routes:
            if not r.nodes:
                r.nodes = [self.problem.depot, self.problem.depot]
            if r.nodes[0] != self.problem.depot:
                r.nodes.insert(0, self.problem.depot)
            if r.nodes[-1] != self.problem.depot:
                r.nodes.append(self.problem.depot)
            r.loads = recalc_loads(r)
            r.calculate_metrics(self.problem)

        # build unassigned set (should be empty)
        assigned_ids = set()
        for r in sol.routes:
            for n in r.nodes:
                if getattr(n, "type", None) == "customer":
                    assigned_ids.add(getattr(n, "id", None))
        all_ids = set(c.id for c in self.problem.customers)
        sol.unassigned_customers = all_ids - assigned_ids

        # aggregated metrics
        sol.calculate_metrics()

        return sol

    def run(self, max_iterations: Optional[int] = None) -> Solution:
        """
        Main ALNS loop.

        The implementation uses the DestroyOperatorManager and RepairOperatorManager
        to perform destroy and repair steps. The acceptance criterion is a simple
        simulated annealing rule. The method returns the best solution found.
        """
        if max_iterations:
            self.max_iterations = max_iterations

        # Setup
        random.seed(self.seed)
        self.start_time = time.time()
        self.iteration = 0
        self.convergence_history = []

        # initial solution
        self.current_solution = self._generate_initial_solution()
        if self.current_solution is None:
            self.current_solution = Solution(self.problem)
        self.best_solution = self.current_solution.copy()

        # ensure best has metrics
        self.best_solution.calculate_metrics()
        if getattr(self.best_solution, "total_cost", None) is None:
            self.best_solution.total_cost = self._calculate_total_cost(
                self.best_solution
            )

        # main loop
        for it in range(self.max_iterations):
            self.iteration = it

            # select operators
            destroy_name = self._select_destroy_operator()
            repair_name = self._select_repair_operator()

            # apply destroy -> repair
            partial = self._destroy(self.current_solution, destroy_name)
            candidate = self._repair(partial, repair_name)

            # ensure candidate metrics
            if candidate is not None:
                candidate.calculate_metrics()
                candidate.total_cost = self._calculate_total_cost(candidate)

            # acceptance
            accept = False
            if candidate is not None:
                accept = self._accept_solution(candidate)

            if accept and candidate is not None:
                self.current_solution = candidate
                # update best
                if candidate.total_cost < self.best_solution.total_cost:
                    self.best_solution = candidate.copy()

            # tracking & adaptation
            self._update_operator_scores(destroy_name, repair_name, candidate)
            if it % self.adaptive_period == 0 and it > 0:
                self._update_operator_weights()

            # cooling
            self.temperature *= self.cooling_rate

            # record convergence
            try:
                self.convergence_history.append(float(self.best_solution.total_cost))
            except Exception:
                self.convergence_history.append(0.0)

            # optional callback
            if callable(self.iteration_callback):
                try:
                    self.iteration_callback(it, self.best_solution)
                except Exception:
                    pass

        # final post-processing: attempt to fill remaining unassigned customers using repair manager
        try:
            if getattr(self.best_solution, "unassigned_customers", None):
                # try applying repair operators until no unassigned left or no progress
                prev_unassigned = -1
                attempts = 0
                while (
                    getattr(self.best_solution, "unassigned_customers", set())
                    and attempts < 10
                ):
                    op = self.repair_manager.select()
                    self.best_solution = op.apply(self.best_solution)
                    self.best_solution.calculate_metrics()
                    attempts += 1
                    if (
                        len(getattr(self.best_solution, "unassigned_customers", set()))
                        == prev_unassigned
                    ):
                        break
                    prev_unassigned = len(
                        getattr(self.best_solution, "unassigned_customers", set())
                    )
        except Exception:
            # tolerant of errors here
            pass

        # Ensure final solution metrics are consistent
        try:
            self.best_solution.calculate_metrics()
            self.best_solution.total_cost = self._calculate_total_cost(
                self.best_solution
            )
        except Exception:
            pass

        return self.best_solution

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _select_destroy_operator(self) -> str:
        """Select destroy operator name using manager weights if possible."""
        try:
            # manager provides select_operator() -> name
            if hasattr(self.destroy_manager, "select_operator"):
                return self.destroy_manager.select_operator()
        except Exception:
            pass

        # fallback to weighted random choice
        names = list(self.destroy_weights.keys())
        weights = [self.destroy_weights[n] for n in names]
        total = sum(weights)
        if total <= 0:
            return random.choice(names)
        probs = [w / total for w in weights]
        return random.choices(names, weights=probs, k=1)[0]

    def _select_repair_operator(self) -> str:
        """Select repair operator name based on internal weights (returns a name string)."""
        # try to map to one of the repair operator names
        names = list(self.repair_weights.keys())
        weights = [self.repair_weights[n] for n in names]
        total = sum(weights)
        if total <= 0:
            return random.choice(names)
        probs = [w / total for w in weights]
        return random.choices(names, weights=probs, k=1)[0]

    def _destroy(self, solution: Solution, operator_name: str) -> Solution:
        """Apply a destroy operator to produce a partial solution."""
        # removal degree: simple fraction of customers (at least 1)
        removal_count = max(1, int(max(1, len(self.problem.customers)) * 0.15))

        # Preferred API: manager.apply_operator(solution, operator_name, removal_count)
        try:
            if hasattr(self.destroy_manager, "apply_operator"):
                return self.destroy_manager.apply_operator(
                    solution.copy(), operator_name, removal_count
                )
        except Exception:
            pass

        # fallback: remove randomly chosen customers
        partial = solution.copy()
        # naive random removal
        all_customers = [
            n
            for r in partial.routes
            for n in r.nodes
            if getattr(n, "type", None) == "customer"
        ]
        if not all_customers:
            return partial
        to_remove = random.sample(all_customers, min(len(all_customers), removal_count))
        for cust in to_remove:
            for r in partial.routes:
                if cust in r.nodes:
                    r.nodes.remove(cust)
                    break
            if not hasattr(partial, "unassigned_customers"):
                partial.unassigned_customers = set()
            partial.unassigned_customers.add(getattr(cust, "id", cust))
        # cleanup empty routes
        partial.routes = [r for r in partial.routes if len(r.nodes) > 1]
        return partial

    def _repair(self, partial_solution: Solution, operator_name: str) -> Solution:
        """Apply a repair operator to rebuild a complete solution from a partial one."""
        # try to find a matching operator in repair_manager by prefix
        try:
            for op in self.repair_manager.operators:
                if op.name.startswith(operator_name):
                    return op.apply(partial_solution.copy())
        except Exception:
            pass

        # fallback: ask manager to select and apply an operator
        try:
            op = self.repair_manager.select()
            return op.apply(partial_solution.copy())
        except Exception:
            # last-resort: return the partial_solution unchanged
            return partial_solution.copy()

    def _calculate_total_cost(self, solution: Solution) -> float:
        """Total distance + heavy penalty for unassigned customers"""
        total = 0.0
        for r in getattr(solution, "routes", []):
            for i in range(len(r.nodes) - 1):
                total += self.problem.calculate_distance(r.nodes[i], r.nodes[i + 1])
        # penalty for unassigned
        penalty_per_unassigned = 1000.0
        n_unassigned = len(getattr(solution, "unassigned_customers", set()) or set())
        total += penalty_per_unassigned * float(n_unassigned)
        return total

    def _accept_solution(self, candidate: Solution) -> bool:
        """Simple simulated annealing acceptance."""
        # if current_solution missing, accept candidate
        if self.current_solution is None:
            return True
        try:
            curr_cost = float(
                getattr(
                    self.current_solution,
                    "total_cost",
                    self._calculate_total_cost(self.current_solution),
                )
            )
            cand_cost = float(
                getattr(candidate, "total_cost", self._calculate_total_cost(candidate))
            )
        except Exception:
            return True

        if cand_cost < curr_cost:
            return True
        # probabilistic acceptance
        try:
            prob = math.exp(-(cand_cost - curr_cost) / max(self.temperature, 1e-6))
            return random.random() < prob
        except Exception:
            return False

    def _update_operator_scores(
        self, destroy_name: str, repair_name: str, candidate: Optional[Solution]
    ):
        """Update scoring for operators based on whether candidate improved best solution."""
        score = 0.0
        try:
            cand_cost = float(getattr(candidate, "total_cost", float("inf")))
            curr_cost = float(
                getattr(self.current_solution, "total_cost", float("inf"))
            )
            # positive score if candidate improves current
            if cand_cost < curr_cost:
                score = 1.0
            elif cand_cost == curr_cost:
                score = 0.5
            else:
                score = 0.0
        except Exception:
            score = 0.0

        # update dictionaries if keys exist
        if destroy_name in self.destroy_scores:
            self.destroy_scores[destroy_name] += score
            self.destroy_usage[destroy_name] += 1
        if repair_name in self.repair_scores:
            self.repair_scores[repair_name] += score
            self.repair_usage[repair_name] += 1

    def _update_operator_weights(self):
        """Normalize and mix in performance into weights (simple scheme)."""
        # destroy weights
        total_destroy_score = sum(self.destroy_scores.values()) or 0.0
        if total_destroy_score > 0:
            for k in self.destroy_weights:
                # combine previous weight (decay) with normalized score
                normalized = (
                    (self.destroy_scores.get(k, 0.0) / total_destroy_score)
                    if total_destroy_score > 0
                    else 0.0
                )
                self.destroy_weights[k] = (
                    1 - self.learning_rate
                ) * self.destroy_weights.get(k, 1.0) + self.learning_rate * normalized

        # repair weights
        total_repair_score = sum(self.repair_scores.values()) or 0.0
        if total_repair_score > 0:
            for k in self.repair_weights:
                normalized = (
                    (self.repair_scores.get(k, 0.0) / total_repair_score)
                    if total_repair_score > 0
                    else 0.0
                )
                self.repair_weights[k] = (
                    1 - self.learning_rate
                ) * self.repair_weights.get(k, 1.0) + self.learning_rate * normalized

        # normalize to sum == 1
        sd = sum(self.destroy_weights.values()) or 1.0
        sr = sum(self.repair_weights.values()) or 1.0
        for k in self.destroy_weights:
            self.destroy_weights[k] = max(0.0, self.destroy_weights[k]) / sd
        for k in self.repair_weights:
            self.repair_weights[k] = max(0.0, self.repair_weights[k]) / sr

        # reset scores
        self.destroy_scores = {k: 0.0 for k in self.destroy_scores}
        self.repair_scores = {k: 0.0 for k in self.repair_scores}
