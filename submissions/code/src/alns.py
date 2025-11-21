"""
Adaptive Large Neighborhood Search (ALNS) for Vehicle Routing Problem with Intermediate Facilities
==============================================================================================

Author: Chaitanya Shinde (231070066) - Core algorithm implementation and optimization

This module implements the Adaptive Large Neighborhood Search (ALNS) algorithm for solving
the Vehicle Routing Problem with Intermediate Facilities (VRP-IF). The algorithm iteratively
destroys and repairs solutions to explore the solution space, adapting the selection of
operators based on their historical performance.

Key Components:
- ALNS class: Main orchestrator of the search process
- Adaptive weight management for operator selection
- Solution destruction and repair mechanisms
- Adaptive acceptance criteria
- Performance tracking and statistics

Algorithm Overview:
1. Generate initial solution
2. While stopping criterion not met:
   a. Select destroy and repair operators based on adaptive weights
   b. Generate new solution by applying selected operators
   c. Evaluate and accept/reject the new solution
   d. Update operator weights based on performance
   e. Update temperature for simulated annealing acceptance

The implementation focuses on:
- Correctness and reliability
- Clear separation of concerns
- Extensibility for new operators
- Deterministic behavior with fixed random seeds

Example Usage:
    >>> from src.problem import ProblemInstance
    >>> from src.alns import ALNS
    >>> problem = ProblemInstance("test")
    >>> # Configure problem instance...
    >>> solver = ALNS(problem)
    >>> best_solution = solver.run(max_iterations=1000)
"""

import random
import math
import time
from typing import Optional, List, Dict, Tuple, Any, Type, Union

from .solution import Solution, Route
from .problem import ProblemInstance, Location
from .destroy_operators import DestroyOperatorManager
from .repair_operators import (
    RepairOperatorManager,
    recalc_loads,
    enforce_if_visits,
    route_is_feasible,
)

# Type aliases for better code readability
OperatorScore = float
OperatorName = str
OperatorWeights = Dict[OperatorName, float]
OperatorScores = Dict[OperatorName, float]
OperatorCounts = Dict[OperatorName, int]
SolutionScore = float

import random
import math
import time
from typing import Optional, List, Dict

from .solution import Solution, Route
from .problem import ProblemInstance, Location
from .destroy_operators import DestroyOperatorManager
from .repair_operators import (
    RepairOperatorManager,
    recalc_loads,
    enforce_if_visits,
    route_is_feasible,
)


class ALNS:
    """
    Adaptive Large Neighborhood Search (ALNS) solver for VRP with Intermediate Facilities.
    
    This class implements the main ALNS algorithm, managing the search process,
    operator selection, and solution acceptance. It maintains the current state of
    the search, including the current solution, best solution found, and operator
    performance statistics.
    
    The algorithm uses adaptive weights to balance exploration and exploitation,
    and includes mechanisms for escaping local optima through simulated annealing.
    
    Attributes:
        problem (ProblemInstance): The problem instance to solve
        current_solution (Optional[Solution]): Current solution in the search
        best_solution (Optional[Solution]): Best solution found so far
        destroy_operators (DestroyOperatorManager): Manager for destroy operators
        repair_operators (RepairOperatorManager): Manager for repair operators
        temperature (float): Current temperature for simulated annealing
        cooling_rate (float): Rate at which temperature decreases
        max_temperature (float): Initial temperature
        min_temperature (float): Minimum temperature threshold
        iteration (int): Current iteration count
        max_iterations (int): Maximum number of iterations to run
        random_seed (Optional[int]): Random seed for reproducibility
        stats (dict): Statistics about the search process
        
    Author: Chaitanya Shinde (231070066)
    """
    
    def __init__(self, problem_instance: ProblemInstance):
        """
        Initialize the ALNS solver with a problem instance.
        
        Args:
            problem_instance (ProblemInstance): The VRP-IF instance to solve
            
        Initializes the solver with default parameters for the search process,
        including temperature settings, operator managers, and statistics tracking.
        """
        # Core problem reference
        self.problem: ProblemInstance = problem_instance
        
        # Current and best solutions found during the search
        self.current_solution: Optional[Solution] = None
        self.best_solution: Optional[Solution] = None

        # Iteration control & logging
        self.iteration: int = 0
        self.max_iterations: int = 200
        self.start_time: Optional[float] = None
        self.convergence_history: List[float] = []

        # Simulated annealing parameters for the acceptance criterion
        self.temperature: float = 1000.0
        self.cooling_rate: float = 0.995

        # Randomness control for reproducibility
        self.seed: int = 42
        random.seed(self.seed)

        # Managers for destroy and repair operators
        self.destroy_manager: DestroyOperatorManager = DestroyOperatorManager(
            self.problem
        )
        self.repair_manager: RepairOperatorManager = RepairOperatorManager()

        # Expose operator names for backward compatibility with tests
        self.destroy_operators: List[str] = list(self.destroy_manager.operators.keys())
        self.repair_operators: List[str] = [
            op.name for op in self.repair_manager.operators
        ]

        # Weights for adaptive operator selection, updated based on performance
        self.destroy_weights: Dict[str, float] = {
            name: 1.0 for name in self.destroy_operators
        }
        self.repair_weights: Dict[str, float] = {
            name: 1.0 for name in self.repair_operators
        }

        # Tracking scores & usage for each operator to adapt weights
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

        # History tracking for video generation
        self.history = []

        # Learning rate for weight adaptation
        self.learning_rate: float = 0.1
        self.adaptive_period: int = 50

    # -----------------------------
    # Public API used by the tests
    # -----------------------------
    def run(self, max_iterations: Optional[int] = None) -> Solution:
        """
        Execute the ALNS optimization process.
        
        This method runs the main ALNS optimization loop, which consists of:
        1. Initialization of the search process
        2. Main iteration loop with destroy and repair operations
        3. Adaptive operator selection and weight updates
        4. Solution acceptance and temperature updates
        5. Progress tracking and statistics collection
        
        Args:
            max_iterations (int, optional): Maximum number of iterations to run.
                If None, uses the default value set in the instance.
                
        Returns:
            Solution: The best solution found during the search.
            
        Raises:
            RuntimeError: If the optimization fails to find a feasible solution
            
        Example:
            >>> solver = ALNS(problem_instance)
            >>> best_solution = solver.run(max_iterations=1000)
            >>> print(f"Best solution cost: {best_solution.total_cost}")
            
        Author: Chaitanya Shinde (231070066)
        """
        if max_iterations:
            self.max_iterations = max_iterations

        # Setup timing, iteration count, and convergence tracking
        random.seed(self.seed)
        self.start_time = time.time()
        self.iteration = 0
        self.convergence_history = []
        self.history = []

        # Generate the initial solution using a greedy heuristic
        self.current_solution = self._generate_initial_solution()
        if self.current_solution is None:
            self.current_solution = Solution(self.problem)
        self.best_solution = self.current_solution.copy()

        # Ensure the best solution has its metrics calculated
        self.best_solution.calculate_metrics()
        if getattr(self.best_solution, "total_cost", None) is None:
            self.best_solution.total_cost = self._calculate_total_cost(
                self.best_solution
            )

        # Main optimization loop
        for it in range(self.max_iterations):
            self.iteration = it

            # Step 1: Select destroy and repair operators based on adaptive weights
            destroy_name = self._select_destroy_operator()
            repair_name = self._select_repair_operator()

            # Step 2: Apply operators to create a new candidate solution
            partial = self._destroy(self.current_solution, destroy_name)
            candidate = self._repair(partial, repair_name)

            # Ensure candidate solution has its metrics calculated
            if candidate is not None:
                candidate.calculate_metrics()
                candidate.total_cost = self._calculate_total_cost(candidate)

            # Step 3: Decide whether to accept the new solution using simulated annealing
            accept = False
            if candidate is not None:
                accept = self._accept_solution(candidate)

            if accept and candidate is not None:
                self.current_solution = candidate
                # If candidate is better than the best found so far, update best
                if candidate.total_cost < self.best_solution.total_cost:
                    self.best_solution = candidate.copy()

            # Step 4: Update operator scores and weights for adaptive learning
            self._update_operator_scores(destroy_name, repair_name, candidate)
            if it % self.adaptive_period == 0 and it > 0:
                self._update_operator_weights()

            # Decrease temperature for simulated annealing
            self.temperature *= self.cooling_rate

            # Record the cost of the best solution for convergence analysis
            try:
                self.convergence_history.append(float(self.best_solution.total_cost))
            except Exception:
                self.convergence_history.append(0.0)

            # Track solution history for video generation if enabled
            routes_for_video = []
            for route in self.best_solution.routes:
                routes_for_video.append([(node.x, node.y) for node in route.nodes])
            
            self.history.append({
                'iteration': it,
                'cost': self.current_solution.total_cost,
                'best_cost': self.best_solution.total_cost,
                'routes': routes_for_video
            })

            # Execute callback for live plotting if provided
            if callable(self.iteration_callback):
                try:
                    self.iteration_callback(it, self.best_solution)
                except Exception:
                    pass

        # Final post-processing: attempt to fill remaining unassigned customers
        try:
            if getattr(self.best_solution, "unassigned_customers", None):
                # Try applying repair operators until no unassigned left or no progress
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
            # Be tolerant of errors during this final, non-critical step
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
        """
        Determine whether to accept a new solution based on simulated annealing.
        
        This method implements the simulated annealing acceptance criterion,
        which allows the algorithm to escape local optima by sometimes accepting
        worse solutions, especially in the early stages of the search.
        
        The acceptance probability is given by:
            P(accept) = exp(-ΔE / T)
            
        Where ΔE is the cost difference and T is the current temperature.
        
        Args:
            candidate (Solution): The new solution to evaluate
            
        Returns:
            bool: True if the new solution should be accepted, False otherwise
            
        Author: Chaitanya Shinde (231070066)
        """
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
        # probabilistic acceptance for worse solutions
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

        # reset scores for the next adaptation period
        self.destroy_scores = {k: 0.0 for k in self.destroy_scores}
        self.repair_scores = {k: 0.0 for k in self.repair_scores}

    def _generate_initial_solution(self) -> Solution:
        """
        Generates an initial solution using a simple greedy heuristic.
        This method provides a starting point for the ALNS algorithm.
        It creates a basic, feasible solution by inserting customers one by one.
        """
        # Create a new empty solution for the problem
        solution = Solution(self.problem)
        
        # Get a list of all customers that need to be assigned to routes
        unassigned_customers = list(self.problem.customers)
        
        # Continue until all customers are assigned
        while unassigned_customers:
            # Create a new route starting from the depot
            route = Route()
            route.nodes = [self.problem.depot]
            current_load = 0
            
            # Greedily add customers to the current route
            for customer in list(unassigned_customers):
                # Check if adding the customer exceeds vehicle capacity
                if current_load + customer.demand <= self.problem.vehicle_capacity:
                    # Add customer to the route and update load
                    route.nodes.append(customer)
                    current_load += customer.demand
                    unassigned_customers.remove(customer)
            
            # Complete the route by returning to the depot
            route.nodes.append(self.problem.depot)
            
            # Add the newly created route to the solution
            solution.routes.append(route)
            
            # This customer is now assigned
            if customer in solution.unassigned_customers:
                solution.unassigned_customers.remove(customer)

        # Calculate all metrics for the initial solution
        solution.calculate_metrics()
        return solution