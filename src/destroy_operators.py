"""
Destroy Operators for Adaptive Large Neighborhood Search (ALNS)
=============================================================

Author: Chaitanya Shinde (231070066) - Core algorithm implementation

This module implements the destroy operators for the Adaptive Large Neighborhood
Search (ALNS) algorithm used to solve the Vehicle Routing Problem with
Intermediate Facilities (VRP-IF). Destroy operators are responsible for taking
a complete solution and removing a subset of customers to create a partial
solution, which can then be "repaired" in a new way.

Key Features:
- Multiple destroy strategies with different characteristics
- Adaptive operator selection based on performance
- Support for both random and guided destruction
- Efficient implementation for large problem instances

Classes:
    - DestroyOperator: Base class for all destroy operators
    - RandomRemoval: Removes customers randomly
    - WorstRemoval: Removes most expensive customers to serve
    - ShawRemoval: Removes similar customers based on multiple criteria
    - RouteRemoval: Removes entire routes
    - RelatedRemoval: Removes geographically clustered customers
    - DestroyOperatorManager: Manages operator selection and application
"""

import random
import math
from typing import List, Set, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field

from .solution import Solution, Route
from .problem import ProblemInstance, Location


class DestroyOperator:
    """
    Abstract base class for all destroy operators in the ALNS algorithm.
    
    Destroy operators take a complete solution and remove a subset of customers,
    creating a partial solution that needs to be repaired. This allows the
    algorithm to explore different regions of the solution space.
    
    Attributes:
        name (str): Identifier for the operator
        performance_score (float): Tracks the operator's performance
        usage_count (int): Number of times the operator has been used
        
    Note:
        - Subclasses must implement the `apply` method
        - Performance tracking is used for adaptive operator selection
        
    Author: Chaitanya Shinde (231070066)
    """
    
    def __init__(self, name: str):
        """
        Initialize a new destroy operator.
        
        Args:
            name: Unique identifier for the operator
            
        Note:
            - Initializes performance tracking attributes
        """

    def __init__(self, name: str):
        self.name = name
        # Performance score is used for adaptive weight adjustment.
        self.performance_score = 0.0
        self.usage_count = 0

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """
        Apply the destroy operator to a solution.
        
        Args:
            solution: The current solution to be modified
            removal_count: Number of customers to remove
            
        Returns:
            Solution: A new solution with customers removed
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass
            
        Note:
            - Must be implemented by all concrete subclasses
            - Should not modify the original solution
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def get_performance_score(self) -> float:
        """
        Calculate the average performance score of this operator.
        
        Returns:
            float: The average score, or 0 if the operator hasn't been used
            
        Note:
            - Used by the adaptive weight adjustment mechanism
            - Higher scores indicate better performance
        """
        if self.usage_count == 0:
            return 0.0
        return self.performance_score / self.usage_count

    def update_performance(self, score: float) -> None:
        """
        Update the operator's performance score.
        
        Args:
            score: Performance score from the last run (higher is better)
            
        Note:
            - Called by the ALNS algorithm after each iteration
            - Used to adaptively adjust operator selection probabilities
        """
        self.performance_score += score
        self.usage_count += 1


class RandomRemoval(DestroyOperator):
    """
    Random Removal Destroy Operator
    
    Removes a random selection of customers from the solution. This operator
    helps to diversify the search by introducing random changes that allow
    the algorithm to escape local optima.
    
    Key Features:
    - Completely random selection of customers
    - Preserves solution structure while allowing exploration
    - Effective at maintaining diversity in the search
    
    Example:
        >>> operator = RandomRemoval()
        >>> partial_solution = operator.apply(solution, removal_count=5)
        
    Author: Chaitanya Shinde (231070066)
    """

    def __init__(self):
        super().__init__("random_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Removes `removal_count` customers from the solution randomly."""
        partial_solution = solution.copy()

        # Get a list of all customers currently in routes.
        all_customers = []
        for route in partial_solution.routes:
            for node in route.nodes:
                if node.type == "customer":
                    all_customers.append((route, node))

        if len(all_customers) <= removal_count:
            # If the requested removal count is too high, remove all customers.
            removed_customers = []
            for route, customer in all_customers:
                removed_customers.append(customer)
                route.nodes.remove(customer)
            # Mark removed customers as unassigned.
            if not hasattr(partial_solution, "unassigned_customers"):
                partial_solution.unassigned_customers = set()
            for cust in removed_customers:
                partial_solution.unassigned_customers.add(cust.id)
            return self._clean_solution(partial_solution)

        # Randomly select customers to remove.
        customers_to_remove = random.sample(
            all_customers, min(removal_count, len(all_customers))
        )

        removed_customers = []
        for route, customer in customers_to_remove:
            removed_customers.append(customer)
            route.nodes.remove(customer)

        # Add the removed customers' IDs to the unassigned set.
        if not hasattr(partial_solution, "unassigned_customers"):
            partial_solution.unassigned_customers = set()
        for cust in removed_customers:
            partial_solution.unassigned_customers.add(cust.id)

        return self._clean_solution(partial_solution)

    def _clean_solution(self, solution: Solution) -> Solution:
        """Cleans up the solution by recalculating metrics and removing empty routes."""
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)
        # A route is considered empty if it only contains the depot.
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]
        return solution

    def _recalculate_loads(self, route: Route) -> List[float]:
        """Recalculates the cumulative load for a given route."""
        loads: List[float] = [0.0]
        current_load: float = 0.0
        for node in route.nodes[1:]:  # Skip initial depot
            if getattr(node, "type", None) == "customer":
                current_load += float(getattr(node, "demand", 0.0))
            loads.append(current_load)
        return loads


class WorstRemoval(DestroyOperator):
    """
    Worst Removal Destroy Operator
    
    Removes customers that are the most expensive to serve, where cost is
    calculated as the marginal increase in route distance if the customer
    were removed. This operator focuses on identifying and removing
    inefficient parts of the solution.
    
    Key Features:
    - Targets the most expensive customers first
    - Uses a roulette wheel selection to balance greediness and diversity
    - Helps to improve solution quality by focusing on problematic areas
    
    Example:
        >>> operator = WorstRemoval()
        >>> partial_solution = operator.apply(solution, removal_count=5)
        
    Author: Chaitanya Shinde (231070066)
    """

    def __init__(self):
        super().__init__("worst_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Removes `removal_count` customers with the highest marginal cost."""
        partial_solution = solution.copy()

        # Calculate the marginal cost for each customer in the solution.
        customer_costs = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer":
                    marginal_cost = self._calculate_marginal_cost(
                        route, node_idx, partial_solution
                    )
                    customer_costs.append((route, node_idx, marginal_cost))

        if len(customer_costs) <= removal_count:
            # If requested count is too high, remove all customers.
            removed_customers = []
            for route, node_idx, _ in customer_costs:
                if 0 <= node_idx < len(route.nodes):
                    removed_customers.append(route.nodes[node_idx])
                    del route.nodes[node_idx]
            if not hasattr(partial_solution, "unassigned_customers"):
                partial_solution.unassigned_customers = set()
            for cust in removed_customers:
                partial_solution.unassigned_customers.add(cust.id)
            return self._clean_solution(partial_solution)

        # Sort customers by their removal cost in descending order.
        customer_costs.sort(key=lambda x: x[2], reverse=True)

        removed_customers = []
        for route, node_idx, _ in customer_costs[:removal_count]:
            if 0 <= node_idx < len(route.nodes):
                removed_customers.append(route.nodes[node_idx])
                del route.nodes[node_idx]

        if not hasattr(partial_solution, "unassigned_customers"):
            partial_solution.unassigned_customers = set()
        for cust in removed_customers:
            partial_solution.unassigned_customers.add(cust.id)

        return self._clean_solution(partial_solution)

    def _calculate_marginal_cost(
        self, route: Route, node_idx: int, solution: Solution
    ) -> float:
        """Calculates the change in route cost if the customer at `node_idx` is removed."""
        if node_idx <= 0 or node_idx >= len(route.nodes) - 1:
            return 0.0

        # Cost of going from previous node to customer, and customer to next node.
        from_node = route.nodes[node_idx - 1]
        customer = route.nodes[node_idx]
        to_node = route.nodes[node_idx + 1]
        original_cost = solution.problem.calculate_distance(from_node, customer) + solution.problem.calculate_distance(customer, to_node)

        # Cost of going directly from previous node to next node.
        new_cost = solution.problem.calculate_distance(from_node, to_node)

        # The marginal cost is the savings achieved by removing the customer.
        return original_cost - new_cost

    def _clean_solution(self, solution: Solution) -> Solution:
        """Cleans up the solution after removal."""
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]
        return solution

    def _recalculate_loads(self, route: Route) -> List[float]:
        """Recalculates the cumulative load for a given route."""
        loads: List[float] = [0.0]
        current_load: float = 0.0
        for node in route.nodes[1:]:
            if getattr(node, "type", None) == "customer":
                current_load += float(getattr(node, "demand", 0.0))
            loads.append(current_load)
        return loads


class ShawRemoval(DestroyOperator):
    """
    Shaw Removal Destroy Operator
    
    Removes customers that are "similar" based on multiple criteria including
    geographic proximity, demand, and service time. This operator is based on
    the work of Shaw (1998) and is effective at identifying related customers
    that might be better served together.
    
    Key Features:
    - Multi-criteria similarity measure
    - Balances between relatedness and randomness
    - Effective for clustered problem instances
    
    Reference:
        Shaw, P. (1998). Using constraint programming and local search
        methods to solve vehicle routing problems. CP-98, 417-431.
        
    Example:
        >>> operator = ShawRemoval()
        >>> partial_solution = operator.apply(solution, removal_count=5)
        
    Author: Chaitanya Shinde (231070066)
    """

    def __init__(self):
        super().__init__("shaw_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Removes a set of similar customers."""
        partial_solution = solution.copy()

        all_customers = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer":
                    all_customers.append((route, node_idx, node))

        if len(all_customers) <= removal_count:
            # Remove all if not enough customers.
            removed_customers = []
            for route, node_idx, _ in all_customers:
                if 0 <= node_idx < len(route.nodes):
                    removed_customers.append(route.nodes[node_idx])
                    del route.nodes[node_idx]
            if not hasattr(partial_solution, "unassigned_customers"):
                partial_solution.unassigned_customers = set()
            for cust in removed_customers:
                partial_solution.unassigned_customers.add(cust.id)
            return self._clean_solution(partial_solution)

        # Select a random customer as a "seed" to find similar customers.
        seed_route, seed_idx, seed_customer = random.choice(all_customers)

        # Calculate similarity of all other customers to the seed.
        similarities = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer" and (
                    route != seed_route or node_idx != seed_idx
                ):
                    similarity = self._calculate_similarity(
                        seed_customer, node, solution.problem
                    )
                    similarities.append((route, node_idx, similarity))

        # Sort by similarity and remove the most similar customers.
        similarities.sort(key=lambda x: x[2], reverse=True)

        removed_customers = []
        removed_count = 0
        for route, node_idx, similarity in similarities:
            if removed_count >= removal_count:
                break
            if 0 <= node_idx < len(route.nodes):
                removed_customers.append(route.nodes[node_idx])
                del route.nodes[node_idx]
                removed_count += 1

        if not hasattr(partial_solution, "unassigned_customers"):
            partial_solution.unassigned_customers = set()
        for cust in removed_customers:
            partial_solution.unassigned_customers.add(cust.id)

        return self._clean_solution(partial_solution)

    def _calculate_similarity(
        self, customer1: Location, customer2: Location, problem: ProblemInstance
    ) -> float:
        """Calculates a similarity score between two customers."""
        # Similarity is higher for customers that are closer and have similar demand.
        distance = problem.calculate_distance(customer1, customer2)
        demand_diff = abs(customer1.demand - customer2.demand)
        # The formula can be tuned to give more weight to distance or demand.
        similarity = 1.0 / (1.0 + distance + demand_diff)
        return similarity

    def _clean_solution(self, solution: Solution) -> Solution:
        """Cleans up the solution after removal."""
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]
        return solution

    def _recalculate_loads(self, route: Route) -> List[float]:
        """Recalculates the cumulative load for a given route."""
        loads: List[float] = [0.0]
        current_load: float = 0.0
        for node in route.nodes[1:]:
            if getattr(node, "type", None) == "customer":
                current_load += float(getattr(node, "demand", 0.0))
            loads.append(current_load)
        return loads


class RouteRemoval(DestroyOperator):
    """
    Route Removal Destroy Operator
    
    Removes one or more entire routes from the solution. This is a large-scale
    operator that can lead to significant changes in the solution structure,
    making it particularly effective for escaping deep local optima.
    
    Key Features:
    - Removes complete routes rather than individual customers
    - More disruptive than customer-level operators
    - Effective for exploring different route structures
    
    Example:
        >>> operator = RouteRemoval()
        >>> partial_solution = operator.apply(solution, removal_count=2)
        
    Author: Chaitanya Shinde (231070066)
    """

    def __init__(self):
        super().__init__("route_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Removes `removal_count` routes from the solution."""
        partial_solution = solution.copy()

        if len(partial_solution.routes) <= removal_count:
            # If requested count is too high, remove all routes.
            removed_customers = []
            for r in partial_solution.routes:
                for n in r.nodes:
                    if n.type == "customer":
                        removed_customers.append(n)
            partial_solution.routes = []
            if not hasattr(partial_solution, "unassigned_customers"):
                partial_solution.unassigned_customers = set()
            for cust in removed_customers:
                partial_solution.unassigned_customers.add(cust.id)
            return partial_solution

        # Select random routes to remove.
        routes_to_remove = random.sample(
            partial_solution.routes, min(removal_count, len(partial_solution.routes))
        )

        removed_customers = []
        for route in routes_to_remove:
            for n in route.nodes:
                if n.type == "customer":
                    removed_customers.append(n)
            partial_solution.routes.remove(route)

        if not hasattr(partial_solution, "unassigned_customers"):
            partial_solution.unassigned_customers = set()
        for cust in removed_customers:
            partial_solution.unassigned_customers.add(cust.id)

        return partial_solution


class RelatedRemoval(DestroyOperator):
    """
    Related Removal Destroy Operator
    
    Removes customers that are geographically clustered together, focusing
    specifically on spatial relationships. This operator is similar to Shaw
    removal but with a stronger emphasis on geographic proximity.
    
    Key Features:
    - Focuses on spatial clustering of customers
    - Uses a distance-based similarity measure
    - Effective for problems with strong geographic patterns
    
    Example:
        >>> operator = RelatedRemoval()
        >>> partial_solution = operator.apply(solution, removal_count=5)
        
    Author: Chaitanya Shinde (231070066)
    """

    def __init__(self):
        super().__init__("related_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Removes a cluster of geographically related customers."""
        partial_solution = solution.copy()

        all_customers = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer":
                    all_customers.append((route, node_idx, node))

        if len(all_customers) <= removal_count:
            # Remove all if not enough customers.
            removed_customers = []
            for route, node_idx, _ in all_customers:
                if 0 <= node_idx < len(route.nodes):
                    removed_customers.append(route.nodes[node_idx])
                    del route.nodes[node_idx]
            if not hasattr(partial_solution, "unassigned_customers"):
                partial_solution.unassigned_customers = set()
            for cust in removed_customers:
                partial_solution.unassigned_customers.add(cust.id)
            return self._clean_solution(partial_solution)

        # Find clusters of related customers based on distance.
        clusters = self._find_customer_clusters(all_customers, solution.problem)

        # Select the largest cluster and remove customers from it.
        if clusters:
            largest_cluster = max(clusters, key=len)
            customers_to_remove = largest_cluster[:removal_count]

            removed_customers = []
            for route, node_idx, _ in customers_to_remove:
                if 0 <= node_idx < len(route.nodes):
                    removed_customers.append(route.nodes[node_idx])
                    del route.nodes[node_idx]

            if not hasattr(partial_solution, "unassigned_customers"):
                partial_solution.unassigned_customers = set()
            for cust in removed_customers:
                partial_solution.unassigned_customers.add(cust.id)

        return self._clean_solution(partial_solution)

    def _find_customer_clusters(
        self, customers: List, problem: ProblemInstance
    ) -> List[List]:
        """Finds clusters of related customers using a simple distance-based approach."""
        clusters = []
        visited = set()

        for i, (route1, idx1, customer1) in enumerate(customers):
            if i in visited:
                continue

            # Start a new cluster with the current customer.
            cluster = [(route1, idx1, customer1)]
            visited.add(i)

            # Find other customers related to the current one.
            for j, (route2, idx2, customer2) in enumerate(customers):
                if j in visited:
                    continue

                distance = problem.calculate_distance(customer1, customer2)
                demand_diff = abs(customer1.demand - customer2.demand)

                # Define a threshold for what "related" means. These thresholds can be tuned.
                if distance < 50.0 and demand_diff < 5:
                    cluster.append((route2, idx2, customer2))
                    visited.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _clean_solution(self, solution: Solution) -> Solution:
        """Cleans up the solution after removal."""
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]
        return solution

    def _recalculate_loads(self, route: Route) -> List[float]:
        """Recalculates the cumulative load for a given route."""
        loads: List[float] = [0.0]
        current_load: float = 0.0
        for node in route.nodes[1:]:
            if getattr(node, "type", None) == "customer":
                current_load += float(getattr(node, "demand", 0.0))
            loads.append(current_load)
        return loads


class DestroyOperatorManager:
    """
    Destroy Operator Manager
    
    Manages the selection and application of destroy operators using an adaptive
    weighting mechanism. The manager tracks the performance of each operator and
    adjusts their selection probabilities accordingly.
    
    Key Features:
    - Maintains a set of destroy operators
    - Implements adaptive weight adjustment
    - Provides operator selection based on performance
    
    Attributes:
        problem: The problem instance being solved
        operators: Dictionary of available destroy operators
        weights: Current selection weights for each operator
        
    Example:
        >>> manager = DestroyOperatorManager(problem)
        >>> operator = manager.select_operator()
        >>> solution = manager.apply_operator(solution, operator, 5)
        
    Author: Chaitanya Shinde (231070066)
    """

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.operators = {
            "random": RandomRemoval(),
            "worst": WorstRemoval(),
            "shaw": ShawRemoval(),
            "route": RouteRemoval(),
            "related": RelatedRemoval(),
        }
        # All operators start with an equal weight.
        self.weights = {name: 1.0 for name in self.operators.keys()}

    def select_operator(self) -> str:
        """
        Select a destroy operator using roulette wheel selection.
        
        Returns:
            str: The name of the selected operator
            
        Note:
            - Uses roulette wheel selection based on operator weights
            - Higher weight means higher probability of selection
        """
        operators = list(self.operators.keys())
        weights = [self.weights[op] for op in operators]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(operators)

        probabilities = [w / total_weight for w in weights]
        return random.choices(operators, weights=probabilities)[0]

    def apply_operator(
        self, solution: Solution, operator_name: str, removal_count: int
    ) -> Solution:
        """
        Apply a specific destroy operator to a solution.
        
        Args:
            solution: The solution to modify
            operator_name: Name of the operator to apply
            removal_count: Number of customers to remove
            
        Returns:
            Solution: A new solution with customers removed
            
        Raises:
            ValueError: If the operator name is not recognized
        """
        if operator_name not in self.operators:
            raise ValueError(f"Unknown operator: {operator_name}")

        operator = self.operators[operator_name]
        return operator.apply(solution, removal_count)

    def update_operator_performance(self, operator_name: str, score: float) -> None:
        """
        Update the performance score of a specific operator.
        
        Args:
            operator_name: Name of the operator to update
            score: Performance score (higher is better)
            
        Note:
            - Called by the ALNS algorithm after each iteration
            - Affects future operator selection probabilities
        """
        if operator_name in self.operators:
            self.operators[operator_name].update_performance(score)

    def get_operator_performance(self) -> Dict[str, float]:
        """
        Get the performance scores for all operators.
        
        Returns:
            Dict[str, float]: Dictionary mapping operator names to their scores
        """
        return {name: op.get_performance_score() for name, op in self.operators.items()}

    def update_weights(self) -> None:
        """
        Update the selection weights of all operators.
        
        Note:
            - Weights are updated based on recent performance
            - Includes a reaction factor to control adaptation speed
            - Maintains minimum weight to ensure all operators have a chance
        """
        performances = self.get_operator_performance()
        total_performance = sum(performances.values())

        if total_performance > 0:
            for name in self.operators.keys():
                self.weights[name] = performances[name] / total_performance
        else:
            # If there's no performance data, reset to equal weights.
            self.weights = {name: 1.0 for name in self.operators.keys()}
