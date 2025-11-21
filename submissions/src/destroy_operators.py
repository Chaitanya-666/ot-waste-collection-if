# Author: Chaitanya Shinde (231070066)
#
# This file implements the "destroy" operators for the ALNS algorithm. These
# operators are responsible for taking a complete solution and removing a
# subset of customers from it to create a partial solution, which can then be
# "repaired" in a new way.
"""
Destroy operators for ALNS in VRP with Intermediate Facilities
"""

import random
import math
from typing import List, Set, Dict, Any
from .solution import Solution, Route
from .problem import ProblemInstance, Location


class DestroyOperator:
    """Base class for all destroy operators."""

    def __init__(self, name: str):
        self.name = name
        # Performance score is used for adaptive weight adjustment.
        self.performance_score = 0.0
        self.usage_count = 0

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Applies the destroy operator to a solution, returning a partial solution."""
        raise NotImplementedError

    def get_performance_score(self) -> float:
        """Calculates the average performance score of this operator."""
        if self.usage_count == 0:
            return 0.0
        return self.performance_score / self.usage_count

    def update_performance(self, score: float):
        """Updates the operator's performance score based on a recent run."""
        self.performance_score += score
        self.usage_count += 1


class RandomRemoval(DestroyOperator):
    """
    Removes a specified number of customers from the solution at random.
    This is the simplest destroy operator and helps to diversify the search by
    introducing random changes to the solution.
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
    Removes customers that are the most expensive to serve. The cost is
    calculated as the savings in distance if the customer were removed.
    This operator focuses on removing "bad" parts of the solution, with the
    hope that the repair operator can find a better placement.
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
    Removes customers that are "similar" to each other. Similarity is based on
    a combination of geographic proximity, demand, and service time. This helps
    to destroy and rebuild entire regions of the solution.
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
    Removes one or more entire routes from the solution. This is a large-scale
    operator that can lead to significant changes in the solution structure,
    allowing the search to escape local optima.
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
    Removes customers that are geographically clustered together. This is similar
    to Shaw removal but focuses only on spatial proximity, making it a more
    geographically-focused destroy operator.
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
    Manages the selection and application of destroy operators. It uses an
    adaptive weighting mechanism to select operators that have been more
    successful in finding good solutions.
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
        """Selects a destroy operator based on their adaptive weights using roulette wheel selection."""
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
        """Applies a specific destroy operator by name."""
        if operator_name not in self.operators:
            raise ValueError(f"Unknown operator: {operator_name}")

        operator = self.operators[operator_name]
        return operator.apply(solution, removal_count)

    def update_operator_performance(self, operator_name: str, score: float):
        """Updates the performance score of a specific operator."""
        if operator_name in self.operators:
            self.operators[operator_name].update_performance(score)

    def get_operator_performance(self) -> Dict[str, float]:
        """Returns a dictionary of performance scores for all operators."""
        return {name: op.get_performance_score() for name, op in self.operators.items()}

    def update_weights(self):
        """Updates the weights of all operators based on their recent performance."""
        performances = self.get_operator_performance()
        total_performance = sum(performances.values())

        if total_performance > 0:
            for name in self.operators.keys():
                self.weights[name] = performances[name] / total_performance
        else:
            # If there's no performance data, reset to equal weights.
            self.weights = {name: 1.0 for name in self.operators.keys()}
