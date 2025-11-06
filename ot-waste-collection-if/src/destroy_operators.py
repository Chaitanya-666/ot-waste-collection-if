"""
Destroy operators for ALNS in VRP with Intermediate Facilities
"""

import random
import math
from typing import List, Set, Dict, Any
from .solution import Solution, Route
from .problem import ProblemInstance, Location


class DestroyOperator:
    """Base class for destroy operators"""

    def __init__(self, name: str):
        self.name = name
        self.performance_score = 0.0
        self.usage_count = 0

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Apply the destroy operator to create a partial solution"""
        raise NotImplementedError

    def get_performance_score(self) -> float:
        """Get the performance score of this operator"""
        if self.usage_count == 0:
            return 0.0
        return self.performance_score / self.usage_count

    def update_performance(self, score: float):
        """Update the performance score"""
        self.performance_score += score
        self.usage_count += 1


class RandomRemoval(DestroyOperator):
    """Randomly remove customers from the solution"""

    def __init__(self):
        super().__init__("random_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Apply random removal to create partial solution"""
        partial_solution = solution.copy()

        # Get all customer nodes
        all_customers = []
        for route in partial_solution.routes:
            for node in route.nodes:
                if node.type == "customer":
                    all_customers.append((route, node))

        if len(all_customers) <= removal_count:
            # Remove all customers if requested
            for route, customer in all_customers:
                route.nodes.remove(customer)
            return self._clean_solution(partial_solution)

        # Randomly select customers to remove
        customers_to_remove = random.sample(
            all_customers, min(removal_count, len(all_customers))
        )

        for route, customer in customers_to_remove:
            route.nodes.remove(customer)

        return self._clean_solution(partial_solution)

    def _clean_solution(self, solution: Solution) -> Solution:
        """Clean up the solution after removal"""
        # Recalculate loads
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)

        # Remove empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _recalculate_loads(self, route: Route) -> List[int]:
        """Recalculate cumulative loads for a route"""
        loads = [0]
        current_load = 0

        for node in route.nodes[1:]:  # Skip depot
            if node.type == "customer":
                current_load += node.demand
            loads.append(current_load)

        return loads


class WorstRemoval(DestroyOperator):
    """Remove customers with highest marginal cost"""

    def __init__(self):
        super().__init__("worst_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Apply worst removal to create partial solution"""
        partial_solution = solution.copy()

        # Get all customer nodes with their marginal costs
        customer_costs = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer":
                    marginal_cost = self._calculate_marginal_cost(
                        route, node_idx, partial_solution
                    )
                    customer_costs.append((route, node_idx, marginal_cost))

        if len(customer_costs) <= removal_count:
            # Remove all customers
            for route, node_idx, _ in customer_costs:
                del route.nodes[node_idx]
            return self._clean_solution(partial_solution)

        # Sort by marginal cost and remove worst ones
        customer_costs.sort(key=lambda x: x[2], reverse=True)

        for route, node_idx, _ in customer_costs[:removal_count]:
            del route.nodes[node_idx]

        return self._clean_solution(partial_solution)

    def _calculate_marginal_cost(
        self, route: Route, node_idx: int, solution: Solution
    ) -> float:
        """Calculate the marginal cost of removing a customer"""
        if node_idx <= 0 or node_idx >= len(route.nodes) - 1:
            return 0.0

        # Get the customer
        customer = route.nodes[node_idx]

        # Calculate original segment cost
        from_node = route.nodes[node_idx - 1]
        to_node = route.nodes[node_idx + 1]
        original_cost = solution.problem.calculate_distance(from_node, to_node)

        # Calculate new segment cost (direct connection)
        new_cost = solution.problem.calculate_distance(from_node, to_node)

        return original_cost - new_cost

    def _clean_solution(self, solution: Solution) -> Solution:
        """Clean up the solution after removal"""
        # Recalculate loads
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)

        # Remove empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _recalculate_loads(self, route: Route) -> List[int]:
        """Recalculate cumulative loads for a route"""
        loads = [0]
        current_load = 0

        for node in route.nodes[1:]:  # Skip depot
            if node.type == "customer":
                current_load += node.demand
            loads.append(current_load)

        return loads


class ShawRemoval(DestroyOperator):
    """Remove similar customers based on proximity and demand similarity"""

    def __init__(self):
        super().__init__("shaw_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Apply Shaw removal to create partial solution"""
        partial_solution = solution.copy()

        # Get all customer nodes
        all_customers = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer":
                    all_customers.append((route, node_idx, node))

        if len(all_customers) <= removal_count:
            # Remove all customers
            for route, node_idx, _ in all_customers:
                del route.nodes[node_idx]
            return self._clean_solution(partial_solution)

        # Select random seed customer
        seed_route, seed_idx, seed_customer = random.choice(all_customers)

        # Calculate similarity for all customers
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

        # Sort by similarity and remove most similar ones
        similarities.sort(key=lambda x: x[2], reverse=True)

        removed_count = 0
        for route, node_idx, similarity in similarities:
            if removed_count >= removal_count:
                break
            del route.nodes[node_idx]
            removed_count += 1

        return self._clean_solution(partial_solution)

    def _calculate_similarity(
        self, customer1: Location, customer2: Location, problem: ProblemInstance
    ) -> float:
        """Calculate similarity between two customers"""
        # Distance component
        distance = problem.calculate_distance(customer1, customer2)

        # Demand similarity component
        demand_diff = abs(customer1.demand - customer2.demand)

        # Combine components
        similarity = 1.0 / (1.0 + distance + demand_diff)

        return similarity

    def _clean_solution(self, solution: Solution) -> Solution:
        """Clean up the solution after removal"""
        # Recalculate loads
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)

        # Remove empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _recalculate_loads(self, route: Route) -> List[int]:
        """Recalculate cumulative loads for a route"""
        loads = [0]
        current_load = 0

        for node in route.nodes[1:]:  # Skip depot
            if node.type == "customer":
                current_load += node.demand
            loads.append(current_load)

        return loads


class RouteRemoval(DestroyOperator):
    """Remove entire route segments"""

    def __init__(self):
        super().__init__("route_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Apply route removal to create partial solution"""
        partial_solution = solution.copy()

        if len(partial_solution.routes) <= removal_count:
            # Remove all routes
            partial_solution.routes = []
            return partial_solution

        # Select random routes to remove
        routes_to_remove = random.sample(
            partial_solution.routes, min(removal_count, len(partial_solution.routes))
        )

        for route in routes_to_remove:
            partial_solution.routes.remove(route)

        return partial_solution


class RelatedRemoval(DestroyOperator):
    """Remove related customers based on clustering"""

    def __init__(self):
        super().__init__("related_removal")

    def apply(self, solution: Solution, removal_count: int) -> Solution:
        """Apply related removal to create partial solution"""
        partial_solution = solution.copy()

        # Get all customer nodes
        all_customers = []
        for route_idx, route in enumerate(partial_solution.routes):
            for node_idx, node in enumerate(route.nodes):
                if node.type == "customer":
                    all_customers.append((route, node_idx, node))

        if len(all_customers) <= removal_count:
            # Remove all customers
            for route, node_idx, _ in all_customers:
                del route.nodes[node_idx]
            return self._clean_solution(partial_solution)

        # Find clusters of related customers
        clusters = self._find_customer_clusters(all_customers, solution.problem)

        # Select largest cluster and remove customers from it
        if clusters:
            largest_cluster = max(clusters, key=len)
            customers_to_remove = largest_cluster[:removal_count]

            for route, node_idx, _ in customers_to_remove:
                del route.nodes[node_idx]

        return self._clean_solution(partial_solution)

    def _find_customer_clusters(
        self, customers: List, problem: ProblemInstance
    ) -> List[List]:
        """Find clusters of related customers"""
        clusters = []
        visited = set()

        for i, (route1, idx1, customer1) in enumerate(customers):
            if i in visited:
                continue

            # Start new cluster
            cluster = [(route1, idx1, customer1)]
            visited.add(i)

            # Find related customers
            for j, (route2, idx2, customer2) in enumerate(customers):
                if j in visited:
                    continue

                # Check if customers are related
                distance = problem.calculate_distance(customer1, customer2)
                demand_diff = abs(customer1.demand - customer2.demand)

                # Threshold for related customers
                if distance < 50.0 and demand_diff < 5:  # Configurable thresholds
                    cluster.append((route2, idx2, customer2))
                    visited.add(j)

            if len(cluster) > 1:  # Only keep clusters with multiple customers
                clusters.append(cluster)

        return clusters

    def _clean_solution(self, solution: Solution) -> Solution:
        """Clean up the solution after removal"""
        # Recalculate loads
        for route in solution.routes:
            route.loads = self._recalculate_loads(route)

        # Remove empty routes
        solution.routes = [route for route in solution.routes if len(route.nodes) > 1]

        return solution

    def _recalculate_loads(self, route: Route) -> List[int]:
        """Recalculate cumulative loads for a route"""
        loads = [0]
        current_load = 0

        for node in route.nodes[1:]:  # Skip depot
            if node.type == "customer":
                current_load += node.demand
            loads.append(current_load)

        return loads


class DestroyOperatorManager:
    """Manager for all destroy operators"""

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.operators = {
            "random": RandomRemoval(),
            "worst": WorstRemoval(),
            "shaw": ShawRemoval(),
            "route": RouteRemoval(),
            "related": RelatedRemoval(),
        }
        self.weights = {name: 1.0 for name in self.operators.keys()}

    def select_operator(self) -> str:
        """Select an operator based on weights"""
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
        """Apply a specific operator"""
        if operator_name not in self.operators:
            raise ValueError(f"Unknown operator: {operator_name}")

        operator = self.operators[operator_name]
        return operator.apply(solution, removal_count)

    def update_operator_performance(self, operator_name: str, score: float):
        """Update operator performance score"""
        if operator_name in self.operators:
            self.operators[operator_name].update_performance(score)

    def get_operator_performance(self) -> Dict[str, float]:
        """Get performance scores for all operators"""
        return {name: op.get_performance_score() for name, op in self.operators.items()}

    def update_weights(self):
        """Update operator weights based on performance"""
        performances = self.get_operator_performance()
        total_performance = sum(performances.values())

        if total_performance > 0:
            for name in self.operators.keys():
                self.weights[name] = performances[name] / total_performance
        else:
            # Equal weights if no performance data
            self.weights = {name: 1.0 for name in self.operators.keys()}
