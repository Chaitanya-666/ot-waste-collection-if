"""
Enhanced Initial Solution Construction Methods for VRP with Intermediate Facilities

This module provides advanced construction heuristics for generating high-quality
initial solutions for the ALNS algorithm. These methods are designed to produce
better starting points than simple greedy approaches, which can significantly
improve algorithm performance and convergence speed.

The enhanced construction methods include:
- Multi-start greedy construction with different strategies
- Cluster-based construction for better spatial organization
- IF-aware construction that properly handles intermediate facilities
- Parallel construction for faster generation
- Adaptive construction that selects the best method based on problem characteristics
"""

import random
import math
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from solution import Solution, Route
from problem import ProblemInstance, Location


class ConstructionStrategy(Enum):
    """Different construction strategies available"""

    GREEDY_NEAREST = "greedy_nearest"
    GREEDY_FARTHEST = "greedy_farthest"
    SAVINGS_BASED = "savings_based"
    CLUSTER_BASED = "cluster_based"
    DEMAND_BASED = "demand_based"
    MIXED_STRATEGY = "mixed_strategy"


@dataclass
class ConstructionResult:
    """Result from a construction method"""

    solution: Solution
    strategy: ConstructionStrategy
    construction_time: float
    solution_cost: float
    quality_score: float
    additional_metrics: Dict[str, float]


class EnhancedConstructionHeuristics:
    """
    Enhanced construction heuristics for VRP with IFs.

    Provides multiple sophisticated construction methods that can be used
    to generate high-quality initial solutions for the ALNS algorithm.
    """

    def __init__(self, problem: ProblemInstance, seed: int = 42):
        """
        Initialize the construction heuristics.

        Args:
            problem: The problem instance to construct solutions for
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.seed = seed
        random.seed(seed)

        # Problem characteristics for adaptive selection
        self.problem_characteristics = self._analyze_problem_characteristics()

    def _analyze_problem_characteristics(self) -> Dict[str, float]:
        """Analyze problem characteristics to guide construction strategy selection."""
        characteristics = {}

        # Calculate spatial characteristics
        if self.problem.customers:
            # Spatial spread
            distances = []
            for i, cust1 in enumerate(self.problem.customers):
                for cust2 in self.problem.customers[i + 1 :]:
                    dist = self.problem.calculate_distance(cust1, cust2)
                    distances.append(dist)

            characteristics["spatial_spread"] = (
                statistics.mean(distances) if distances else 0.0
            )
            characteristics["spatial_variance"] = (
                statistics.stdev(distances) if distances else 0.0
            )

            # Demand characteristics
            demands = [c.demand for c in self.problem.customers]
            characteristics["avg_demand"] = statistics.mean(demands)
            characteristics["demand_variance"] = (
                statistics.stdev(demands) if len(demands) > 1 else 0.0
            )
            characteristics["max_demand"] = max(demands)
            characteristics["min_demand"] = min(demands)

            # Customer density
            total_area = self._calculate_total_area()
            characteristics["customer_density"] = (
                len(self.problem.customers) / total_area
            )

            # IF availability
            characteristics["if_ratio"] = len(
                self.problem.intermediate_facilities
            ) / max(len(self.problem.customers), 1)

            # Capacity ratio
            total_demand = sum(demands)
            characteristics["capacity_ratio"] = total_demand / (
                self.problem.vehicle_capacity * max(len(self.problem.customers), 1)
            )
        else:
            characteristics["spatial_spread"] = 0.0
            characteristics["spatial_variance"] = 0.0
            characteristics["avg_demand"] = 0.0
            characteristics["demand_variance"] = 0.0
            characteristics["max_demand"] = 0.0
            characteristics["min_demand"] = 0.0
            characteristics["customer_density"] = 0.0
            characteristics["if_ratio"] = 0.0
            characteristics["capacity_ratio"] = 0.0

        return characteristics

    def _calculate_total_area(self) -> float:
        """Calculate the total area covered by customer locations."""
        if not self.problem.customers:
            return 1.0

        min_x = min(c.x for c in self.problem.customers)
        max_x = max(c.x for c in self.problem.customers)
        min_y = min(c.y for c in self.problem.customers)
        max_y = max(c.y for c in self.problem.customers)

        return (max_x - min_x) * (max_y - min_y)

    def construct_solution(
        self, strategy: ConstructionStrategy = None
    ) -> ConstructionResult:
        """
        Construct a solution using the specified strategy.

        Args:
            strategy: Construction strategy to use (auto-selects if None)

        Returns:
            ConstructionResult: The constructed solution and metadata
        """
        start_time = time.time()

        if strategy is None:
            strategy = self._select_adaptive_strategy()

        solution = None

        if strategy == ConstructionStrategy.GREEDY_NEAREST:
            solution = self._greedy_nearest_construction()
        elif strategy == ConstructionStrategy.GREEDY_FARTHEST:
            solution = self._greedy_farthest_construction()
        elif strategy == ConstructionStrategy.SAVINGS_BASED:
            solution = self._savings_based_construction()
        elif strategy == ConstructionStrategy.CLUSTER_BASED:
            solution = self._cluster_based_construction()
        elif strategy == ConstructionStrategy.DEMAND_BASED:
            solution = self._demand_based_construction()
        elif strategy == ConstructionStrategy.MIXED_STRATEGY:
            solution = self._mixed_strategy_construction()

        construction_time = time.time() - start_time
        solution_cost = solution.total_cost if solution else float("inf")
        quality_score = self._calculate_quality_score(solution)
        additional_metrics = self._calculate_construction_metrics(solution)

        return ConstructionResult(
            solution=solution,
            strategy=strategy,
            construction_time=construction_time,
            solution_cost=solution_cost,
            quality_score=quality_score,
            additional_metrics=additional_metrics,
        )

    def _select_adaptive_strategy(self) -> ConstructionStrategy:
        """Select the best construction strategy based on problem characteristics."""
        chars = self.problem_characteristics

        # Decision rules based on problem characteristics
        if (
            chars["customer_density"] > 0.1
            and chars["spatial_variance"] < chars["spatial_spread"] * 0.3
        ):
            # High density, low variance -> cluster-based
            return ConstructionStrategy.CLUSTER_BASED
        elif chars["demand_variance"] > chars["avg_demand"] * 0.5:
            # High demand variance -> demand-based
            return ConstructionStrategy.DEMAND_BASED
        elif chars["if_ratio"] > 0.2:
            # Many IFs available -> savings-based
            return ConstructionStrategy.SAVINGS_BASED
        elif chars["capacity_ratio"] > 0.8:
            # High capacity utilization -> greedy nearest
            return ConstructionStrategy.GREEDY_NEAREST
        else:
            # Default to mixed strategy
            return ConstructionStrategy.MIXED_STRATEGY

    def _greedy_nearest_construction(self) -> Solution:
        """Greedy nearest neighbor construction with IF awareness."""
        solution = Solution(self.problem)
        unassigned = set(c.id for c in self.problem.customers)

        while unassigned:
            # Start a new route
            route = Route()
            route.nodes = [self.problem.depot]
            current_load = 0.0
            current_location = self.problem.depot
            route_loads = [0.0]  # Load at depot is 0

            while unassigned:
                # Find nearest unassigned customer
                candidates = []

                for customer_id in unassigned:
                    customer = next(
                        c for c in self.problem.customers if c.id == customer_id
                    )

                    # Check if we can add this customer directly
                    if current_load + customer.demand <= self.problem.vehicle_capacity:
                        direct_distance = self.problem.calculate_distance(
                            current_location, customer
                        )
                        candidates.append((customer, direct_distance, False))
                    else:
                        # Need to visit IF first
                        if self.problem.intermediate_facilities:
                            # Find best IF + customer combination
                            best_if = None
                            min_combined_distance = float("inf")

                            for if_loc in self.problem.intermediate_facilities:
                                if_distance = self.problem.calculate_distance(
                                    current_location, if_loc
                                )
                                customer_distance = self.problem.calculate_distance(
                                    if_loc, customer
                                )
                                combined_distance = if_distance + customer_distance

                                if combined_distance < min_combined_distance:
                                    min_combined_distance = combined_distance
                                    best_if = if_loc

                            if best_if:
                                candidates.append(
                                    (customer, min_combined_distance, True, best_if)
                                )

                if not candidates:
                    break

                # Select best candidate
                candidates.sort(key=lambda x: x[1])
                best_customer, best_distance, needs_if, best_if = (
                    candidates[0]
                    if len(candidates[0]) == 4
                    else (candidates[0][0], candidates[0][1], candidates[0][2], None)
                )

                if needs_if and best_if:
                    # Add IF visit
                    route.nodes.append(best_if)
                    current_load = 0.0
                    current_location = best_if
                    route_loads.append(0.0)

                # Add customer
                route.nodes.append(best_customer)
                current_load += best_customer.demand
                unassigned.remove(best_customer.id)
                current_location = best_customer
                route_loads.append(current_load)

            # Return to depot
            route.nodes.append(self.problem.depot)
            route.loads = route_loads
            route.calculate_metrics(self.problem)
            solution.routes.append(route)

        solution.calculate_metrics()
        return solution

    def _greedy_farthest_construction(self) -> Solution:
        """Greedy farthest neighbor construction for diversity."""
        solution = Solution(self.problem)
        unassigned = set(c.id for c in self.problem.customers)

        while unassigned:
            # Start a new route
            route = Route()
            route.nodes = [self.problem.depot]
            current_load = 0.0
            current_location = self.problem.depot

            while unassigned:
                # Find farthest unassigned customer
                candidates = []

                for customer_id in unassigned:
                    customer = next(
                        c for c in self.problem.customers if c.id == customer_id
                    )

                    # Check if we can add this customer directly
                    if current_load + customer.demand <= self.problem.vehicle_capacity:
                        direct_distance = self.problem.calculate_distance(
                            current_location, customer
                        )
                        candidates.append((customer, direct_distance, False))
                    else:
                        # Need to visit IF first
                        if self.problem.intermediate_facilities:
                            # Find best IF + customer combination
                            best_if = None
                            max_combined_distance = 0.0

                            for if_loc in self.problem.intermediate_facilities:
                                if_distance = self.problem.calculate_distance(
                                    current_location, if_loc
                                )
                                customer_distance = self.problem.calculate_distance(
                                    if_loc, customer
                                )
                                combined_distance = if_distance + customer_distance

                                if combined_distance > max_combined_distance:
                                    max_combined_distance = combined_distance
                                    best_if = if_loc

                            if best_if:
                                candidates.append(
                                    (customer, max_combined_distance, True, best_if)
                                )

                if not candidates:
                    break

                # Select best candidate (farthest)
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_customer, best_distance, needs_if, best_if = (
                    candidates[0]
                    if len(candidates[0]) == 4
                    else (candidates[0][0], candidates[0][1], candidates[0][2], None)
                )

                if needs_if and best_if:
                    # Add IF visit
                    route.nodes.append(best_if)
                    current_load = 0.0
                    current_location = best_if

                # Add customer
                route.nodes.append(best_customer)
                current_load += best_customer.demand
                unassigned.remove(best_customer.id)
                current_location = best_customer

            # Return to depot
            route.nodes.append(self.problem.depot)
            route.calculate_metrics(self.problem)
            solution.routes.append(route)

        solution.calculate_metrics()
        return solution

    def _savings_based_construction(self) -> Solution:
        """Clarke-Wright savings algorithm with IF awareness."""
        solution = Solution(self.problem)

        # Initialize individual customer routes
        customer_routes = {}
        for customer in self.problem.customers:
            route = Route()
            route.nodes = [self.problem.depot, customer, self.problem.depot]
            route.calculate_metrics(self.problem)
            customer_routes[customer.id] = route

        # Calculate savings with IF awareness
        savings = []
        for i, cust1 in enumerate(self.problem.customers):
            for j, cust2 in enumerate(self.problem.customers):
                if i < j:  # Avoid duplicates
                    # Basic savings
                    basic_saving = (
                        self.problem.calculate_distance(self.problem.depot, cust1)
                        + self.problem.calculate_distance(self.problem.depot, cust2)
                        - self.problem.calculate_distance(cust1, cust2)
                    )

                    # IF-aware savings penalty
                    if_penalty = 0.0
                    if self.problem.intermediate_facilities:
                        # Estimate IF visits needed for combined route
                        combined_demand = cust1.demand + cust2.demand
                        if_visits_needed = (
                            math.ceil(combined_demand / self.problem.vehicle_capacity)
                            - 1
                        )

                        if if_visits_needed > 0:
                            # Add penalty for additional IF visits
                            if_penalty = if_visits_needed * min(
                                self.problem.calculate_distance(
                                    if_loc, self.problem.depot
                                )
                                for if_loc in self.problem.intermediate_facilities
                            )

                    net_saving = basic_saving - if_penalty
                    savings.append((net_saving, cust1.id, cust2.id))

        # Sort savings in descending order
        savings.sort(reverse=True, key=lambda x: x[0])

        # Merge routes
        for saving, cust1_id, cust2_id in savings:
            route1 = customer_routes.get(cust1_id)
            route2 = customer_routes.get(cust2_id)

            if route1 and route2 and route1 != route2:
                # Try to merge routes with IF awareness
                if self._can_merge_routes_with_if(route1, route2):
                    # Merge route2 into route1
                    merged_route = self._merge_routes_with_if(route1, route2)
                    customer_routes[cust1_id] = merged_route
                    del customer_routes[cust2_id]

        # Collect all routes
        solution.routes = list(customer_routes.values())
        solution.calculate_metrics()
        return solution

    def _can_merge_routes_with_if(self, route1: Route, route2: Route) -> bool:
        """Check if two routes can be merged considering IF constraints."""
        # Check capacity
        total_demand = 0.0
        for node in route1.nodes[1:-1]:  # Exclude depot
            if node.type == "customer":
                total_demand += node.demand
        for node in route2.nodes[1:-1]:  # Exclude depot
            if node.type == "customer":
                total_demand += node.demand

        # Check if we can serve all customers without exceeding capacity
        if total_demand > self.problem.vehicle_capacity:
            return False

        # Check if IF visits are sufficient
        max_route_length = max(len(route1.nodes), len(route2.nodes))
        if max_route_length > 10:  # Heuristic: long routes may need more IFs
            return len(self.problem.intermediate_facilities) >= 2

        return True

    def _merge_routes_with_if(self, route1: Route, route2: Route) -> Route:
        """Merge two routes with IF insertion if needed."""
        merged_route = Route()

        # Simple merge: combine route1 and route2
        merged_route.nodes = route1.nodes[:-1] + route2.nodes[1:]

        # Check if we need to insert IF visits
        current_load = 0.0
        if_nodes_to_insert = []

        for node in merged_route.nodes:
            if node.type == "customer":
                current_load += node.demand
                if current_load > self.problem.vehicle_capacity:
                    # Need to insert IF
                    if self.problem.intermediate_facilities:
                        nearest_if = min(
                            self.problem.intermediate_facilities,
                            key=lambda if_loc: self.problem.calculate_distance(
                                node, if_loc
                            ),
                        )
                        if_nodes_to_insert.append(
                            (merged_route.nodes.index(node) + 1, nearest_if)
                        )
                        current_load = 0.0

        # Insert IF nodes
        for insert_pos, if_node in reversed(if_nodes_to_insert):
            merged_route.nodes.insert(insert_pos, if_node)

        merged_route.calculate_metrics(self.problem)
        return merged_route

    def _cluster_based_construction(self) -> Solution:
        """Construction based on customer clustering."""
        solution = Solution(self.problem)

        # Perform customer clustering
        clusters = self._cluster_customers()

        # Construct routes for each cluster
        for cluster_id, cluster_customers in clusters.items():
            if not cluster_customers:
                continue

            # Create route for this cluster
            route = Route()
            route.nodes = [self.problem.depot]
            current_load = 0.0
            current_location = self.problem.depot

            # Sort customers within cluster by distance from depot
            cluster_customers.sort(
                key=lambda c: self.problem.calculate_distance(self.problem.depot, c)
            )

            for customer in cluster_customers:
                if current_load + customer.demand <= self.problem.vehicle_capacity:
                    # Add customer directly
                    route.nodes.append(customer)
                    current_load += customer.demand
                    current_location = customer
                else:
                    # Need to visit IF
                    if self.problem.intermediate_facilities:
                        nearest_if = min(
                            self.problem.intermediate_facilities,
                            key=lambda if_loc: self.problem.calculate_distance(
                                current_location, if_loc
                            ),
                        )
                        route.nodes.append(nearest_if)
                        current_load = 0.0
                        current_location = nearest_if

                        # Add customer after IF
                        route.nodes.append(customer)
                        current_load += customer.demand
                        current_location = customer

            # Return to depot
            route.nodes.append(self.problem.depot)
            route.calculate_metrics(self.problem)
            solution.routes.append(route)

        solution.calculate_metrics()
        return solution

    def _cluster_customers(self) -> Dict[int, List[Location]]:
        """Perform customer clustering using simple distance-based approach."""
        if not self.problem.customers:
            return {}

        clusters = {}
        cluster_id = 0

        unclustered = set(self.problem.customers)

        while unclustered:
            # Start new cluster with random customer
            seed_customer = random.choice(list(unclustered))
            cluster = [seed_customer]
            unclustered.remove(seed_customer)

            # Add nearby customers
            cluster_center = seed_customer
            max_cluster_distance = self.problem_characteristics["spatial_spread"] * 0.3

            to_add = list(unclustered)
            for customer in to_add:
                distance = self.problem.calculate_distance(cluster_center, customer)
                if distance <= max_cluster_distance:
                    cluster.append(customer)
                    unclustered.remove(customer)

            clusters[cluster_id] = cluster
            cluster_id += 1

        return clusters

    def _demand_based_construction(self) -> Solution:
        """Construction based on customer demand characteristics."""
        solution = Solution(self.problem)
        unassigned = set(c.id for c in self.problem.customers)

        # Sort customers by demand (descending)
        sorted_customers = sorted(
            self.problem.customers, key=lambda c: c.demand, reverse=True
        )

        while unassigned:
            # Start a new route
            route = Route()
            route.nodes = [self.problem.depot]
            current_load = 0.0
            current_location = self.problem.depot

            # Add high-demand customers first
            for customer in sorted_customers:
                if customer.id not in unassigned:
                    continue

                if current_load + customer.demand <= self.problem.vehicle_capacity:
                    # Add customer
                    route.nodes.append(customer)
                    current_load += customer.demand
                    unassigned.remove(customer.id)
                    current_location = customer
                else:
                    # Try to find a smaller customer that fits
                    found_smaller = False
                    for small_customer in sorted_customers:
                        if (
                            small_customer.id in unassigned
                            and current_load + small_customer.demand
                            <= self.problem.vehicle_capacity
                        ):
                            route.nodes.append(small_customer)
                            current_load += small_customer.demand
                            unassigned.remove(small_customer.id)
                            current_location = small_customer
                            found_smaller = True
                            break

                    if not found_smaller:
                        # Need to visit IF
                        if self.problem.intermediate_facilities:
                            nearest_if = min(
                                self.problem.intermediate_facilities,
                                key=lambda if_loc: self.problem.calculate_distance(
                                    current_location, if_loc
                                ),
                            )
                            route.nodes.append(nearest_if)
                            current_load = 0.0
                            current_location = nearest_if

                            # Try to add the original customer again
                            if customer.id in unassigned:
                                route.nodes.append(customer)
                                current_load += customer.demand
                                unassigned.remove(customer.id)
                                current_location = customer

            # Return to depot
            route.nodes.append(self.problem.depot)
            route.calculate_metrics(self.problem)
            solution.routes.append(route)

        solution.calculate_metrics()
        return solution

    def _mixed_strategy_construction(self) -> Solution:
        """Mixed strategy construction using multiple approaches."""
        # Try different strategies and select the best
        strategies = [
            ConstructionStrategy.GREEDY_NEAREST,
            ConstructionStrategy.CLUSTER_BASED,
            ConstructionStrategy.DEMAND_BASED,
        ]

        best_solution = None
        best_score = float("-inf")

        for strategy in strategies:
            result = self.construct_solution(strategy)
            score = result.quality_score

            if score > best_score:
                best_score = score
                best_solution = result.solution

        return best_solution

    def _calculate_quality_score(self, solution: Solution) -> float:
        """Calculate a quality score for a solution."""
        if not solution or not solution.routes:
            return 0.0

        # Multiple quality factors
        factors = {}

        # Route balance (lower variance is better)
        route_distances = [route.total_distance for route in solution.routes]
        if len(route_distances) > 1:
            factors["route_balance"] = 1.0 / (1.0 + statistics.stdev(route_distances))
        else:
            factors["route_balance"] = 1.0

        # Vehicle utilization (higher is better, but not too high)
        total_demand = sum(c.demand for c in self.problem.customers)
        total_capacity = len(solution.routes) * self.problem.vehicle_capacity
        utilization = total_demand / total_capacity
        factors["utilization"] = min(1.0, utilization)  # Cap at 1.0

        # IF visit efficiency (moderate is better)
        if_visits = sum(
            len([n for n in route.nodes if n.type == "if"]) for route in solution.routes
        )
        if_efficiency = if_visits / max(len(solution.routes), 1)
        factors["if_efficiency"] = 1.0 / (
            1.0 + abs(if_efficiency - 2.0)
        )  # Prefer around 2 IFs per route

        # Distance efficiency (higher is better)
        distance_efficiency = total_demand / max(solution.total_distance, 1.0)
        factors["distance_efficiency"] = min(1.0, distance_efficiency)

        # Weighted combination
        weights = {
            "route_balance": 0.3,
            "utilization": 0.3,
            "if_efficiency": 0.2,
            "distance_efficiency": 0.2,
        }

        quality_score = sum(factors[key] * weights[key] for key in factors)
        return quality_score

    def _calculate_construction_metrics(self, solution: Solution) -> Dict[str, float]:
        """Calculate additional metrics for construction evaluation."""
        metrics = {}

        if solution and solution.routes:
            metrics["num_routes"] = len(solution.routes)
            metrics["avg_route_length"] = sum(
                len(route.nodes) for route in solution.routes
            ) / len(solution.routes)
            metrics["total_if_visits"] = sum(
                len([n for n in route.nodes if n.type == "if"])
                for route in solution.routes
            )
            metrics["avg_customers_per_route"] = sum(
                len([n for n in route.nodes if n.type == "customer"])
                for route in solution.routes
            ) / len(solution.routes)
        else:
            metrics["num_routes"] = 0
            metrics["avg_route_length"] = 0.0
            metrics["total_if_visits"] = 0
            metrics["avg_customers_per_route"] = 0.0

        return metrics

    def multi_start_construction(self, num_starts: int = 5) -> ConstructionResult:
        """
        Perform multi-start construction and return the best solution.

        Args:
            num_starts: Number of construction attempts

        Returns:
            ConstructionResult: Best solution found
        """
        best_result = None
        best_score = float("-inf")

        for i in range(num_starts):
            # Use different seeds for each construction
            original_seed = self.seed
            self.seed = original_seed + i

            # Try different strategies
            strategies = [
                ConstructionStrategy.GREEDY_NEAREST,
                ConstructionStrategy.GREEDY_FARTHEST,
                ConstructionStrategy.CLUSTER_BASED,
                ConstructionStrategy.DEMAND_BASED,
            ]

            for strategy in strategies:
                result = self.construct_solution(strategy)

                if result.quality_score > best_score:
                    best_score = result.quality_score
                    best_result = result

        # Restore original seed
        self.seed = original_seed

        return best_result
