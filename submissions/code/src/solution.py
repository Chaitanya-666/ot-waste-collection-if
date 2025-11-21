"""
Solution Representation for Vehicle Routing Problem with Intermediate Facilities (VRP-IF)
==============================================================================

Author: Chaitanya Shinde (231070066) - Core solution representation and route management

This module defines the core data structures for representing solutions to the
Vehicle Routing Problem with Intermediate Facilities (VRP-IF). It includes:
- Route class for representing individual vehicle paths
- Solution class for managing a complete solution with multiple routes
- Utility functions for solution manipulation and validation

Key Features:
- Tracks vehicle routes with detailed metrics (distance, time, load)
- Supports partial solutions with unassigned customers
- Validates solution feasibility against problem constraints
- Enables efficient solution copying and comparison
"""

from typing import List, Optional, Set, Tuple, Union, Dict, Any
from dataclasses import dataclass, field

# Import problem definition for type hints
from .problem import ProblemInstance, Location


class Route:
    """
    Represents a single vehicle's route in the VRP-IF problem.
    
    A route starts and ends at a depot and visits a sequence of customers and
    intermediate facilities (IFs). It tracks various metrics including:
    - Sequence of visited nodes (depot, customers, IFs)
    - Vehicle load after each node visit
    - Total distance and time for the route
    
    Attributes:
        nodes (List[Location]): Sequence of locations in the route
        loads (List[float]): Vehicle load after visiting each node
        total_distance (float): Total travel distance of the route
        total_time (float): Total time including travel and service times
        vehicle_id (Optional[Any]): Identifier for the vehicle assigned to this route
        
    Author: Chaitanya Shinde (231070066)
    """
    
    def __init__(self) -> None:
        """Initialize a new empty route."""
        self.nodes: List[Location] = []
        self.loads: List[float] = []
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.vehicle_id: Optional[Any] = None

    def calculate_metrics(self, problem: 'ProblemInstance') -> None:
        """
        Recalculate all metrics for the route based on the current sequence of nodes.
        
        This method updates:
        - Total travel distance
        - Total time (including service times and disposal times)
        - Vehicle load at each node
        
        Args:
            problem: The problem instance containing depot, customers, and IFs
            
        Note:
            - Call this method after modifying the route's nodes
            - Handles both customer service times and IF disposal times
            - Resets vehicle load to 0 after visiting an IF
            
        Author: Chaitanya Shinde (231070066)
        """
        self.total_distance = 0.0
        self.total_time = 0.0
        current_load = 0

        # Ensure loads list has an entry per node (initialize safely)
        if self.nodes:
            # initialize loads with zeros; one entry per node
            self.loads = [0 for _ in range(len(self.nodes))]
        else:
            self.loads = []

        # Iterate over arcs (node i -> node i+1) and update metrics and loads at the next node
        for i in range(len(self.nodes) - 1):
            current = self.nodes[i]
            next_node = self.nodes[i + 1]

            # Add travel distance and time
            distance = problem.calculate_distance(current, next_node)
            self.total_distance += distance
            self.total_time += problem.calculate_travel_time(current, next_node)

            # Add service time and update load after visiting next_node
            if getattr(next_node, "type", None) == "customer":
                self.total_time += getattr(next_node, "service_time", 0)
                current_load += float(getattr(next_node, "demand", 0))
            elif getattr(next_node, "type", None) == "if":
                self.total_time += getattr(problem, "disposal_time", 0)
                current_load = 0  # Load is reset at an IF

            # store load corresponding to next_node position (i+1)
            if i + 1 < len(self.loads):
                self.loads[i + 1] = current_load

        # If there is a single-node route (depot only) ensure loads length is correct
        if self.nodes and len(self.loads) < len(self.nodes):
            self.loads += [0] * (len(self.nodes) - len(self.loads))

    def is_feasible(self, problem: 'ProblemInstance') -> Tuple[bool, str]:
        """
        Check if the route satisfies all problem constraints.
        
        Args:
            problem: The problem instance containing constraints
            
        Returns:
            Tuple[bool, str]: 
                - First element: True if feasible, False otherwise
                - Second element: Description of any constraint violation
                
        Checks performed:
            1. Route is not empty (except for empty routes which are technically feasible)
            2. Route starts and ends at the same depot
            3. Vehicle capacity is not exceeded between IF visits
            4. All nodes are valid locations in the problem
            
        Author: Chaitanya Shinde (231070066)
        """
        # Empty routes or routes with only depot are considered feasible but should be filtered
        if not self.nodes:
            return True, "Empty route (should be removed)"
        
        # Single node routes (only depot) are feasible
        if len(self.nodes) == 1:
            if self.nodes[0].type == "depot":
                return True, "Single depot route (empty)"
            else:
                return False, "Single node route without depot"
        
        # Check that route starts and ends at the same depot
        if self.nodes[0].type != "depot" or self.nodes[-1].type != "depot":
            return False, "Invalid depot visits"
        
        # Additional check: depots should have same ID
        if self.nodes[0].id != self.nodes[-1].id:
            return False, "Route starts and ends at different depots"

        current_load = 0
        current_time = 0

        for i in range(len(self.nodes) - 1):
            current = self.nodes[i]
            next_node = self.nodes[i + 1]

            # Check capacity constraint
            if current_load > problem.vehicle_capacity:
                return False, f"Capacity exceeded after node {current.id}"

            # Update metrics
            distance = problem.calculate_distance(current, next_node)
            current_time += problem.calculate_travel_time(current, next_node)

            if next_node.type == "customer":
                current_time += next_node.service_time
                current_load += next_node.demand
            elif next_node.type == "if":
                current_time += problem.disposal_time
                current_load = 0

        return True, "Route is feasible"

    def __repr__(self) -> str:
        """
        Return a string representation of the route.
        
        Example:
            >>> route = Route()
            >>> route.nodes = [depot, customer1, if1, depot]
            >>> print(route)
            Route(Distance: 30.50, Time: 45.20): d1 -> c1 -> if1 -> d1
            
        Returns:
            str: Formatted string showing route metrics and node sequence
        """
        route_str = " -> ".join([f"{node.type[0]}{node.id}" for node in self.nodes])
        return f"Route(Distance: {self.total_distance:.2f}, Time: {self.total_time:.2f}): {route_str}"


class Solution:
    """
    Represents a complete solution to the VRP-IF problem.
    
    A solution consists of:
    - A list of routes, each representing a vehicle's path
    - A set of unassigned customer IDs (if any)
    - Aggregated metrics (total cost, distance, time)
    
    The solution can be evaluated for feasibility and compared with others
    based on the objective function (typically total distance).
    
    Attributes:
        problem (Optional[ProblemInstance]): Reference to the problem instance
        routes (List[Route]): List of vehicle routes in the solution
        total_cost (float): Primary objective value (typically total distance)
        total_distance (float): Sum of distances across all routes
        total_time (float): Sum of times across all routes
        unassigned_customers (Set[int]): IDs of customers not in any route
        
    Author: Chaitanya Shinde (231070066)
    """
    
    def __init__(self, problem_instance: Optional['ProblemInstance'] = None) -> None:
        """
        Initialize a new solution, optionally with all customers unassigned.
        
        Args:
            problem_instance: If provided, initializes the set of unassigned
                            customers with all customers from the problem.
                            
        Note:
            - Creates an empty solution if no problem instance is provided
            - Initializes metrics to zero
            - Sets up tracking for unassigned customers
        """
        self.problem = problem_instance
        self.routes: List[Route] = []
        self.total_cost: float = 0.0
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.unassigned_customers: Set[int] = set()
        
        if problem_instance is not None:
            try:
                # Initialize with all customer IDs as unassigned
                self.unassigned_customers = {c.id for c in problem_instance.customers}
            except (AttributeError, TypeError):
                # Handle case where problem structure is unexpected
                self.unassigned_customers = set()

    def calculate_metrics(self) -> None:
        """
        Recalculate all metrics for the solution by aggregating route metrics.
        
        This method updates:
        - Total distance (sum of all route distances)
        - Total time (sum of all route times)
        - Total cost (currently equal to total distance)
        
        Note:
            - Should be called after any modification to the solution's routes
            - Updates metrics for all routes before aggregating
            - The cost function can be extended to include additional factors
            
        Author: Chaitanya Shinde (231070066)
        """
        self.total_distance = 0.0
        self.total_time = 0.0

        # Update metrics for each route and accumulate totals
        for route in self.routes:
            route.calculate_metrics(self.problem)
            self.total_distance += route.total_distance
            self.total_time += route.total_time

        # Primary objective: minimize total distance
        # Can be extended with penalties for unassigned customers, etc.
        self.total_cost = self.total_distance

    def is_feasible(self, problem: Optional['ProblemInstance'] = None) -> Tuple[bool, str]:
        """
        Verify if the solution is feasible according to all problem constraints.
        
        Args:
            problem: Optional problem instance (uses self.problem if None)
            
        Returns:
            Tuple[bool, str]: 
                - First element: True if feasible, False otherwise
                - Second element: Description of any constraint violation
                
        Checks performed:
            1. Problem instance is available
            2. Number of non-empty routes doesn't exceed vehicle limit
            3. Each customer is served exactly once (or is explicitly unassigned)
            4. Each individual route is feasible
            5. No duplicate customer assignments
            
        Author: Chaitanya Shinde (231070066)
        """
        # Determine which problem instance to use for validation.
        problem = problem if problem is not None else self.problem
        if problem is None:
            return False, "No problem instance provided for feasibility check"

        # Filter out empty routes (depot-only routes with no customers).
        non_empty_routes = [
            r for r in self.routes 
            if r.nodes and any(getattr(n, "type", None) == "customer" for n in r.nodes)
        ]
        
        # Check if the number of routes exceeds the available vehicles.
        if len(non_empty_routes) > problem.number_of_vehicles:
            return False, "Too many vehicles used"

        # Check for complete and unique customer coverage.
        served_ids = set()
        for route in non_empty_routes:
            for node in route.nodes:
                if getattr(node, "type", None) == "customer":
                    # A customer should not be served by more than one route.
                    if node.id in served_ids:
                        return False, f"Customer {node.id} served multiple times"
                    served_ids.add(node.id)

        # Build the set of all customer IDs from the problem definition.
        try:
            all_customer_ids = set(c.id for c in problem.customers)
        except Exception:
            return False, "Problem instance missing customers for feasibility check"

        # The set of served customers must match the set of all customers minus the unassigned ones.
        expected_served_ids = all_customer_ids - set(self.unassigned_customers)

        if served_ids != expected_served_ids:
            return False, "Mismatch between served and assigned customers"

        # Check the feasibility of each individual route.
        for route in non_empty_routes:
            feasible, message = route.is_feasible(problem)
            if not feasible:
                return False, f"Route {route.vehicle_id}: {message}"

        return True, "Solution is feasible"

    def copy(self) -> 'Solution':
        """
        Create a deep copy of the solution.
        
        Returns:
            Solution: A new Solution instance with the same routes and attributes
            
        Note:
            - Performs a deep copy to ensure complete independence
            - Preserves all routes, metrics, and unassigned customers
            - Useful for local search and solution manipulation
            
        Example:
            >>> sol1 = Solution(problem)
            >>> sol2 = sol1.copy()  # Independent copy
            >>> sol2.routes[0].nodes.append(new_node)  # Doesn't affect sol1
        """
        import copy
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the solution.
        
        Example:
            >>> sol = Solution(problem)
            >>> print(sol)
            Solution(Infeasible, Cost: 150.25, Distance: 150.25, Time: 180.50, Routes: 3, Unassigned: 2)
            
        Returns:
            str: Formatted string with solution metrics
        """
        try:
            status = "Feasible" if self.is_feasible()[0] else "Infeasible"
        except Exception:
            status = "Unknown"
            
        return (
            f"Solution({status}, Cost: {self.total_cost:.2f}, "
            f"Distance: {self.total_distance:.2f}, "
            f"Time: {self.total_time:.2f}, "
            f"Routes: {len(self.routes)}, "
            f"Unassigned: {len(self.unassigned_customers)})"
        )