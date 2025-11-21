"""
Problem Definition for Vehicle Routing Problem with Intermediate Facilities (VRP-IF)
==============================================================================

Authors:
    - Chaitanya Shinde (231070066) - Core data structures and problem definition
    - Harsh Sharma (231070064) - Feasibility checks and distance calculations

This module defines the core data structures and problem representation for the
Vehicle Routing Problem with Intermediate Facilities (VRP-IF). It includes:
- Location class for representing nodes (customers, depots, and facilities)
- ProblemInstance class for defining the complete problem instance
- Utility functions for distance calculations and feasibility checks

The module is designed to be flexible and can work with or without NumPy for
numerical computations, with automatic fallback to pure Python implementations.

Key Features:
- Support for depots, customers, and intermediate facilities
- Flexible distance matrix calculation
- Feasibility checking for problem instances and routes
- Vehicle capacity and route constraints
"""

from typing import List, Optional, Tuple, Union

# Try to import numpy if available for distance matrix convenience; otherwise fall back to lists
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


class Location:
    """
    Represents a physical location in the VRP-IF problem.
    
    A location can be one of three types:
    - 'depot': Starting and ending point for vehicles
    - 'customer': Location that requires service (has demand)
    - 'if': Intermediate facility where vehicles can unload
    
    Attributes:
        id (int): Unique identifier for the location
        x (float): X-coordinate (Euclidean)
        y (float): Y-coordinate (Euclidean)
        demand (float): Demand at this location (positive for customers, 0 otherwise)
        type (str): Type of location ('depot', 'customer', or 'if')
        service_time (float): Time required to service this location
        
    Author: Chaitanya Shinde (231070066)
    """
    
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        demand: float = 0.0,
        location_type: str = "customer",
        service_time: float = 0.0,
    ) -> None:
        """
        Initialize a new location.
        
        Args:
            id: Unique identifier for the location
            x: X-coordinate in the Euclidean plane
            y: Y-coordinate in the Euclidean plane
            demand: Demand at this location (default: 0.0)
            location_type: Type of location ('depot', 'customer', or 'if') (default: 'customer')
            service_time: Time required to service this location (default: 0.0)
        """
        self.id: int = id
        self.x: float = x
        self.y: float = y
        self.demand: float = demand
        self.type: str = location_type
        self.service_time: float = service_time

    def __repr__(self) -> str:
        """
        Return a string representation of the location.
        
        Returns:
            str: String in the format "Type(id, (x,y), demand=value)"
            
        Example:
            >>> loc = Location(1, 10.0, 20.0, 5.0, "customer")
            >>> print(loc)
            Customer(1, (10.0,20.0), demand=5.0)
        """
        return f"{self.type.capitalize()}({self.id}, ({self.x},{self.y}), demand={self.demand})"


class ProblemInstance:
    """
    Defines a complete VRP-IF problem instance with all locations and constraints.
    
    This class represents the entire problem to be solved, including:
    - All locations (depot, customers, intermediate facilities)
    - Vehicle properties (capacity, count)
    - Problem constraints (route lengths, times)
    - Distance calculations and caching
    
    The class provides methods for:
    - Calculating distances between locations
    - Checking solution feasibility
    - Validating routes
    - Computing problem statistics
    
    Attributes:
        name (str): Name/identifier for the problem instance
        depot (Optional[Location]): The depot location (start/end of routes)
        customers (List[Location]): List of customer locations
        intermediate_facilities (List[Location]): List of intermediate facilities
        vehicle_capacity (float): Maximum capacity of each vehicle
        max_route_time (float): Maximum allowed route duration
        max_route_length (float): Maximum allowed route distance
        number_of_vehicles (float): Number of available vehicles
        disposal_time (float): Time required to unload at intermediate facilities
        distance_matrix: Pre-computed distance matrix (numpy array or nested list)
        
    Author: Chaitanya Shinde (231070066), Harsh Sharma (231070064)
    """
    
    def __init__(self, name: str = "Unknown") -> None:
        """
        Initialize a new VRP-IF problem instance.
        
        Args:
            name: Optional name/identifier for the problem instance
            
        Initializes all attributes with default values that can be modified later.
        """
        self.name: str = name
        self.depot: Optional[Location] = None
        self.customers: List[Location] = []
        self.intermediate_facilities: List[Location] = []
        self.vehicle_capacity: float = 0.0
        self.max_route_time: float = float("inf")
        self.max_route_length: float = float("inf")
        self.number_of_vehicles: float = float("inf")
        self.disposal_time: float = 0.0
        self.distance_matrix = None  # Will be initialized when needed

    def calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate the Euclidean distance between two locations.
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            float: Straight-line (Euclidean) distance between the locations
            
        Example:
            >>> p = ProblemInstance()
            >>> loc1 = Location(1, 0, 0)
            >>> loc2 = Location(2, 3, 4)
            >>> p.calculate_distance(loc1, loc2)
            5.0
        """
        return ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5

    def calculate_travel_time(
        self, loc1: Location, loc2: Location, speed: float = 1.0
    ) -> float:
        """
        Calculate travel time between two locations based on distance and speed.
        
        Args:
            loc1: Starting location
            loc2: Destination location
            speed: Travel speed (distance units per time unit, default: 1.0)
            
        Returns:
            float: Travel time between the locations
            
        Note:
            This is a simplified model that assumes constant speed and no
            traffic or other real-world constraints.
        """
        if speed <= 0:
            raise ValueError("Speed must be positive")
        return self.calculate_distance(loc1, loc2) / speed

    def calculate_distance_matrix(self) -> Union[np.ndarray, List[List[float]]]:
        """
        Build and cache a distance matrix for all locations.
        
        The distance matrix is a square matrix where entry (i,j) contains the
        distance from location i to location j. The order of locations is:
        [depot] + customers + intermediate_facilities.
        
        Returns:
            Union[np.ndarray, List[List[float]]]: Distance matrix where
                - result[i][j] = distance from location i to location j
                
        Note:
            - Uses NumPy if available for better performance
            - Falls back to nested lists if NumPy is not available
            - Caches the result in self.distance_matrix
            
        Author: Harsh Sharma (231070064)
        """
        # Build an ordered list of all nodes in the problem.
        nodes = []
        if self.depot is not None:
            nodes.append(self.depot)
        nodes.extend(self.customers)
        nodes.extend(self.intermediate_facilities)

        n = len(nodes)
        if n == 0:
            self.distance_matrix = None
            return self.distance_matrix

        # Use numpy for efficient matrix operations if available.
        if np is not None:
            mat = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    mat[i, j] = self.calculate_distance(nodes[i], nodes[j])
            self.distance_matrix = mat
        else:
            # Fallback to native Python lists if numpy is not installed.
            mat = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    mat[i][j] = self.calculate_distance(nodes[i], nodes[j])
            self.distance_matrix = mat

        return self.distance_matrix

    def get_total_demand(self) -> float:
        """
        Calculate the total demand across all customers.
        
        Returns:
            float: Sum of all customer demands
            
        Example:
            >>> p = ProblemInstance()
            >>> p.customers = [
            ...     Location(1, 0, 0, 5.0, "customer"),
            ...     Location(2, 1, 1, 3.0, "customer")
            ... ]
            >>> p.get_total_demand()
            8.0
        """
        return float(sum(float(customer.demand) for customer in self.customers))

    def get_min_vehicles_needed(self) -> float:
        """
        Calculate the minimum number of vehicles required to serve all customers.
        
        This provides a theoretical lower bound based on capacity constraints.
        The actual number needed may be higher due to routing constraints.
        
        Returns:
            float: Minimum number of vehicles needed (at least 1)
            
        Note:
            - Returns infinity if vehicle capacity is non-positive
            - Returns at least 1 even if there's no demand
            
        Author: Harsh Sharma (231070064)
        """
        if self.vehicle_capacity <= 0:
            return float("inf")
            
        # Use ceiling division: (total + capacity - 1) // capacity
        total_demand = self.get_total_demand()
        if total_demand <= 0:
            return 1.0
            
        return float(math.ceil(total_demand / self.vehicle_capacity))

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the problem instance.
        
        Returns:
            str: Summary of the problem instance
            
        Example:
            >>> p = ProblemInstance("Test")
            >>> p.vehicle_capacity = 100
            >>> str(p)
            'Problem: Test, Customers: 0, IFs: 0, Vehicle Capacity: 100, Min Vehicles: 1.0'
        """
        return (
            f"Problem: {self.name}, "
            f"Customers: {len(self.customers)}, "
            f"IFs: {len(self.intermediate_facilities)}, "
            f"Vehicle Capacity: {self.vehicle_capacity}, "
            f"Min Vehicles: {self.get_min_vehicles_needed():.1f}"
        )

    def is_feasible(self) -> Tuple[bool, str]:
        """
        Check if the problem instance is feasible.
        
        Performs basic validation checks to ensure the problem can potentially
        be solved. This includes checking for required components and basic
        constraint satisfaction.
        
        Returns:
            Tuple[bool, str]: 
                - First element: True if feasible, False otherwise
                - Second element: Description of any infeasibility
                
        Checks performed:
            1. Depot is defined
            2. At least one customer exists
            3. At least one intermediate facility exists
            4. Vehicle capacity is positive
            5. No customer demand exceeds vehicle capacity
            
        Author: Harsh Sharma (231070064)
        """
        if not self.depot:
            return False, "No depot defined"
        if not self.customers:
            return False, "No customers defined"
        if not self.intermediate_facilities:
            return False, "No intermediate facilities defined"
        if self.vehicle_capacity <= 0:
            return False, "Invalid vehicle capacity"
            
        # Check if any customer demand exceeds vehicle capacity
        if self.customers and max(c.demand for c in self.customers) > self.vehicle_capacity:
            return False, "Customer demand exceeds vehicle capacity"
            
        return True, "Problem instance is feasible"

    def is_route_feasible(self, route) -> Tuple[bool, str]:
        """
        Check if a route satisfies all problem constraints.
        
        Args:
            route: The route to check (must have a 'nodes' attribute)
            
        Returns:
            Tuple[bool, str]: 
                - First element: True if feasible, False otherwise
                - Second element: Description of any constraint violation
                
        Checks performed:
            1. Route is not empty
            2. Route starts and ends at the depot
            3. Vehicle capacity is not exceeded between IF visits
            4. All nodes are valid locations in the problem
            
        Note:
            - Assumes route.nodes is a list of Location objects
            - Intermediate facilities reset the vehicle load to zero
            
        Author: Harsh Sharma (231070064)
        """
        if not route.nodes:
            return False, "Empty route"

        # Check start and end at depot
        if route.nodes[0] != self.depot or route.nodes[-1] != self.depot:
            return False, "Route must start and end at depot"

        current_load = 0
        for node in route.nodes:
            if node.type == "customer":
                current_load += node.demand
                if current_load > self.vehicle_capacity + 1e-6:  # Allow for floating point imprecision
                    return False, f"Capacity exceeded at customer {node.id}"
            elif node.type == "if":
                # Reset load at intermediate facilities
                current_load = 0

        return True, "Route is feasible"

    def get_customer_data(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Extract customer data for visualization.
        
        Returns:
            Dict[str, List[Tuple[float, float, float]]]: 
                Dictionary with keys 'locations' and 'demands' containing
                (x, y, demand) tuples for each customer.
                
        Example:
            >>> p = ProblemInstance()
            >>> p.customers = [
            ...     Location(1, 10, 20, 5.0, "customer"),
            ...     Location(2, 30, 40, 10.0, "customer")
            ... ]
            >>> p.get_customer_data()
            {
                'locations': [(10.0, 20.0, 5.0), (30.0, 40.0, 10.0)],
                'demands': [5.0, 10.0]
            }
            
        Author: Harsh Sharma (231070064)
        """
        return {(c.x, c.y): c.demand for c in self.customers}
