"""
Problem definition for VRP with Intermediate Facilities

Classes:
    Location: Represents a node (customer, depot, or IF) in the problem
    ProblemInstance: Defines the complete VRP-IF problem instance
"""

from typing import List, Optional, Tuple, Union

# Try to import numpy if available for distance matrix convenience; otherwise fall back to lists
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


class Location:
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        demand: float = 0.0,
        location_type: str = "customer",
        service_time: float = 0.0,
    ) -> None:
        self.id: int = id
        self.x: float = x
        self.y: float = y
        self.demand: float = demand
        self.type: str = location_type
        self.service_time: float = service_time  # Time needed to service this location

    def __repr__(self) -> str:
        return f"{self.type.capitalize()}({self.id}, ({self.x},{self.y}), demand={self.demand})"


class ProblemInstance:
    def __init__(self, name: str = "Unknown") -> None:
        self.name: str = name
        self.depot: Optional[Location] = None
        self.customers: List[Location] = []
        self.intermediate_facilities: List[Location] = []
        self.vehicle_capacity: float = 0.0
        self.max_route_time: float = float("inf")  # Maximum duration of any route
        self.max_route_length: float = float("inf")  # Maximum distance of any route
        self.number_of_vehicles: float = float("inf")  # Available vehicles
        self.disposal_time: float = 0.0  # Time needed at intermediate facilities

        # distance_matrix will be created by calculate_distance_matrix()
        # It is set to None until the method is called.
        # When present it will be a numpy.ndarray if numpy is available, otherwise a nested list.
        self.distance_matrix = None

    def calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate Euclidean distance between two locations"""
        return ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5

    def calculate_travel_time(
        self, loc1: Location, loc2: Location, speed: float = 1.0
    ) -> float:
        """Calculate travel time between locations given speed"""
        return self.calculate_distance(loc1, loc2) / speed

    def calculate_distance_matrix(self):
        """
        Build a full distance matrix for this problem instance.

        The ordering used is: [depot] + customers + intermediate_facilities.
        The result is stored in `self.distance_matrix` as a numpy array if numpy
        is available, otherwise as a nested Python list.
        """
        # Build ordered list of nodes
        nodes = []
        if self.depot is not None:
            nodes.append(self.depot)
        nodes.extend(self.customers)
        nodes.extend(self.intermediate_facilities)

        n = len(nodes)
        if n == 0:
            self.distance_matrix = None
            return self.distance_matrix

        # Compute distances
        if np is not None:
            mat = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    mat[i, j] = self.calculate_distance(nodes[i], nodes[j])
            self.distance_matrix = mat
        else:
            mat = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    mat[i][j] = self.calculate_distance(nodes[i], nodes[j])
            self.distance_matrix = mat

        return self.distance_matrix

    def get_total_demand(self) -> float:
        """Calculate total demand across all customers"""
        return float(sum(float(customer.demand) for customer in self.customers))

    def get_min_vehicles_needed(self) -> float:
        """Calculate minimum number of vehicles needed based on demand.

        Returns a float so `float('inf')` can be returned for infeasible settings
        (e.g., vehicle_capacity <= 0).
        """
        if self.vehicle_capacity <= 0:
            return float("inf")
        # Use ceil-like integer division but return a float as legacy callers expect
        total = self.get_total_demand()
        per_vehicle = float(self.vehicle_capacity)
        needed = (
            int((total + per_vehicle - 1) // per_vehicle)
            if per_vehicle > 0
            else float("inf")
        )
        return float(max(1, needed))

    def __str__(self) -> str:
        return (
            f"Problem: {self.name}, Customers: {len(self.customers)}, "
            f"IFs: {len(self.intermediate_facilities)}, "
            f"Vehicle Capacity: {self.vehicle_capacity}, "
            f"Min Vehicles: {self.get_min_vehicles_needed()}"
        )

    def is_feasible(self) -> Tuple[bool, str]:
        """Check if problem instance is feasible"""
        if not self.depot:
            return False, "No depot defined"
        if not self.customers:
            return False, "No customers defined"
        if not self.intermediate_facilities:
            return False, "No intermediate facilities defined"
        if self.vehicle_capacity <= 0:
            return False, "Invalid vehicle capacity"
        if (
            self.customers
            and max(c.demand for c in self.customers) > self.vehicle_capacity
        ):
            return False, "Customer demand exceeds vehicle capacity"
        return True, "Problem instance is feasible"

    def is_route_feasible(self, route) -> Tuple[bool, str]:
        """Check if a single route is feasible"""
        if not route.nodes:
            return False, "Empty route"

        if route.nodes[0] != self.depot or route.nodes[-1] != self.depot:
            return False, "Route must start and end at depot"

        current_load = 0
        for i, node in enumerate(route.nodes):
            if node.type == "customer":
                current_load += node.demand
                if current_load > self.vehicle_capacity:
                    return False, f"Capacity exceeded at customer {node.id}"
            elif node.type == "if":
                current_load = 0

        return True, "Route is feasible"
