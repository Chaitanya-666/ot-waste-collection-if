"""
Problem definition for VRP with Intermediate Facilities

Classes:
    Location: Represents a node (customer, depot, or IF) in the problem
    ProblemInstance: Defines the complete VRP-IF problem instance
"""

from typing import List, Optional, Tuple, Union


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

    def calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate Euclidean distance between two locations"""
        return ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5

    def calculate_travel_time(
        self, loc1: Location, loc2: Location, speed: float = 1.0
    ) -> float:
        """Calculate travel time between locations given speed"""
        return self.calculate_distance(loc1, loc2) / speed

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
        return float(
            max(
                1,
                int(
                    (self.get_total_demand() + self.vehicle_capacity - 1)
                    // self.vehicle_capacity
                ),
            )
        )

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
