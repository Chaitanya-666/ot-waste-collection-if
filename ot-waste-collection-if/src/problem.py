"""
Problem definition for VRP with Intermediate Facilities
"""

class Location:
    def __init__(self, id, x, y, demand=0, location_type="customer"):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.type = location_type

    def __repr__(self):
        return f"{self.type.capitalize()}({self.id}, ({self.x},{self.y}), demand={self.demand})"

class ProblemInstance:
    def __init__(self, name="Unknown"):
        self.name = name
        self.depot = None
        self.customers = []
        self.intermediate_facilities = []
        self.vehicle_capacity = 0

    def calculate_distance(self, loc1, loc2):
        return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)**0.5

    def __str__(self):
        return f"Problem: {self.name}, Customers: {len(self.customers)}, IFs: {len(self.intermediate_facilities)}"
