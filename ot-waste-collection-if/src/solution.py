"""
Solution representation for VRP with IFs
"""

class Route:
    def __init__(self):
        self.nodes = []
        self.loads = []
        self.total_distance = 0.0

    def __repr__(self):
        route_str = " -> ".join([f"{node.type[0]}{node.id}" for node in self.nodes])
        return f"Route(Distance: {self.total_distance:.2f}): {route_str}"

class Solution:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.routes = []
        self.total_cost = 0.0

    def __repr__(self):
        return f"Solution(Cost: {self.total_cost:.2f}, Routes: {len(self.routes)})"
