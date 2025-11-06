"""
Solution representation for VRP with IFs

Classes:
    Route: Represents a single vehicle route with load tracking
    Solution: Complete solution representation with multiple routes
"""


class Route:
    def __init__(self):
        self.nodes = []  # Sequence of locations (depot, customers, IFs)
        self.loads = []  # Load after visiting each node
        self.total_distance = 0.0
        self.total_time = 0.0
        self.vehicle_id = None

    def calculate_metrics(self, problem):
        """Recalculate route metrics"""
        self.total_distance = 0.0
        self.total_time = 0.0
        current_load = 0

        for i in range(len(self.nodes) - 1):
            current = self.nodes[i]
            next_node = self.nodes[i + 1]

            # Add travel distance and time
            distance = problem.calculate_distance(current, next_node)
            self.total_distance += distance
            self.total_time += problem.calculate_travel_time(current, next_node)

            # Add service time
            if next_node.type == "customer":
                self.total_time += next_node.service_time
                current_load += next_node.demand
            elif next_node.type == "if":
                self.total_time += problem.disposal_time
                current_load = 0

            self.loads[i + 1] = current_load

    def is_feasible(self, problem):
        """Check route feasibility"""
        if (
            not self.nodes
            or self.nodes[0] != problem.depot
            or self.nodes[-1] != problem.depot
        ):
            return False, "Invalid depot visits"

        current_load = 0
        current_time = 0

        for i in range(len(self.nodes) - 1):
            current = self.nodes[i]
            next_node = self.nodes[i + 1]

            # Check capacity
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

    def __repr__(self):
        route_str = " -> ".join([f"{node.type[0]}{node.id}" for node in self.nodes])
        return f"Route(Distance: {self.total_distance:.2f}, Time: {self.total_time:.2f}): {route_str}"


class Solution:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.routes = []
        self.total_cost = 0.0
        self.total_distance = 0.0
        self.total_time = 0.0
        # Keep unassigned customers as a set of customer IDs to avoid identity/mutation issues
        # with Location objects across copy/deepcopy and different modules.
        self.unassigned_customers = set(c.id for c in problem_instance.customers)

    def calculate_metrics(self):
        """Recalculate solution metrics"""
        self.total_distance = 0.0
        self.total_time = 0.0

        for route in self.routes:
            route.calculate_metrics(self.problem)
            self.total_distance += route.total_distance
            self.total_time += route.total_time

        # Cost can include multiple factors
        self.total_cost = self.total_distance  # Can be enhanced with time/vehicle costs

    def is_feasible(self):
        """Check if solution is feasible"""
        # Check vehicle limit
        if len(self.routes) > self.problem.number_of_vehicles:
            return False, "Too many vehicles used"

        # Check customer coverage (use IDs; unassigned_customers stores IDs)
        served_ids = set()
        for route in self.routes:
            for node in route.nodes:
                if node.type == "customer":
                    if node.id in served_ids:
                        return False, f"Customer {node.id} served multiple times"
                    served_ids.add(node.id)

        # Build expected served set from all problem customers minus unassigned IDs
        all_customer_ids = set(c.id for c in self.problem.customers)
        expected_served_ids = all_customer_ids - set(self.unassigned_customers)

        if served_ids != expected_served_ids:
            return False, "Not all customers are served"

        # Check individual routes
        for route in self.routes:
            feasible, message = route.is_feasible(self.problem)
            if not feasible:
                return False, f"Route {route.vehicle_id}: {message}"

        return True, "Solution is feasible"

    def copy(self):
        """Create a deep copy of the solution"""
        import copy

        return copy.deepcopy(self)

    def __repr__(self):
        status = "Feasible" if self.is_feasible()[0] else "Infeasible"
        return (
            f"Solution({status}, Cost: {self.total_cost:.2f}, "
            f"Distance: {self.total_distance:.2f}, "
            f"Time: {self.total_time:.2f}, "
            f"Routes: {len(self.routes)}, "
            f"Unassigned: {len(self.unassigned_customers)})"
        )
