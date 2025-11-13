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
                current_load = 0

            # store load corresponding to next_node position (i+1)
            if i + 1 < len(self.loads):
                self.loads[i + 1] = current_load

        # If there is a single-node route (depot only) ensure loads length is correct
        if self.nodes and len(self.loads) < len(self.nodes):
            self.loads += [0] * (len(self.nodes) - len(self.loads))

    def is_feasible(self, problem):
        """Check route feasibility"""
        # Empty routes or routes with only depot are considered feasible but should be filtered
        if not self.nodes:
            return True, "Empty route (should be removed)"
        
        # Single node routes (only depot) are feasible
        if len(self.nodes) == 1:
            if self.nodes[0].type == "depot":
                return True, "Single depot route (empty)"
            else:
                return False, "Single node route without depot"
        
        # Check depot at start and end (compare by type and id, not object identity)
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
    def __init__(self, problem_instance=None):
        """
        Solution container.

        Args:
            problem_instance: optional ProblemInstance. If provided, the solution will
                initialize `unassigned_customers` as the set of all customer IDs from the
                problem. If not provided, `unassigned_customers` starts empty and the
                `problem` attribute can be set later by the caller.
        """
        self.problem = problem_instance
        self.routes = []
        self.total_cost = 0.0
        self.total_distance = 0.0
        self.total_time = 0.0
        # Keep unassigned customers as a set of customer IDs to avoid identity/mutation issues.
        # If a problem instance is provided, initialize to all customer IDs; otherwise empty.
        if problem_instance is not None:
            try:
                self.unassigned_customers = set(
                    c.id for c in problem_instance.customers
                )
            except Exception:
                # defensive fallback if problem structure is unexpected
                self.unassigned_customers = set()
        else:
            self.unassigned_customers = set()

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

    def is_feasible(self, problem=None):
        """Check if solution is feasible.

        Optional `problem` parameter allows callers to provide a ProblemInstance for
        the feasibility check. If omitted, `self.problem` is used. If neither is
        present the method returns infeasible with a message.
        """
        # Determine which problem instance to use
        problem = problem if problem is not None else self.problem
        if problem is None:
            return False, "No problem instance provided for feasibility check"

        # Filter out empty routes (depot-only routes with no customers)
        non_empty_routes = [
            r for r in self.routes 
            if r.nodes and any(getattr(n, "type", None) == "customer" for n in r.nodes)
        ]
        
        # Check vehicle limit (respect the problem's available vehicles)
        if len(non_empty_routes) > problem.number_of_vehicles:
            return False, "Too many vehicles used"

        # Check customer coverage (use IDs; unassigned_customers stores IDs)
        served_ids = set()
        for route in non_empty_routes:
            for node in route.nodes:
                if getattr(node, "type", None) == "customer":
                    if node.id in served_ids:
                        return False, f"Customer {node.id} served multiple times"
                    served_ids.add(node.id)

        # Build expected served set from all problem customers minus unassigned IDs
        try:
            all_customer_ids = set(c.id for c in problem.customers)
        except Exception:
            return False, "Problem instance missing customers for feasibility check"

        expected_served_ids = all_customer_ids - set(self.unassigned_customers)

        if served_ids != expected_served_ids:
            return False, "Not all customers are served"

        # Check individual routes (only non-empty ones)
        for route in non_empty_routes:
            feasible, message = route.is_feasible(problem)
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
