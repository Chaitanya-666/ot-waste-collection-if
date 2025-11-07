"""
Utility functions for the project

This module provides comprehensive utilities for:
- Route visualization and plotting
- Performance metrics calculation
- Data generation for testing
- Solution analysis and reporting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import random
import math
from typing import List, Dict, Tuple, Optional
from .solution import Solution, Route
from .problem import ProblemInstance, Location


class RouteVisualizer:
    """Visualize routes and solutions using matplotlib"""

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

    def plot_solution(self, solution: Solution, title: str = "Waste Collection Routes"):
        """Plot the complete solution with all routes"""
        self.ax.clear()
        self.ax.set_title(title, fontsize=16, fontweight="bold")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Plot depot
        depot = self.problem.depot
        self.ax.plot(depot.x, depot.y, "ks", markersize=15, label="Depot", zorder=5)
        self.ax.annotate(
            "Depot",
            (depot.x, depot.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

        # Plot intermediate facilities
        for i, if_node in enumerate(self.problem.intermediate_facilities):
            self.ax.plot(
                if_node.x,
                if_node.y,
                "D",
                color="orange",
                markersize=12,
                label="IF" if i == 0 else "",
                zorder=4,
            )
            self.ax.annotate(
                "IF",
                (if_node.x, if_node.y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        # Plot customers
        for customer in self.problem.customers:
            self.ax.plot(
                customer.x, customer.y, "o", color="lightblue", markersize=8, zorder=3
            )
            self.ax.annotate(
                f"C{customer.id}",
                (customer.x, customer.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )

        # Plot routes
        for i, route in enumerate(solution.routes):
            if not route.nodes:
                continue

            color = self.colors[i % len(self.colors)]
            route_x = [node.x for node in route.nodes]
            route_y = [node.y for node in route.nodes]

            # Plot route line
            self.ax.plot(
                route_x,
                route_y,
                "-",
                color=color,
                linewidth=2,
                alpha=0.7,
                label=f"Vehicle {i + 1}",
            )

            # Plot route points
            for j, node in enumerate(route.nodes):
                if node.type == "depot":
                    continue
                elif node.type == "if":
                    self.ax.plot(
                        node.x, node.y, "D", color=color, markersize=10, zorder=4
                    )
                else:  # customer
                    self.ax.plot(
                        node.x, node.y, "o", color=color, markersize=8, zorder=3
                    )

            # Add route info
            mid_idx = len(route.nodes) // 2
            if mid_idx < len(route.nodes):
                mid_node = route.nodes[mid_idx]
                self.ax.annotate(
                    f"V{i + 1}\n{route.total_distance:.1f}",
                    (mid_node.x, mid_node.y),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                )

        # Add legend
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        # Set equal aspect ratio and grid
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self.fig

    def plot_convergence(
        self, convergence_history: List[float], title: str = "ALNS Convergence"
    ):
        """Plot convergence history"""
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_history, "b-", linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Solution Cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


class PerformanceAnalyzer:
    """Analyze solution performance and generate metrics"""

    def __init__(self, problem: ProblemInstance):
        self.problem = problem

    def analyze_solution(self, solution: Solution) -> Dict:
        """Comprehensive solution analysis"""
        analysis = {
            "total_cost": solution.total_cost,
            "total_distance": solution.total_distance,
            "total_time": solution.total_time,
            "num_vehicles": len(solution.routes),
            "num_unassigned": len(solution.unassigned_customers),
            "vehicle_utilization": [],
            "if_visits": 0,
            "route_details": [],
            "efficiency_metrics": {},
        }

        total_capacity = self.problem.vehicle_capacity * len(solution.routes)
        total_demand = sum(customer.demand for customer in self.problem.customers)

        for i, route in enumerate(solution.routes):
            route_analysis = {
                "vehicle_id": i + 1,
                "distance": route.total_distance,
                "time": route.total_time,
                "nodes": len(route.nodes),
                "customers_served": sum(
                    1 for node in route.nodes if node.type == "customer"
                ),
                "if_visits": sum(1 for node in route.nodes if node.type == "if"),
                "max_load": max(route.loads) if route.loads else 0,
                "load_utilization": max(route.loads) / self.problem.vehicle_capacity
                if route.loads
                else 0,
            }

            analysis["route_details"].append(route_analysis)
            analysis["if_visits"] += route_analysis["if_visits"]
            analysis["vehicle_utilization"].append(route_analysis["load_utilization"])

        # Calculate efficiency metrics
        analysis["efficiency_metrics"] = {
            "distance_efficiency": total_demand / solution.total_distance
            if solution.total_distance > 0
            else 0,
            "capacity_utilization": total_demand / total_capacity
            if total_capacity > 0
            else 0,
            "vehicle_efficiency": total_demand
            / (len(solution.routes) * self.problem.vehicle_capacity),
            "if_efficiency": analysis["if_visits"] / len(solution.routes)
            if solution.routes
            else 0,
        }

        return analysis

    def generate_report(self, solution: Solution) -> str:
        """Generate a human-readable performance report"""
        analysis = self.analyze_solution(solution)

        report = f"""
=== WASTE COLLECTION PERFORMANCE REPORT ===

SOLUTION OVERVIEW:
- Total Cost: {analysis["total_cost"]:.2f}
- Total Distance: {analysis["total_distance"]:.2f} units
- Total Time: {analysis["total_time"]:.2f} units
- Vehicles Used: {analysis["num_vehicles"]}
- Unassigned Customers: {analysis["num_unassigned"]}
- IF Visits: {analysis["if_visits"]}

VEHICLE PERFORMANCE:
"""

        for route in analysis["route_details"]:
            report += f"""
Vehicle {route["vehicle_id"]}:
  - Distance: {route["distance"]:.2f}
  - Time: {route["time"]:.2f}
  - Customers Served: {route["customers_served"]}
  - IF Visits: {route["if_visits"]}
  - Max Load: {route["max_load"]:.1f}/{self.problem.vehicle_capacity}
  - Load Utilization: {route["load_utilization"]:.1%}
"""

        report += f"""
EFFICIENCY METRICS:
- Distance Efficiency: {analysis["efficiency_metrics"]["distance_effency"]:.3f} (demand/distance)
- Capacity Utilization: {analysis["efficiency_metrics"]["capacity_utilization"]:.1%}
- Vehicle Efficiency: {analysis["efficiency_metrics"]["vehicle_efficiency"]:.1%}
- IF Efficiency: {analysis["efficiency_metrics"]["if_efficiency"]:.1f} visits/vehicle

SOLUTION QUALITY: {"EXCELLENT" if analysis["num_unassigned"] == 0 else "NEEDS IMPROVEMENT"}
"""

        return report


class DataGenerator:
    """Generate synthetic problem instances for testing"""

    @staticmethod
    def generate_instance(
        name: str,
        n_customers: int,
        n_ifs: int,
        vehicle_capacity: int = 20,
        area_size: int = 100,
        demand_range: Tuple[int, int] = (1, 10),
        seed: Optional[int] = None,
    ) -> ProblemInstance:
        """Generate a synthetic problem instance"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        problem = ProblemInstance(name)
        problem.vehicle_capacity = vehicle_capacity
        problem.number_of_vehicles = max(2, n_customers // 5)  # Rule of thumb
        problem.disposal_time = 2

        # Add depot at center
        depot = Location(0, area_size // 2, area_size // 2, 0, "depot")
        problem.depot = depot

        # Add intermediate facilities
        for i in range(n_ifs):
            x = random.randint(10, area_size - 10)
            y = random.randint(10, area_size - 10)
            if_node = Location(i + 1000, x, y, 0, "if")
            problem.intermediate_facilities.append(if_node)

        # Add customers
        for i in range(n_customers):
            x = random.randint(5, area_size - 5)
            y = random.randint(5, area_size - 5)
            demand = random.randint(*demand_range)
            customer = Location(i + 1, x, y, demand, "customer")
            problem.customers.append(customer)

        return problem

    @staticmethod
    def generate_instances_from_file(filename: str) -> List[ProblemInstance]:
        """Generate instances from a configuration file"""
        instances = []

        try:
            with open(filename, "r") as f:
                config = json.load(f)

            for instance_config in config.get("instances", []):
                instance = DataGenerator.generate_instance(
                    name=instance_config["name"],
                    n_customers=instance_config["customers"],
                    n_ifs=instance_config["ifs"],
                    vehicle_capacity=instance_config.get("capacity", 20),
                    area_size=instance_config.get("area_size", 100),
                    demand_range=tuple(instance_config.get("demand_range", [1, 10])),
                    seed=instance_config.get("seed"),
                )
                instances.append(instance)

        except FileNotFoundError:
            print(
                f"Warning: Configuration file {filename} not found. Using default instance."
            )
            # Generate a default instance
            instances.append(DataGenerator.generate_instance("Default", 10, 1))

        return instances

    @staticmethod
    def create_config_template(filename: str = "instances_config.json"):
        """Create a template configuration file"""
        template = {
            "instances": [
                {
                    "name": "Small Instance",
                    "customers": 6,
                    "ifs": 1,
                    "capacity": 20,
                    "area_size": 100,
                    "demand_range": [1, 10],
                    "seed": 42,
                },
                {
                    "name": "Medium Instance",
                    "customers": 20,
                    "ifs": 2,
                    "capacity": 30,
                    "area_size": 150,
                    "demand_range": [1, 15],
                    "seed": 123,
                },
                {
                    "name": "Large Instance",
                    "customers": 50,
                    "ifs": 3,
                    "capacity": 40,
                    "area_size": 200,
                    "demand_range": [1, 20],
                    "seed": 456,
                },
            ]
        }

        with open(filename, "w") as f:
            json.dump(template, f, indent=2)

        print(f"Configuration template created: {filename}")


def save_solution_to_file(solution: Solution, filename: str):
    """Save solution to JSON file for later analysis"""
    import json
    from datetime import datetime

    solution_data = {
        "timestamp": datetime.now().isoformat(),
        "total_cost": solution.total_cost,
        "total_distance": solution.total_distance,
        "total_time": solution.total_time,
        "num_vehicles": len(solution.routes),
        "unassigned_customers": list(solution.unassigned_customers),
        "routes": [],
    }

    for i, route in enumerate(solution.routes):
        route_data = {
            "vehicle_id": i + 1,
            "nodes": [
                {"id": node.id, "type": node.type, "x": node.x, "y": node.y}
                for node in route.nodes
            ],
            "total_distance": route.total_distance,
            "total_time": route.total_time,
            "loads": route.loads,
        }
        solution_data["routes"].append(route_data)

    with open(filename, "w") as f:
        json.dump(solution_data, f, indent=2)

    print(f"Solution saved to: {filename}")


def load_solution_from_file(filename: str) -> Dict:
    """Load solution data from JSON file"""
    with open(filename, "r") as f:
        return json.load(f)


def benchmark_algorithm(
    problem: ProblemInstance, max_iterations_list: List[int] = [100, 200, 500, 1000]
) -> Dict:
    """Benchmark algorithm performance with different iteration counts"""
    from .alns import ALNS

    results = {}

    for max_iterations in max_iterations_list:
        print(f"Benchmarking with {max_iterations} iterations...")

        solver = ALNS(problem)
        start_time = time.time()
        solution = solver.run(max_iterations=max_iterations)
        end_time = time.time()

        results[max_iterations] = {
            "cost": solution.total_cost,
            "distance": solution.total_distance,
            "time": end_time - start_time,
            "vehicles": len(solution.routes),
            "unassigned": len(solution.unassigned_customers),
        }

    return results
