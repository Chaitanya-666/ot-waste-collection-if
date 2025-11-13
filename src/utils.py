"""
Utility functions for the project

This module provides comprehensive utilities for:
- Route visualization and plotting (including optional live plotting updates)
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
import time
from typing import List, Dict, Tuple, Optional, Callable
from .solution import Solution, Route
from .problem import ProblemInstance, Location


class RouteVisualizer:
    """Visualize routes and solutions using matplotlib.

    This visualizer supports:
      - static plotting via `plot_solution` and `plot_convergence`
      - live updates via `start_live` / `update_live` / `stop_live`
        where the ALNS solver can provide iteration callbacks to update plots.
    """

    def __init__(
        self,
        problem: ProblemInstance,
        figsize: Tuple[int, int] = (12, 10),
        live: bool = False,
    ):
        self.problem = problem
        self.fig = plt.figure(figsize=figsize)
        # Use a grid: left axes for routes, right (bottom) axes for convergence
        self.ax = self.fig.add_subplot(2, 1, 1)
        self.conv_ax = self.fig.add_subplot(2, 1, 2)
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
        self._live = bool(live)
        if self._live:
            # enable interactive mode
            try:
                plt.ion()
            except Exception:
                pass

    def _draw_base_map(self, title: str = "Waste Collection Routes"):
        """Draw depot, IFs and customers (static background) onto self.ax"""
        self.ax.clear()
        self.ax.set_title(title, fontsize=14, fontweight="bold")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        depot = self.problem.depot
        if depot is not None:
            self.ax.plot(depot.x, depot.y, "ks", markersize=12, label="Depot", zorder=5)
            self.ax.annotate(
                "Depot",
                (depot.x, depot.y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        # Plot intermediate facilities
        for i, if_node in enumerate(self.problem.intermediate_facilities):
            self.ax.plot(
                if_node.x,
                if_node.y,
                "D",
                color="orange",
                markersize=10,
                label="IF" if i == 0 else "",
                zorder=4,
            )
            self.ax.annotate(
                "IF",
                (if_node.x, if_node.y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

        # Plot customers (background)
        for customer in self.problem.customers:
            self.ax.plot(
                customer.x, customer.y, "o", color="lightblue", markersize=6, zorder=3
            )
            self.ax.annotate(
                f"C{customer.id}",
                (customer.x, customer.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

        self.ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

    def plot_solution(self, solution: Solution, title: str = "Waste Collection Routes"):
        """Plot the complete solution with all routes (static)."""
        # Draw base map
        self._draw_base_map(title)

        # Plot routes on top
        # Plot routes (show only routes that serve at least one customer)
        display_idx = 0
        for i, route in enumerate(solution.routes):
            # count customers on route (defensive access)
            customers_on_route = sum(
                1 for n in route.nodes if getattr(n, "type", None) == "customer"
            )
            if customers_on_route == 0:
                # skip empty placeholder routes
                continue

            color = self.colors[display_idx % len(self.colors)]
            route_x = [node.x for node in route.nodes]
            route_y = [node.y for node in route.nodes]

            # Plot route line
            self.ax.plot(
                route_x,
                route_y,
                "-",
                color=color,
                linewidth=2,
                alpha=0.8,
                label=f"Vehicle {display_idx + 1}",
            )

            # Plot route points and annotate IFs/customers
            for j, node in enumerate(route.nodes):
                if getattr(node, "type", None) == "depot":
                    continue
                elif getattr(node, "type", None) == "if":
                    self.ax.plot(
                        node.x, node.y, "D", color=color, markersize=10, zorder=4
                    )
                else:  # customer
                    self.ax.plot(
                        node.x, node.y, "o", color=color, markersize=8, zorder=3
                    )

            # Add route info (distance) near route midpoint
            mid_idx = len(route.nodes) // 2
            if mid_idx < len(route.nodes):
                mid_node = route.nodes[mid_idx]
                self.ax.annotate(
                    f"V{display_idx + 1}\n{route.total_distance:.1f}",
                    (mid_node.x, mid_node.y),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                )

            display_idx += 1

        plt.tight_layout()
        return self.fig

    def plot_convergence(
        self, convergence_history: List[float], title: str = "ALNS Convergence"
    ):
        """Plot convergence history on the convergence axis (static)."""
        self.conv_ax.clear()
        self.conv_ax.plot(convergence_history, "b-", linewidth=1.5)
        self.conv_ax.set_title(title, fontsize=12, fontweight="bold")
        self.conv_ax.set_xlabel("Iteration")
        self.conv_ax.set_ylabel("Solution Cost")
        self.conv_ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self.fig

    # Live plotting API
    def start_live(self, title: str = "Live Waste Collection Routes"):
        """Prepare the figure for live updates. Call once before iterative updates."""
        self._draw_base_map(title)
        self.conv_ax.clear()
        if self._live:
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception:
                pass
        return self.fig

    def update_live(
        self,
        solution: Solution,
        convergence_history: Optional[List[float]] = None,
        pause: float = 0.01,
    ):
        """Update the live plot with the current solution and convergence history.

        This method is intentionally lightweight and defensive: failures here won't
        interrupt the solver. It clears previous dynamic route plots and redraws.
        """
        try:
            # redraw base map but keep it lightweight
            self._draw_base_map("Live - Waste Collection Routes")

            # Draw routes (same as plot_solution but onto existing axes)
            for i, route in enumerate(solution.routes):
                if not route.nodes:
                    continue
                color = self.colors[i % len(self.colors)]
                route_x = [node.x for node in route.nodes]
                route_y = [node.y for node in route.nodes]
                self.ax.plot(
                    route_x, route_y, "-", color=color, linewidth=1.5, alpha=0.9
                )
                # plot IFs/customers for the route overlay
                for node in route.nodes:
                    if node.type == "if":
                        self.ax.plot(node.x, node.y, "D", color=color, markersize=6)
                    elif node.type == "customer":
                        self.ax.plot(node.x, node.y, "o", color=color, markersize=4)

            # Update convergence
            if convergence_history is not None:
                self.conv_ax.clear()
                self.conv_ax.plot(convergence_history, "b-", linewidth=1.0)
                self.conv_ax.set_title(
                    "ALNS Convergence", fontsize=12, fontweight="bold"
                )
                self.conv_ax.set_xlabel("Iteration")
                self.conv_ax.set_ylabel("Solution Cost")
                self.conv_ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if self._live:
                try:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                except Exception:
                    pass
        except Exception:
            # fail-safe: do not allow visualization errors to stop solver
            pass

    def stop_live(self):
        """Stop live plotting and switch back to blocking mode."""
        if self._live:
            try:
                plt.ioff()
            except Exception:
                pass
        try:
            self.fig.canvas.draw()
        except Exception:
            pass
        return self.fig


class PerformanceAnalyzer:
    """Analyze solution performance and generate metrics"""

    def __init__(self, problem: ProblemInstance):
        self.problem = problem

    def analyze_solution(self, solution: Solution) -> Dict:
        """Comprehensive solution analysis

        This analyzer filters out empty placeholder routes (those serving zero customers)
        when computing vehicle-level metrics and efficiencies so reports focus on
        vehicles that actually perform service.
        """
        analysis = {
            "total_cost": solution.total_cost,
            "total_distance": solution.total_distance,
            "total_time": solution.total_time,
            # number of vehicles considered = vehicles actually serving customers (non-empty)
            "num_vehicles": 0,
            "num_unassigned": len(solution.unassigned_customers),
            "vehicle_utilization": [],
            "if_visits": 0,
            "route_details": [],
            "efficiency_metrics": {},
        }

        # consider only routes that serve at least one customer for vehicle-level metrics
        served_routes = [
            r
            for r in solution.routes
            if any(getattr(n, "type", None) == "customer" for n in r.nodes)
        ]
        analysis["num_vehicles"] = len(served_routes)

        # Calculate total demand ONLY from served customers (exclude unassigned)
        served_customer_ids = set()
        for r in served_routes:
            for n in r.nodes:
                if getattr(n, "type", None) == "customer":
                    served_customer_ids.add(n.id)
        
        total_demand = sum(
            customer.demand 
            for customer in self.problem.customers 
            if customer.id in served_customer_ids
        )
        
        # Calculate minimum vehicles needed for this demand (theoretical minimum)
        min_vehicles_needed = max(1, int((total_demand + self.problem.vehicle_capacity - 1) // self.problem.vehicle_capacity))
        
        # Use the larger of actual vehicles or minimum needed for capacity calculation
        # This ensures efficiency never exceeds 100% even with IF visits
        effective_vehicles = max(len(served_routes), min_vehicles_needed)
        total_capacity = self.problem.vehicle_capacity * effective_vehicles

        for idx, route in enumerate(served_routes):
            route_analysis = {
                "vehicle_id": idx + 1,
                "distance": route.total_distance,
                "time": route.total_time,
                "nodes": len(route.nodes),
                "customers_served": sum(
                    1
                    for node in route.nodes
                    if getattr(node, "type", None) == "customer"
                ),
                "if_visits": sum(
                    1 for node in route.nodes if getattr(node, "type", None) == "if"
                ),
                "max_load": max(route.loads) if route.loads else 0,
                "load_utilization": (max(route.loads) / self.problem.vehicle_capacity)
                if route.loads
                else 0,
            }

            analysis["route_details"].append(route_analysis)
            analysis["if_visits"] += route_analysis["if_visits"]
            analysis["vehicle_utilization"].append(route_analysis["load_utilization"])

        # Calculate efficiency metrics (guard denominators)
        num_served_routes = len(served_routes)
        analysis["efficiency_metrics"] = {
            "distance_efficiency": total_demand / solution.total_distance
            if solution.total_distance > 0
            else 0,
            "capacity_utilization": total_demand / total_capacity
            if total_capacity > 0
            else 0,
            "vehicle_efficiency": (total_demand / total_capacity) if total_capacity > 0 else 0,
            "if_efficiency": analysis["if_visits"] / num_served_routes
            if num_served_routes > 0
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

        # Be robust when efficiency keys are missing or zero (avoid KeyError / formatting issues)
        eff = analysis.get("efficiency_metrics", {})
        dist_eff = eff.get("distance_efficiency", 0.0)
        cap_util = eff.get("capacity_utilization", 0.0)
        veh_eff = eff.get("vehicle_efficiency", 0.0)
        if_eff = eff.get("if_efficiency", 0.0)

        report += f"""
EFFICIENCY METRICS:
- Distance Efficiency: {dist_eff:.3f} (demand/distance)
- Capacity Utilization: {cap_util:.1%}
- Vehicle Efficiency: {veh_eff:.1%}
- IF Efficiency: {if_eff:.1f} visits/vehicle

SOLUTION QUALITY: {"EXCELLENT" if analysis["num_unassigned"] == 0 else "NEEDS IMPROVEMENT"}
"""

        return report


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
    from alns import ALNS

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
