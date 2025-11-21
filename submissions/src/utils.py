# Author: Harsh Sharma (231070064)
#
# This file provides utility classes for visualization and performance analysis.
# - RouteVisualizer: Uses matplotlib to create plots of the solution routes
#   and the algorithm's convergence over time. Supports both static and live plotting.
# - PerformanceAnalyzer: Calculates a variety of metrics to assess the quality
#   of a solution, such as cost, efficiency, and vehicle utilization.
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
    """
    Visualizes VRP solutions and convergence history using matplotlib.
    Supports both static image generation and live, interactive plotting
    during the optimization process.
    """

    def __init__(
        self,
        problem: ProblemInstance,
        figsize: Tuple[int, int] = (12, 10),
        live: bool = False,
    ):
        self.problem = problem
        self.fig = plt.figure(figsize=figsize)
        # Create a grid for two plots: one for the routes, one for convergence.
        self.ax = self.fig.add_subplot(2, 1, 1)
        self.conv_ax = self.fig.add_subplot(2, 1, 2)
        self.colors = plt.cm.get_cmap('tab10').colors
        self._live = bool(live)
        if self._live:
            plt.ion() # Turn on interactive mode for live plotting.

    def _draw_base_map(self, title: str = "Waste Collection Routes"):
        """Draws the static elements of the map (depot, customers, IFs)."""
        self.ax.clear()
        self.ax.set_title(title, fontsize=14, fontweight="bold")

        # Plot the depot.
        depot = self.problem.depot
        if depot is not None:
            self.ax.plot(depot.x, depot.y, "ks", markersize=12, label="Depot", zorder=5)
            self.ax.annotate("Depot", (depot.x, depot.y), xytext=(5, 5), textcoords="offset points")

        # Plot intermediate facilities.
        for i, if_node in enumerate(self.problem.intermediate_facilities):
            self.ax.plot(if_node.x, if_node.y, "D", color="orange", markersize=10, label="IF" if i == 0 else "", zorder=4)
            self.ax.annotate("IF", (if_node.x, if_node.y), xytext=(5, 5), textcoords="offset points")

        # Plot all customer locations.
        for customer in self.problem.customers:
            self.ax.plot(customer.x, customer.y, "o", color="lightblue", markersize=6, zorder=3)
            self.ax.annotate(f"C{customer.id}", (customer.x, customer.y), xytext=(3, 3), textcoords="offset points")

        self.ax.legend(loc="upper right")
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

    def plot_solution(self, solution: Solution, title: str = "Waste Collection Routes"):
        """Plots a complete solution, drawing all vehicle routes."""
        self._draw_base_map(title)

        # Plot each route with a different color.
        for i, route in enumerate(solution.routes):
            if not any(n.type == "customer" for n in route.nodes):
                continue # Skip empty routes.

            color = self.colors[i % len(self.colors)]
            route_x = [node.x for node in route.nodes]
            route_y = [node.y for node in route.nodes]

            self.ax.plot(route_x, route_y, "-", color=color, linewidth=2, alpha=0.8, label=f"Vehicle {i + 1}")

        plt.tight_layout()
        return self.fig

    def plot_convergence(
        self, convergence_history: List[float], title: str = "ALNS Convergence"
    ):
        """Plots the convergence of the solution cost over iterations."""
        self.conv_ax.clear()
        self.conv_ax.plot(convergence_history, "b-", linewidth=1.5)
        self.conv_ax.set_title(title, fontsize=12, fontweight="bold")
        self.conv_ax.set_xlabel("Iteration")
        self.conv_ax.set_ylabel("Solution Cost")
        self.conv_ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self.fig

    def start_live(self, title: str = "Live Waste Collection Routes"):
        """Initializes the figure for live plotting."""
        self._draw_base_map(title)
        if self._live:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def update_live(
        self,
        solution: Solution,
        convergence_history: Optional[List[float]] = None,
        pause: float = 0.01,
    ):
        """
        Updates the live plot with a new solution and convergence history.
        This is intended to be called from within the ALNS loop.
        """
        try:
            self._draw_base_map("Live - Waste Collection Routes")

            # Redraw all routes for the current solution.
            for i, route in enumerate(solution.routes):
                if not route.nodes: continue
                color = self.colors[i % len(self.colors)]
                route_x = [node.x for node in route.nodes]
                route_y = [node.y for node in route.nodes]
                self.ax.plot(route_x, route_y, "-", color=color, linewidth=1.5, alpha=0.9)

            # Update the convergence plot.
            if convergence_history is not None:
                self.conv_ax.clear()
                self.conv_ax.plot(convergence_history, "b-", linewidth=1.0)
                self.conv_ax.set_title("ALNS Convergence", fontsize=12)
                self.conv_ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if self._live:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except Exception:
            # Fail silently to prevent visualization errors from crashing the solver.
            pass

    def stop_live(self):
        """Stops live plotting and returns the final figure."""
        if self._live:
            plt.ioff()
        return self.fig


class PerformanceAnalyzer:
    """
    Analyzes a given solution to calculate various performance and
    efficiency metrics, providing a quantitative assessment of the
    solution's quality.
    """

    def __init__(self, problem: ProblemInstance):
        self.problem = problem

    def analyze_solution(self, solution: Solution) -> Dict:
        """
        Performs a comprehensive analysis of the solution, calculating key metrics
        such as cost, vehicle usage, and efficiency.
        """
        analysis = {
            "total_cost": solution.total_cost,
            "total_distance": solution.total_distance,
            "total_time": solution.total_time,
            "num_vehicles": 0,
            "num_unassigned": len(solution.unassigned_customers),
            "if_visits": 0,
            "route_details": [],
            "efficiency_metrics": {},
        }

        # Only consider routes that serve at least one customer.
        served_routes = [r for r in solution.routes if any(n.type == "customer" for n in r.nodes)]
        analysis["num_vehicles"] = len(served_routes)

        # Calculate total demand served by the solution.
        served_customer_ids = {n.id for r in served_routes for n in r.nodes if n.type == "customer"}
        total_demand = sum(c.demand for c in self.problem.customers if c.id in served_customer_ids)
        
        # Calculate total capacity based on the number of vehicles used.
        total_capacity = self.problem.vehicle_capacity * len(served_routes)

        # Analyze each individual route.
        for idx, route in enumerate(served_routes):
            route_analysis = {
                "vehicle_id": idx + 1,
                "distance": route.total_distance,
                "time": route.total_time,
                "customers_served": sum(1 for n in route.nodes if n.type == "customer"),
                "if_visits": sum(1 for n in route.nodes if n.type == "if"),
                "max_load": max(route.loads) if route.loads else 0,
                "load_utilization": (max(route.loads) / self.problem.vehicle_capacity) if route.loads else 0,
            }
            analysis["route_details"].append(route_analysis)
            analysis["if_visits"] += route_analysis["if_visits"]

        # Calculate overall efficiency metrics.
        analysis["efficiency_metrics"] = {
            # Demand served per unit of distance traveled.
            "distance_efficiency": total_demand / solution.total_distance if solution.total_distance > 0 else 0,
            # Percentage of total vehicle capacity that is used.
            "capacity_utilization": total_demand / total_capacity if total_capacity > 0 else 0,
        }

        return analysis

    def generate_report(self, solution: Solution) -> str:
        """Generates a human-readable performance report string."""
        analysis = self.analyze_solution(solution)

        report = f"=== WASTE COLLECTION PERFORMANCE REPORT ===\n\n"
        report += f"SOLUTION OVERVIEW:\n"
        report += f"- Total Cost: {analysis['total_cost']:.2f}\n"
        report += f"- Vehicles Used: {analysis['num_vehicles']}\n"
        report += f"- Unassigned Customers: {analysis['num_unassigned']}\n\n"
        
        report += "ROUTE DETAILS:\n"
        for route in analysis["route_details"]:
            report += f"  Vehicle {route['vehicle_id']}:\n"
            report += f"    - Distance: {route['distance']:.2f}\n"
            report += f"    - Customers Served: {route['customers_served']}\n"
            report += f"    - Load Utilization: {route['load_utilization']:.1%}\n"

        eff = analysis.get("efficiency_metrics", {})
        report += f"\nEFFICIENCY METRICS:\n"
        report += f"- Distance Efficiency: {eff.get('distance_efficiency', 0.0):.3f} (demand/distance)\n"
        report += f"- Capacity Utilization: {eff.get('capacity_utilization', 0.0):.1%}\n"

        return report


def save_solution_to_file(solution: Solution, filename: str):
    """Saves a solution object to a JSON file for later analysis."""
    solution_data = {
        "timestamp": datetime.now().isoformat(),
        "total_cost": solution.total_cost,
        "unassigned_customers": list(solution.unassigned_customers),
        "routes": [],
    }

    for i, route in enumerate(solution.routes):
        route_data = {
            "vehicle_id": i + 1,
            "nodes": [{"id": n.id, "type": n.type} for n in route.nodes],
            "total_distance": route.total_distance,
        }
        solution_data["routes"].append(route_data)

    with open(filename, "w") as f:
        json.dump(solution_data, f, indent=2)
    print(f"Solution saved to: {filename}")
