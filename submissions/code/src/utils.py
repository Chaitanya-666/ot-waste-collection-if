"""
Utilities for VRP-IF Solver
===========================

Author: Harsh Sharma (231070064) - Utility functions and helper classes

This module provides a collection of utility functions and helper classes
that support the main VRP-IF solver. These utilities include:
- Logging and debugging tools
- Performance monitoring and profiling
- Data serialization and deserialization
- Visualization and plotting functions
- Mathematical and statistical helpers

Key Features:
- Configurable logging with different verbosity levels
- Performance timers for profiling code sections
- JSON serialization for problem instances and solutions
- Matplotlib-based plotting for visualizing routes and solutions
- Statistical functions for analyzing results
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

# Try to import matplotlib for plotting, but handle the case where it's not installed
try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.animation as animation  # type: ignore
    from matplotlib.patches import Circle, FancyArrowPatch  # type: ignore
except ImportError:
    plt = None
    animation = None
    Circle = None
    FancyArrowPatch = None

from .problem import ProblemInstance, Location
from .solution import Solution, Route


# ==============================================================================
# Logging Utilities
# ==============================================================================
def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the application.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        
    Note:
        - Sets up a basic console logger with a timestamp
        - Can be extended to log to files or other destinations
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: The name of the logger (usually __name__)
        
    Returns:
        logging.Logger: A logger instance
    """
    return logging.getLogger(name)


# ==============================================================================
# Performance Monitoring
# ==============================================================================
class PerformanceTimer:
    """
    A simple context manager for timing code blocks.
    
    Example:
        >>> with PerformanceTimer("My code block") as timer:
        ...     # Code to be timed
        ...
        My code block...
        My code block finished in 0.123 seconds
        
    Attributes:
        name (str): Name of the timed block
        start_time (float): Time when the block was entered
        end_time (float): Time when the block was exited
        elapsed_time (float): Total time spent in the block
    """
    
    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    def __enter__(self):
        """Start the timer when entering the context."""
        print(f"{self.name}...")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and print the elapsed time when exiting the context."""
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
            print(f"{self.name} finished in {self.elapsed_time:.3f} seconds")


# ==============================================================================
# Data Serialization
# ==============================================================================
def serialize_solution(solution: Solution, filename: str) -> None:
    """
    Serialize a solution object to a JSON file.
    
    Args:
        solution: The solution to serialize
        filename: The name of the output file
        
    Note:
        - Saves the solution's routes, metrics, and unassigned customers
        - Uses a human-readable JSON format
    """
    data = {
        "total_cost": solution.total_cost,
        "total_distance": solution.total_distance,
        "total_time": solution.total_time,
        "unassigned_customers": list(solution.unassigned_customers),
        "routes": [
            {
                "nodes": [node.id for node in route.nodes],
                "total_distance": route.total_distance,
                "total_time": route.total_time,
            }
            for route in solution.routes
        ],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def deserialize_solution(filename: str, problem: ProblemInstance) -> Solution:
    """
    Deserialize a solution object from a JSON file.
    
    Args:
        filename: The name of the input file
        problem: The problem instance to associate with the solution
        
    Returns:
        Solution: The deserialized solution object
        
    Note:
        - Reconstructs the solution's routes and metrics
        - Requires a problem instance to map node IDs back to locations
    """
    with open(filename, "r") as f:
        data = json.load(f)

    solution = Solution(problem)
    solution.total_cost = data["total_cost"]
    solution.total_distance = data["total_distance"]
    solution.total_time = data["total_time"]
    solution.unassigned_customers = set(data["unassigned_customers"])

    # Create a mapping from node ID to Location object for quick lookup
    node_map = {node.id: node for node in problem.customers + [problem.depot] + problem.intermediate_facilities}

    for route_data in data["routes"]:
        route = Route()
        route.nodes = [node_map[node_id] for node_id in route_data["nodes"]]
        route.total_distance = route_data["total_distance"]
        route.total_time = route_data["total_time"]
        solution.routes.append(route)

    return solution


# ==============================================================================
# Visualization and Plotting
# ==============================================================================
def plot_solution(
    solution: Solution,
    title: str = "VRP-IF Solution",
    show_labels: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a VRP-IF solution using Matplotlib.
    
    Args:
        solution: The solution to plot
        title: The title of the plot
        show_labels: Whether to show node labels (IDs)
        save_path: If provided, save the plot to this file path
        
    Note:
        - Requires Matplotlib to be installed
        - Plots the depot, customers, and intermediate facilities
        - Draws arrows to show the direction of travel in each route
    """
    if plt is None:
        print("Matplotlib is not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_aspect("equal", adjustable="box")

    # Plot depot
    depot = solution.problem.depot
    ax.plot(depot.x, depot.y, "ks", markersize=10, label="Depot")
    if show_labels:
        ax.text(depot.x, depot.y, f"D{depot.id}", fontsize=9, ha="right")

    # Plot customers
    customers = solution.problem.customers
    ax.plot(
        [c.x for c in customers], [c.y for c in customers], "bo", markersize=5, label="Customers"
    )
    if show_labels:
        for c in customers:
            ax.text(c.x, c.y, f"C{c.id}", fontsize=9, ha="right")

    # Plot intermediate facilities
    ifs = solution.problem.intermediate_facilities
    ax.plot([i.x for i in ifs], [i.y for i in ifs], "g^", markersize=8, label="IFs")
    if show_labels:
        for i in ifs:
            ax.text(i.x, i.y, f"IF{i.id}", fontsize=9, ha="right")

    # Plot routes
    colors = plt.cm.get_cmap("tab10", len(solution.routes))
    for i, route in enumerate(solution.routes):
        color = colors(i)
        for j in range(len(route.nodes) - 1):
            start_node = route.nodes[j]
            end_node = route.nodes[j + 1]
            ax.add_patch(
                FancyArrowPatch(
                    (start_node.x, start_node.y),
                    (end_node.x, end_node.y),
                    arrowstyle="->",
                    mutation_scale=15,
                    color=color,
                    lw=1.5,
                )
            )

    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_animation(
    solution_history: List[Solution],
    problem: ProblemInstance,
    save_path: str = "solution_animation.gif",
) -> None:
    """
    Create an animation of the solution process over time.
    
    Args:
        solution_history: A list of solutions at different stages of the algorithm
        problem: The problem instance
        save_path: The path to save the animation file (e.g., .gif, .mp4)
        
    Note:
        - Requires Matplotlib and a writer like Pillow or FFmpeg
        - Animates the routes as they change over the course of the algorithm
    """
    if plt is None or animation is None:
        print("Matplotlib or animation support is not installed. Skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")

    # Plot static elements (depot, customers, IFs)
    depot = problem.depot
    ax.plot(depot.x, depot.y, "ks", markersize=10, label="Depot")
    customers = problem.customers
    ax.plot([c.x for c in customers], [c.y for c in customers], "bo", markersize=5, label="Customers")
    ifs = problem.intermediate_facilities
    ax.plot([i.x for i in ifs], [i.y for i in ifs], "g^", markersize=8, label="IFs")
    ax.legend()
    ax.grid(True)

    # Store route artists to update them in each frame
    route_artists: List[List[FancyArrowPatch]] = []

    def update(frame: int):
        """Update function for the animation."""
        # Clear previous routes
        for route_set in route_artists:
            for artist in route_set:
                artist.remove()
        route_artists.clear()

        solution = solution_history[frame]
        ax.set_title(f"Iteration {frame} - Cost: {solution.total_cost:.2f}")

        # Plot new routes
        colors = plt.cm.get_cmap("tab10", len(solution.routes))
        for i, route in enumerate(solution.routes):
            color = colors(i)
            route_set = []
            for j in range(len(route.nodes) - 1):
                start_node = route.nodes[j]
                end_node = route.nodes[j + 1]
                arrow = FancyArrowPatch(
                    (start_node.x, start_node.y),
                    (end_node.x, end_node.y),
                    arrowstyle="->",
                    mutation_scale=15,
                    color=color,
                    lw=1.5,
                )
                ax.add_patch(arrow)
                route_set.append(arrow)
            route_artists.append(route_set)

        return [artist for route_set in route_artists for artist in route_set]

    ani = animation.FuncAnimation(
        fig, update, frames=len(solution_history), blit=True, interval=200
    )
    ani.save(save_path, writer="pillow")
    plt.close(fig)


# ==============================================================================
# Mathematical and Statistical Helpers
# ==============================================================================
def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        data: A list of floating-point numbers
        
    Returns:
        A dictionary with mean, median, standard deviation, min, and max
    """
    if not data:
        return {"mean": 0, "median": 0, "std_dev": 0, "min": 0, "max": 0}

    n = len(data)
    mean = sum(data) / n
    sorted_data = sorted(data)
    median = (
        (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        if n % 2 == 0
        else sorted_data[n // 2]
    )
    std_dev = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    min_val = min(data)
    max_val = max(data)

    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val,
    }