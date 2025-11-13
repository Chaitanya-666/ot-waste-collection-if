#!/usr/bin/env python3
"""
OT Project: Municipal Waste Collection Modelling with Intermediate Facilities
Enhanced Main Entry Point with CLI and Comprehensive Demonstration

This enhanced main program provides:
- Command-line interface for flexible usage
- Multiple demonstration modes
- Comprehensive performance analysis
- Visualization capabilities
- Benchmarking functionality
- Configuration file support
"""

import argparse
import sys
import os
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Ensure the project's `src` directory is importable when the script runs directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Top-level imports from the src package
from src.problem import ProblemInstance, Location
from src.alns import ALNS
from src.data_generator import DataGenerator
from src.utils import RouteVisualizer, PerformanceAnalyzer, save_solution_to_file


def create_sample_instance() -> ProblemInstance:
    """Create the original sample instance for demonstration"""
    print("Creating sample problem instance...")

    problem = ProblemInstance("Sample Instance")
    problem.vehicle_capacity = 20
    problem.number_of_vehicles = 3
    problem.disposal_time = 2

    # Add depot
    depot = Location(0, 0, 0, 0, "depot")
    problem.depot = depot

    # Add intermediate facility(ies)
    if1 = Location(100, 20, 20, 0, "if")
    problem.intermediate_facilities.append(if1)

    # Add customers (id, x, y, demand, type)
    customers = [
        Location(1, 5, 2, 4, "customer"),
        Location(2, 3, 8, 6, "customer"),
        Location(3, 9, 1, 5, "customer"),
        Location(4, 6, 7, 7, "customer"),
        Location(5, 2, 3, 3, "customer"),
        Location(6, 8, 6, 4, "customer"),
    ]
    for c in customers:
        problem.customers.append(c)

    problem.calculate_distance_matrix()
    return problem


def create_comprehensive_instance() -> ProblemInstance:
    """Create a more comprehensive instance for demonstration"""
    print("Creating comprehensive problem instance...")

    return DataGenerator.generate_instance(
        name="Comprehensive Demo",
        n_customers=15,
        n_ifs=2,
        vehicle_capacity=25,
        area_size=120,
        demand_range=(1, 12),
        service_time_range=(1, 6),
        cluster_factor=0.4,
        seed=42,
    )


def run_basic_demonstration():
    """Run the basic demonstration with original sample"""
    print("\n" + "=" * 60)
    print("BASIC DEMONSTRATION")
    print("=" * 60)

    # Create problem
    problem = create_sample_instance()

    print(f"\nProblem: {problem}")
    print(
        f"Customers: {len(problem.customers)}, IFs: {len(problem.intermediate_facilities)}"
    )
    print(f"Vehicle Capacity: {problem.vehicle_capacity}")

    # Initialize solver
    solver = ALNS(problem)
    solver.max_iterations = 200

    # Run optimization
    print(f"\nStarting ALNS optimization with {solver.max_iterations} iterations...")
    start_time = time.time()

    solution = solver.run(max_iterations=solver.max_iterations)
    end_time = time.time()

    print(f"ALNS completed in {end_time - start_time:.2f} seconds")

    # Display results
    print("\n" + "-" * 40)
    print("OPTIMIZATION RESULTS")
    print("-" * 40)
    print(f"Final best solution: {solution.total_cost:.2f}")
    print(f"Total iterations: {solver.iteration}")
    print(f"Routes: {len(solution.routes)}")
    print(f"Unassigned customers: {len(solution.unassigned_customers)}")

    # Display route details
    print("\nRoute Details:")
    for idx, route in enumerate(solution.routes):
        print(
            f"Route {idx + 1}: Distance={route.total_distance:.2f}, Time={route.total_time:.2f}"
        )
        print(
            f"  Sequence: {' -> '.join([f'{node.type[0].upper()}{node.id}' for node in route.nodes])}"
        )

    return solution, problem, solver


def run_comprehensive_demonstration(
    live: bool = False, iterations: int = 500
) -> Tuple[
    Optional[object], ProblemInstance, Optional[object], Optional[Dict[str, Any]]
]:
    """Run a comprehensive demonstration with analysis

    Args:
        live: when True, enable live plotting via RouteVisualizer and wire the solver
              iteration callback so the plot updates during optimization.
        iterations: number of ALNS iterations to run.

    Returns:
        A 4-tuple: (solution or None, problem, solver or None, analysis dict or None).
        If the instance is infeasible the function returns (None, problem, None, None).
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)

    # Create problem
    problem = create_comprehensive_instance()

    # Ensure the solver has enough vehicles to feasibly serve total demand.
    try:
        min_needed = int(problem.get_min_vehicles_needed())
        # Fix infinity comparison issue - default number_of_vehicles is infinity
        current_limit = problem.number_of_vehicles
        needs_adjustment = (
            current_limit == float('inf') or 
            current_limit < min_needed or
            current_limit <= 0
        )
        
        if needs_adjustment:
            problem.number_of_vehicles = min_needed
            print(
                f"Note: adjusted problem.number_of_vehicles to minimum required: {problem.number_of_vehicles}"
            )
    except Exception:
        # If anything goes wrong here, continue — a subsequent feasibility check will catch issues.
        pass

    # Sanity / feasibility check: if instance is still infeasible, warn and exit early.
    feasible_flag, feasible_msg = problem.is_feasible()
    print(f"\nProblem: {problem}")
    print(
        f"Customers: {len(problem.customers)}, IFs: {len(problem.intermediate_facilities)}"
    )
    print(f"Vehicle Capacity: {problem.vehicle_capacity}")
    if not feasible_flag:
        print(f"Problem feasibility check: {feasible_flag} ({feasible_msg})")
        print(
            "Instance is infeasible with the current settings. Adjust vehicle capacity or number_of_vehicles and retry."
        )
        # Return a consistent tuple so callers do not crash on unpacking
        return None, problem, None, None

    # Initialize solver
    solver = ALNS(problem)
    solver.max_iterations = int(iterations)

    # Progress tracking for optimization
    ProgressTracker = create_progress_tracker()
    progress = ProgressTracker(solver.max_iterations)
    
    # Optionally enable live plotting: set up visualizer and register iteration callback
    visualizer = None
    if live:
        try:
            visualizer = RouteVisualizer(problem, live=True)
            visualizer.start_live(title=f"Live - {problem.name}")

            def _iteration_callback(iteration_idx, best_solution):
                progress.update(iteration_idx)
                try:
                    visualizer.update_live(
                        best_solution, getattr(solver, "convergence_history", [])
                    )
                except Exception:
                    # swallow visualization errors to keep solver running
                    pass

            # Assign callback (acceptable at runtime; some static checkers may warn)
            solver.iteration_callback = _iteration_callback
            print("🎬 Live plotting enabled for comprehensive demonstration.")
        except Exception:
            visualizer = None
    else:
        # Add progress callback without visualization
        def _progress_callback(iteration_idx, best_solution):
            progress.update(iteration_idx)
        
        solver.iteration_callback = _progress_callback

    # Run optimization
    print(f"\n🚀 Starting ALNS optimization with {solver.max_iterations} iterations...")
    if not live:
        print("📊 Progress tracking enabled - watch the optimization progress!")
    start_time = time.time()

    solution = solver.run(max_iterations=solver.max_iterations)
    end_time = time.time()

    print(f"ALNS completed in {end_time - start_time:.2f} seconds")

    # If live plotting was enabled, do a final update and stop interactive mode.
    if visualizer is not None:
        try:
            visualizer.update_live(solution, getattr(solver, "convergence_history", []))
            visualizer.stop_live()
        except Exception:
            pass

    # Performance analysis
    analyzer = PerformanceAnalyzer(problem)
    analysis = analyzer.analyze_solution(solution)

    # Display comprehensive results
    print("\n" + "-" * 60)
    print("COMPREHENSIVE ANALYSIS")
    print("-" * 60)

    print(f"\nSOLUTION OVERVIEW:")
    print(f"  Total Cost: {analysis['total_cost']:.2f}")
    print(f"  Total Distance: {analysis['total_distance']:.2f}")
    print(f"  Total Time: {analysis['total_time']:.2f}")
    print(f"  Vehicles Used: {analysis['num_vehicles']}")
    print(f"  IF Visits: {analysis['if_visits']}")

    print(f"\nEFFICIENCY METRICS:")
    metrics = analysis["efficiency_metrics"]
    print(f"  Distance Efficiency: {metrics['distance_efficiency']:.3f}")
    print(f"  Capacity Utilization: {metrics['capacity_utilization']:.1%}")
    print(f"  Vehicle Efficiency: {metrics['vehicle_efficiency']:.1%}")
    print(f"  IF Efficiency: {metrics['if_efficiency']:.1f} visits/vehicle")

    print(f"\nROUTE DETAILS:")
    for route in analysis["route_details"]:
        print(f"  Vehicle {route['vehicle_id']}:")
        print(f"    Distance: {route['distance']:.2f}")
        print(f"    Time: {route['time']:.2f}")
        print(f"    Customers: {route['customers_served']}")
        print(f"    IF Visits: {route['if_visits']}")
        print(f"    Max Load: {route['max_load']:.1f}/{problem.vehicle_capacity}")
        print(f"    Load Utilization: {route['load_utilization']:.1%}")

    # Generate performance report
    print(f"\n" + "-" * 60)
    print("PERFORMANCE REPORT")
    print("-" * 60)
    report = analyzer.generate_report(solution)
    print(report)

    return solution, problem, solver, analysis


def run_benchmark_demonstration():
    """Run benchmarking with multiple instances"""
    print("\n" + "=" * 60)
    print("BENCHMARK DEMONSTRATION")
    print("=" * 60)

    # Create test instances
    instances = [
        ("Small", 6, 1),
        ("Medium", 15, 2),
        ("Large", 25, 3),
    ]

    results = []

    for name, n_customers, n_ifs in instances:
        print(f"\n--- {name} Instance ({n_customers} customers, {n_ifs} IFs) ---")

        # Create instance
        problem = DataGenerator.generate_instance(
            f"{name} Benchmark",
            n_customers,
            n_ifs,
            vehicle_capacity=20 + n_customers // 5,
            seed=42,
        )

        # Solve
        solver = ALNS(problem)
        solver.max_iterations = 300

        start_time = time.time()
        solution = solver.run(max_iterations=solver.max_iterations)
        end_time = time.time()

        # Analyze
        analyzer = PerformanceAnalyzer(problem)
        analysis = analyzer.analyze_solution(solution)

        result = {
            "name": name,
            "customers": n_customers,
            "ifs": n_ifs,
            "cost": solution.total_cost,
            "distance": solution.total_distance,
            "vehicles": len(solution.routes),
            "time": end_time - start_time,
            "convergence": len(getattr(solver, "convergence_history", [])),
        }

        results.append(result)

        print(f"  Solution Cost: {solution.total_cost:.2f}")
        print(f"  Vehicles Used: {len(solution.routes)}")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Unassigned: {len(solution.unassigned_customers)}")

    # Display benchmark summary
    print(f"\n" + "-" * 60)
    print("BENCHMARK SUMMARY")
    print("-" * 60)

    print(
        f"{'Instance':<10} {'Customers':<10} {'IFs':<5} {'Cost':<10} {'Vehicles':<10} {'Time(s)':<10} {'Convergence':<12}"
    )
    print("-" * 70)

    for result in results:
        print(
            f"{result['name']:<10} {result['customers']:<10} {result['ifs']:<5} "
            f"{result['cost']:<10.2f} {result['vehicles']:<10} {result['time']:<10.2f} "
            f"{result['convergence']:<12}"
        )

    return results


def display_ascii_route_map(solution, problem):
    """Display ASCII art route visualization in terminal"""
    print("\n" + "=" * 80)
    print("🗺️  ASCII ROUTE VISUALIZATION")
    print("=" * 80)
    
    # Normalize coordinates for ASCII display
    all_x = [problem.depot.x]
    all_y = [problem.depot.y]
    
    for customer in problem.customers:
        all_x.append(customer.x)
        all_y.append(customer.y)
    
    for ifac in problem.intermediate_facilities:
        all_x.append(ifac.x)
        all_y.append(ifac.y)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width, height = 60, 20
    
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Function to convert coordinates to grid position
    def coord_to_grid(x, y):
        grid_x = int((x - min_x) / (max_x - min_x) * (width - 1))
        grid_y = int((y - min_y) / (max_y - min_y) * (height - 1))
        return grid_x, height - 1 - grid_y  # Flip Y axis
    
    # Place depot (red)
    depot_x, depot_y = coord_to_grid(problem.depot.x, problem.depot.y)
    grid[depot_y][depot_x] = '🏢'
    
    # Place intermediate facilities (green)
    for ifac in problem.intermediate_facilities:
        if_x, if_y = coord_to_grid(ifac.x, ifac.y)
        grid[if_y][if_x] = '🏭'
    
    # Place customers (blue)
    for customer in problem.customers:
        c_x, c_y = coord_to_grid(customer.x, customer.y)
        grid[c_y][c_x] = '📍'
    
    # Draw routes with different symbols
    route_symbols = ['➤', '→', '↗', '↘', '◆', '◇']
    
    for i, route in enumerate(solution.routes):
        symbol = route_symbols[i % len(route_symbols)]
        
        for j in range(len(route.nodes) - 1):
            current = route.nodes[j]
            next_node = route.nodes[j + 1]
            
            curr_x, curr_y = coord_to_grid(current.x, current.y)
            next_x, next_y = coord_to_grid(next_node.x, next_node.y)
            
            # Draw simple line between points
            if abs(curr_x - next_x) > abs(curr_y - next_y):
                # Horizontal line
                start, end = (curr_x, next_x) if curr_x < next_x else (next_x, curr_x)
                for x in range(start + 1, end):
                    if 0 <= x < width and 0 <= curr_y < height:
                        grid[curr_y][x] = symbol
            else:
                # Vertical line
                start, end = (curr_y, next_y) if curr_y < next_y else (next_y, curr_y)
                for y in range(start + 1, end):
                    if 0 <= depot_x < width and 0 <= y < height:
                        grid[y][depot_x] = symbol
    
    # Display grid with legend
    print(f"Map Legend: 🏢=Depot, 🏭=IF, 📍=Customers, Route symbols show vehicle paths")
    print(f"Map Area: {max_x - min_x:.0f} x {max_y - min_y:.0f} units")
    print("\n" + "─" * (width + 4))
    for row in grid:
        print("│" + "".join(row) + "│")
    print("─" * (width + 4))


def display_detailed_route_analysis(solution, problem):
    """Display detailed route analysis with ASCII art"""
    print("\n" + "=" * 80)
    print("📊 DETAILED ROUTE ANALYSIS")
    print("=" * 80)
    
    for idx, route in enumerate(solution.routes):
        print(f"\n🚛 Vehicle {idx + 1} Route Analysis:")
        print("─" * 50)
        
        # Route sequence
        print("📋 Route Sequence:")
        sequence = " → ".join([f"{node.type.upper()}" for node in route.nodes])
        print(f"   {sequence}")
        
        # Load profile
        print(f"\n📦 Load Profile:")
        current_load = 0
        for i, node in enumerate(route.nodes):
            if node.type == "customer":
                current_load += node.demand
                print(f"   {node.type.upper()}{node.id}: +{node.demand} (load: {current_load})")
            elif node.type == "if":
                current_load = 0  # Vehicle dumps waste
                print(f"   {node.type.upper()}{node.id}: DUMP (load: {current_load})")
        
        # Efficiency metrics
        print(f"\n📈 Efficiency Metrics:")
        route_demand = sum(node.demand for node in route.nodes if node.type == 'customer')
        utilization = route_demand / problem.vehicle_capacity
        print(f"   Capacity Utilization: {utilization:.1%}")
        print(f"   Distance: {route.total_distance:.2f} units")
        print(f"   Time: {route.total_time:.2f} time units")
        print(f"   Total Demand Served: {route_demand:.1f} units")
        
        # ASCII progress bar for utilization
        bar_width = 20
        filled = int(bar_width * utilization)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"   Utilization Bar: |{bar}|")


def create_progress_tracker():
    """Create a simple progress tracker for ALNS iterations"""
    class ProgressTracker:
        def __init__(self, total_iterations):
            self.total = total_iterations
            self.current = 0
            self.last_percentage = -1
        
        def update(self, iteration):
            self.current = iteration
            percentage = int((iteration / self.total) * 100)
            
            # Only update if percentage changed (reduces spam)
            if percentage != self.last_percentage:
                self.last_percentage = percentage
                bar_width = 30
                filled = int(bar_width * iteration / self.total)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\r🔄 ALNS Progress: |{bar}| {percentage}% ({iteration}/{self.total})", end="")
                
                if percentage == 100:
                    print()  # New line when complete
    
    return ProgressTracker


def run_visualization_demo(
    solution, problem, save_plots=False, convergence_history=None
):
    """Enhanced visualization demonstration with ASCII art and detailed analysis"""
    print("\n" + "=" * 80)
    print("🎨 COMPREHENSIVE VISUALIZATION DEMONSTRATION")
    print("=" * 80)

    try:
        # Display ASCII route map
        display_ascii_route_map(solution, problem)
        
        # Display detailed route analysis
        display_detailed_route_analysis(solution, problem)
        
        # Create visualizer for matplotlib plots
        print(f"\n" + "=" * 80)
        print("📊 MATPLOTLIB VISUALIZATION")
        print("=" * 80)
        
        visualizer = RouteVisualizer(problem)

        # Plot solution
        print("Generating route visualization...")
        fig = visualizer.plot_solution(solution, "Waste Collection Routes - ALNS Optimized")

        if save_plots:
            filename = f"routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✅ Route plot saved as: {filename}")

        # Plot convergence
        print("Generating convergence analysis...")
        conv_hist = convergence_history
        if conv_hist is None:
            solver_obj = getattr(solution, "solver", None)
            conv_hist = (
                getattr(solver_obj, "convergence_history", []) if solver_obj else []
            )

        conv_fig = visualizer.plot_convergence(conv_hist)

        if save_plots:
            conv_filename = f"convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            conv_fig.savefig(conv_filename, dpi=300, bbox_inches="tight")
            print(f"✅ Convergence plot saved as: {conv_filename}")

        # Performance summary
        print(f"\n" + "=" * 80)
        print("🎯 VISUALIZATION SUMMARY")
        print("=" * 80)
        print("✅ ASCII route map generated (terminal)")
        print("✅ Detailed route analysis completed")
        print("✅ Route visualization plotted")
        if conv_hist:
            print("✅ Convergence analysis generated")
        print(f"✅ Total routes visualized: {len(solution.routes)}")
        print("✅ All visualization complete!")

    except ImportError as e:
        print(f"⚠️ Matplotlib visualization not available: {e}")
        print("📦 Install matplotlib: pip install matplotlib")
        
        # Fallback to ASCII-only visualization
        print("\n" + "=" * 60)
        print("🖥️  ASCII-ONLY VISUALIZATION MODE")
        print("=" * 60)
        display_ascii_route_map(solution, problem)
        display_detailed_route_analysis(solution, problem)
        
    except Exception as e:
        # Do not let visualization failures crash the run.
        print(f"⚠️ Visualization failed: {e}")
        print("🔄 Attempting ASCII fallback...")
        try:
            display_ascii_route_map(solution, problem)
            display_detailed_route_analysis(solution, problem)
            print("✅ ASCII visualization completed successfully!")
        except Exception as e2:
            print(f"❌ Complete visualization failure: {e2}")
            print("📋 Basic route summary:")
            for idx, route in enumerate(solution.routes):
                print(f"   Route {idx+1}: {len([n for n in route.nodes if n.type == 'customer'])} customers, {route.total_distance:.2f} distance")


def save_results(solution, problem, analysis=None, filename=None):
    """Save results to file"""
    if filename is None:
        filename = f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_solution_to_file(solution, filename)
    print(f"Results saved to: {filename}")


def display_ascii_route_map(solution, problem):
    """Display ASCII art route visualization in terminal"""
    print("\n" + "=" * 80)
    print("🗺️  ASCII ROUTE VISUALIZATION")
    print("=" * 80)
    
    # Normalize coordinates for ASCII display
    all_x = [problem.depot.x]
    all_y = [problem.depot.y]
    
    for customer in problem.customers:
        all_x.append(customer.x)
        all_y.append(customer.y)
    
    for ifac in problem.intermediate_facilities:
        all_x.append(ifac.x)
        all_y.append(ifac.y)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width, height = 60, 20
    
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Function to convert coordinates to grid position
    def coord_to_grid(x, y):
        grid_x = int((x - min_x) / (max_x - min_x) * (width - 1))
        grid_y = int((y - min_y) / (max_y - min_y) * (height - 1))
        return grid_x, height - 1 - grid_y  # Flip Y axis
    
    # Place depot (red)
    depot_x, depot_y = coord_to_grid(problem.depot.x, problem.depot.y)
    grid[depot_y][depot_x] = '🏢'
    
    # Place intermediate facilities (green)
    for ifac in problem.intermediate_facilities:
        if_x, if_y = coord_to_grid(ifac.x, ifac.y)
        grid[if_y][if_x] = '🏭'
    
    # Place customers (blue)
    for customer in problem.customers:
        c_x, c_y = coord_to_grid(customer.x, customer.y)
        grid[c_y][c_x] = '📍'
    
    # Draw routes with different symbols
    route_symbols = ['➤', '→', '↗', '↘', '◆', '◇']
    
    for i, route in enumerate(solution.routes):
        symbol = route_symbols[i % len(route_symbols)]
        
        for j in range(len(route.nodes) - 1):
            current = route.nodes[j]
            next_node = route.nodes[j + 1]
            
            curr_x, curr_y = coord_to_grid(current.x, current.y)
            next_x, next_y = coord_to_grid(next_node.x, next_node.y)
            
            # Draw simple line between points
            if abs(curr_x - next_x) > abs(curr_y - next_y):
                # Horizontal line
                start, end = (curr_x, next_x) if curr_x < next_x else (next_x, curr_x)
                for x in range(start + 1, end):
                    if 0 <= x < width and 0 <= curr_y < height:
                        grid[curr_y][x] = symbol
            else:
                # Vertical line
                start, end = (curr_y, next_y) if curr_y < next_y else (next_y, curr_y)
                for y in range(start + 1, end):
                    if 0 <= depot_x < width and 0 <= y < height:
                        grid[y][depot_x] = symbol
    
    # Display grid with legend
    print(f"Map Legend: 🏢=Depot, 🏭=IF, 📍=Customers, Route symbols show vehicle paths")
    print(f"Map Area: {max_x - min_x:.0f} x {max_y - min_y:.0f} units")
    print("\n" + "─" * (width + 4))
    for row in grid:
        print("│" + "".join(row) + "│")
    print("─" * (width + 4))


def display_detailed_route_analysis(solution, problem):
    """Display detailed route analysis with ASCII art"""
    print("\n" + "=" * 80)
    print("📊 DETAILED ROUTE ANALYSIS")
    print("=" * 80)
    
    for idx, route in enumerate(solution.routes):
        print(f"\n🚛 Vehicle {idx + 1} Route Analysis:")
        print("─" * 50)
        
        # Route sequence
        print("📋 Route Sequence:")
        sequence = " → ".join([f"{node.type.upper()}" for node in route.nodes])
        print(f"   {sequence}")
        
        # Load profile
        print(f"\n📦 Load Profile:")
        current_load = 0
        for i, node in enumerate(route.nodes):
            if node.type == "customer":
                current_load += node.demand
                print(f"   {node.type.upper()}{node.id}: +{node.demand} (load: {current_load})")
            elif node.type == "if":
                current_load = 0  # Vehicle dumps waste
                print(f"   {node.type.upper()}{node.id}: DUMP (load: {current_load})")
        
        # Efficiency metrics
        print(f"\n📈 Efficiency Metrics:")
        route_demand = sum(node.demand for node in route.nodes if node.type == 'customer')
        utilization = route_demand / problem.vehicle_capacity
        print(f"   Capacity Utilization: {utilization:.1%}")
        print(f"   Distance: {route.total_distance:.2f} units")
        print(f"   Time: {route.total_time:.2f} time units")
        print(f"   Total Demand Served: {route_demand:.1f} units")
        
        # ASCII progress bar for utilization
        bar_width = 20
        filled = int(bar_width * utilization)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"   Utilization Bar: |{bar}|")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Municipal Waste Collection Route Optimization with ALNS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo basic           # Run basic demonstration
  python main.py --demo comprehensive   # Run comprehensive demonstration
  python main.py --demo benchmark       # Run benchmark demonstration
  python main.py --instance small.json  # Solve from instance file
  python main.py --config config.json   # Use configuration file
        """,
    )

    parser.add_argument(
        "--demo",
        choices=["basic", "comprehensive", "benchmark"],
        help="Run demonstration mode",
    )

    parser.add_argument("--instance", help="Path to instance file (JSON format)")

    parser.add_argument("--config", help="Path to configuration file")

    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of ALNS iterations (default: 200)",
    )

    parser.add_argument(
        "--save-plots", action="store_true", help="Save visualization plots to files"
    )

    parser.add_argument(
        "--live", action="store_true", help="Enable live plotting during optimization"
    )

    parser.add_argument(
        "--save-results", action="store_true", help="Save results to JSON file"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create configuration template file",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print("=== Waste Collection Route Optimization ===")
    print("Project: VRP with Intermediate Facilities using ALNS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Handle configuration creation
    if args.create_config:
        DataGenerator.create_config_template()
        return

    # Initialize placeholders
    solution = None
    problem = None
    solver = None
    analysis = None

    # Handle different modes
    if args.demo:
        if args.demo == "basic":
            solution, problem, solver = run_basic_demonstration()
        elif args.demo == "comprehensive":
            result = run_comprehensive_demonstration(
                live=getattr(args, "live", False), iterations=args.iterations
            )
            solution, problem, solver, analysis = result
            if solution is None:
                print(
                    "Comprehensive demonstration did not produce a solution. Exiting."
                )
                return
        elif args.demo == "benchmark":
            _results = run_benchmark_demonstration()
            return

    elif args.instance:
        # Load instance from file
        print(f"Loading instance from: {args.instance}")
        problem = DataGenerator.load_instance_from_file(args.instance)
        solver = ALNS(problem)
        solver.max_iterations = args.iterations

        start_time = time.time()
        solution = solver.run(max_iterations=solver.max_iterations)
        end_time = time.time()

        print(f"Solution found in {end_time - start_time:.2f} seconds")
        print(f"Cost: {solution.total_cost:.2f}")
        print(f"Routes: {len(solution.routes)}")

    else:
        # Default: run comprehensive demonstration (respect --live if provided)
        result = run_comprehensive_demonstration(
            live=getattr(args, "live", False), iterations=args.iterations
        )
        solution, problem, solver, analysis = result
        if solution is None:
            print("Comprehensive demonstration did not produce a solution. Exiting.")
            return

    # Enhanced Visualization (only if we have a solution)
    if solution is not None and (args.verbose or args.save_plots or args.demo):
        try:
            # Get convergence history for visualization
            conv_hist = None
            if solver is not None:
                conv_hist = getattr(solver, "convergence_history", [])
            
            run_visualization_demo(solution, problem, save_plots=args.save_plots, convergence_history=conv_hist)
        except Exception as e:
            print(f"Visualization aborted: {e}")
    elif solution is not None and args.demo == "comprehensive":
        # Always show ASCII visualization for comprehensive demo
        try:
            display_ascii_route_map(solution, problem)
            display_detailed_route_analysis(solution, problem)
        except Exception as e:
            print(f"ASCII visualization aborted: {e}")

    # Save results (only if we have a solution)
    if solution is not None and args.save_results:
        try:
            save_results(solution, problem, analysis)
        except Exception as e:
            print(f"Failed to save results: {e}")

    # Final summary
    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    if solution is not None:
        try:
            print(f"Final Solution Cost: {solution.total_cost:.2f}")
            print(f"Total Routes: {len(solution.routes)}")
            print(f"Unassigned Customers: {len(solution.unassigned_customers)}")
            print(
                f"All customers assigned: {'Yes' if len(solution.unassigned_customers) == 0 else 'No'}"
            )
        except Exception:
            print("Solution summary: (failed to display some fields)")
    else:
        print("No solution available to summarize.")

    if args.verbose and solver is not None:
        try:
            print(f"Solver completed {solver.iteration} iterations")
            print(
                f"Best solution found at iteration: {len(getattr(solver, 'convergence_history', []))}"
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
