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
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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


# some test
def run_comprehensive_demonstration(live: bool = False, iterations: int = 500):
    """Run a comprehensive demonstration with analysis

    Args:
        live: when True, enable live plotting via RouteVisualizer and wire the solver
              iteration callback so the plot updates during optimization.
        iterations: number of ALNS iterations to run.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)

    # Create problem
    problem = create_comprehensive_instance()

    # Ensure the solver has enough vehicles to feasibly serve total demand.
    # If the generated instance would require more vehicles than currently configured,
    # bump `number_of_vehicles` to the minimum required so the solver can attempt a full assignment.
    try:
        min_needed = int(problem.get_min_vehicles_needed())
        if (
            getattr(problem, "number_of_vehicles", None) is None
            or problem.number_of_vehicles < min_needed
        ):
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
        return

    # Initialize solver
    solver = ALNS(problem)
    solver.max_iterations = int(iterations)

    # Optionally enable live plotting: set up visualizer and register iteration callback
    visualizer = None
    if live:
        try:
            visualizer = RouteVisualizer(problem, live=True)
            visualizer.start_live(title=f"Live - {problem.name}")

            def _iteration_callback(iteration_idx, best_solution):
                try:
                    visualizer.update_live(
                        best_solution, getattr(solver, "convergence_history", [])
                    )
                except Exception:
                    # swallow visualization errors to keep solver running
                    pass

            solver.iteration_callback = _iteration_callback
            print("Live plotting enabled for comprehensive demonstration.")
        except Exception:
            visualizer = None

    # Run optimization
    print(f"\nStarting ALNS optimization with {solver.max_iterations} iterations...")
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
            "convergence": len(solver.convergence_history),
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


def run_visualization_demo(
    solution, problem, save_plots=False, convergence_history=None
):
    """Run visualization demonstration

    This function now accepts an optional `convergence_history` list. If it is not
    provided it will attempt to read `solution.solver.convergence_history` as a
    fallback (defensive). This decouples the visualizer from requiring `solution`
    to carry a `solver` attribute.
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)

    try:
        # Create visualizer
        visualizer = RouteVisualizer(problem)

        # Plot solution
        print("Generating route visualization...")
        fig = visualizer.plot_solution(solution, "Waste Collection Routes")

        if save_plots:
            filename = f"routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Plot saved as: {filename}")

        # Plot convergence
        print("Generating convergence plot...")
        # Prefer explicit convergence_history argument; otherwise attempt to read from solution.solver
        conv_hist = convergence_history
        if conv_hist is None:
            solver_obj = getattr(solution, "solver", None)
            conv_hist = (
                getattr(solver_obj, "convergence_history", []) if solver_obj else []
            )

        conv_fig = visualizer.plot_convergence(conv_hist)

        if save_plots:
            conv_filename = (
                f"convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            conv_fig.savefig(conv_filename, dpi=300, bbox_inches="tight")
            print(f"Convergence plot saved as: {conv_filename}")

        print("Visualization complete! (Plots would be displayed in interactive mode)")

    except ImportError as e:
        print(f"Visualization not available: {e}")
        print("Install matplotlib: pip install matplotlib")


def save_results(solution, problem, analysis=None, filename=None):
    """Save results to file"""
    if filename is None:
        filename = f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_solution_to_file(solution, filename)
    print(f"Results saved to: {filename}")


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

    # Handle different modes
    if args.demo:
        if args.demo == "basic":
            solution, problem, solver = run_basic_demonstration()
        elif args.demo == "comprehensive":
            solution, problem, solver, analysis = run_comprehensive_demonstration(
                live=getattr(args, "live", False)
            )
        elif args.demo == "benchmark":
            results = run_benchmark_demonstration()
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
        solution, problem, solver, analysis = run_comprehensive_demonstration(
            live=getattr(args, "live", False)
        )
        # Optionally enable live plotting by wiring an iteration callback on the solver
        visualizer = None
        if getattr(args, "live", False):
            try:
                visualizer = RouteVisualizer(problem, live=True)
                visualizer.start_live(title=f"Live - {problem.name}")

                def _iteration_callback(iteration_idx, best_solution):
                    try:
                        visualizer.update_live(
                            best_solution, getattr(solver, "convergence_history", [])
                        )
                    except Exception:
                        pass

                solver.iteration_callback = _iteration_callback
                print("Live plotting enabled for comprehensive demonstration.")
            except Exception:
                visualizer = None

        start_time = time.time()
        solution = solver.run(max_iterations=solver.max_iterations)
        end_time = time.time()

        # If live plotting was enabled, do a final update and stop interactive mode.
        if visualizer is not None:
            try:
                visualizer.update_live(
                    solution, getattr(solver, "convergence_history", [])
                )
                visualizer.stop_live()
            except Exception:
                pass

        # Analyze solution
        analyzer = PerformanceAnalyzer(problem)
        analysis = analyzer.analyze_solution(solution)

    # Visualization
    if args.verbose or args.save_plots:
        run_visualization_demo(solution, problem, save_plots=args.save_plots)

    # Save results
    if args.save_results:
        save_results(solution, problem, analysis)

    # Final summary
    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"Final Solution Cost: {solution.total_cost:.2f}")
    print(f"Total Routes: {len(solution.routes)}")
    print(f"Unassigned Customers: {len(solution.unassigned_customers)}")
    print(
        f"All customers assigned: {'Yes' if len(solution.unassigned_customers) == 0 else 'No'}"
    )

    if args.verbose:
        print(f"Solver completed {solver.iteration} iterations")
        print(f"Best solution found at iteration: {len(solver.convergence_history)}")


if __name__ == "__main__":
    main()
