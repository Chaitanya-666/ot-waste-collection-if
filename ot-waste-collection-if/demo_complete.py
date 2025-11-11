#!/usr/bin/env python3
"""
Complete demonstration of the OT Project: Municipal Waste Collection with ALNS

This script demonstrates the complete functionality including:
- Instance generation
- ALNS optimization
- Performance analysis
- Visualization
- Export capabilities
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .src.problem import ProblemInstance, Location
    from .src.solution import Solution, Route
    from .src.alns import ALNS
    from .src.data_generator import DataGenerator
    from .src.utils import RouteVisualizer, PerformanceAnalyzer, save_solution_to_file
    from .src.destroy_operators import DestroyOperatorManager
    from .src.repair_operators import RepairOperatorManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the correct location")
    sys.exit(1)


def create_demo_instance(size: str = "medium") -> ProblemInstance:
    """Create a demonstration instance based on size"""

    if size == "small":
        return DataGenerator.generate_instance(
            name="Small Demo Instance",
            n_customers=8,
            n_ifs=1,
            vehicle_capacity=20,
            area_size=100,
            demand_range=(1, 8),
            service_time_range=(1, 4),
            cluster_factor=0.3,
            seed=42,
        )
    elif size == "medium":
        return DataGenerator.generate_instance(
            name="Medium Demo Instance",
            n_customers=20,
            n_ifs=2,
            vehicle_capacity=30,
            area_size=150,
            demand_range=(1, 12),
            service_time_range=(1, 6),
            cluster_factor=0.5,
            seed=123,
        )
    else:  # large
        return DataGenerator.generate_instance(
            name="Large Demo Instance",
            n_customers=50,
            n_ifs=3,
            vehicle_capacity=40,
            area_size=200,
            demand_range=(2, 20),
            service_time_range=(2, 10),
            cluster_factor=0.7,
            seed=456,
        )


def run_alns_optimization(
    problem: ProblemInstance, iterations: int = 500
) -> Tuple[ALNS, Solution]:
    """Run ALNS optimization on the problem instance"""

    print(f"\nStarting ALNS optimization with {iterations} iterations...")
    print(f"Problem: {problem.name}")
    print(
        f"Customers: {len(problem.customers)}, IFs: {len(problem.intermediate_facilities)}"
    )
    print(f"Vehicle Capacity: {problem.vehicle_capacity}")

    # Ensure enough vehicles
    min_vehicles = int(problem.get_min_vehicles_needed())
    if problem.number_of_vehicles < min_vehicles:
        problem.number_of_vehicles = min_vehicles
        print(f"Adjusted number of vehicles to: {problem.number_of_vehicles}")

    # Check feasibility
    feasible, msg = problem.is_feasible()
    if not feasible:
        print(f"Problem infeasible: {msg}")
        return None, None

    # Create and run solver
    solver = ALNS(problem)
    solver.max_iterations = iterations

    start_time = time.time()
    solution = solver.run()
    end_time = time.time()

    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

    return solver, solution


def analyze_and_visualize(
    problem: ProblemInstance, solution: Solution, solver: ALNS, output_dir: str
):
    """Analyze solution and create visualizations"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Performance analysis
    print("\n=== PERFORMANCE ANALYSIS ===")
    analyzer = PerformanceAnalyzer(problem)
    analysis = analyzer.analyze_solution(solution)

    print(f"Total Cost: {analysis['total_cost']:.2f}")
    print(f"Total Distance: {analysis['total_distance']:.2f}")
    print(f"Total Time: {analysis['total_time']:.2f}")
    print(f"Vehicles Used: {analysis['num_vehicles']}")
    print(f"Unassigned Customers: {analysis['num_unassigned']}")
    print(f"IF Visits: {analysis['if_visits']}")

    # Efficiency metrics
    eff = analysis["efficiency_metrics"]
    print(f"\nEfficiency Metrics:")
    print(f"  Distance Efficiency: {eff['distance_efficiency']:.3f}")
    print(f"  Capacity Utilization: {eff['capacity_utilization']:.1%}")
    print(f"  Vehicle Efficiency: {eff['vehicle_efficiency']:.1%}")
    print(f"  IF Efficiency: {eff['if_efficiency']:.1f} visits/vehicle")

    # Route details
    print(f"\nRoute Details:")
    for route_info in analysis["route_details"]:
        print(
            f"  Vehicle {route_info['vehicle_id']}: "
            f"Distance={route_info['distance']:.2f}, "
            f"Time={route_info['time']:.2f}, "
            f"Customers={route_info['customers_served']}, "
            f"IF Visits={route_info['if_visits']}, "
            f"Max Load={route_info['max_load']:.1f}/{problem.vehicle_capacity}"
        )

    # Generate visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    visualizer = RouteVisualizer(problem)

    # Route plot
    try:
        fig = visualizer.plot_solution(solution, f"Routes - {problem.name}")
        route_file = os.path.join(output_dir, f"routes_{timestamp}.png")
        fig.savefig(route_file, dpi=300, bbox_inches="tight")
        print(f"Route plot saved: {route_file}")
    except Exception as e:
        print(f"Route visualization failed: {e}")

    # Convergence plot
    try:
        if hasattr(solver, "convergence_history") and solver.convergence_history:
            conv_fig = visualizer.plot_convergence(solver.convergence_history)
            conv_file = os.path.join(output_dir, f"convergence_{timestamp}.png")
            conv_fig.savefig(conv_file, dpi=300, bbox_inches="tight")
            print(f"Convergence plot saved: {conv_file}")
    except Exception as e:
        print(f"Convergence visualization failed: {e}")

    # Save solution to JSON
    try:
        solution_file = os.path.join(output_dir, f"solution_{timestamp}.json")
        save_solution_to_file(solution, solution_file)
        print(f"Solution saved: {solution_file}")
    except Exception as e:
        print(f"Solution export failed: {e}")

    # Generate performance report
    try:
        report = analyzer.generate_report(solution)
        report_file = os.path.join(output_dir, f"report_{timestamp}.txt")
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Performance report saved: {report_file}")
    except Exception as e:
        print(f"Report generation failed: {e}")


def demonstrate_operators(problem: ProblemInstance):
    """Demonstrate destroy and repair operators"""

    print("\n=== OPERATOR DEMONSTRATION ===")

    # Create a simple solution
    solution = Solution(problem)
    route = Route()
    route.nodes = [problem.depot] + problem.customers[:5] + [problem.depot]
    route.calculate_metrics(problem)
    solution.routes = [route]
    solution.unassigned_customers = set()
    solution.calculate_metrics()

    print(f"Initial solution cost: {solution.total_cost:.2f}")

    # Test destroy operators
    print("\n--- Destroy Operators ---")
    destroy_manager = DestroyOperatorManager(problem)

    for name, operator in destroy_manager.operators.items():
        print(f"Testing {name}...")
        try:
            partial = operator.apply(solution.copy(), removal_count=3)
            removed_count = len(partial.unassigned_customers)
            print(f"  Removed {removed_count} customers")
        except Exception as e:
            print(f"  Failed: {e}")

    # Test repair operators
    print("\n--- Repair Operators ---")
    repair_manager = RepairOperatorManager()

    # Create a partial solution for repair testing
    partial_solution = solution.copy()
    if len(partial_solution.routes) > 0 and len(partial_solution.routes[0].nodes) > 4:
        # Remove some customers to create unassigned
        removed_customers = partial_solution.routes[0].nodes[2:4]
        for customer in removed_customers:
            if hasattr(customer, "id"):
                partial_solution.unassigned_customers.add(customer.id)
        partial_solution.routes[0].nodes = (
            [partial_solution.routes[0].nodes[0]]
            + [partial_solution.routes[0].nodes[1]]
            + [partial_solution.routes[0].nodes[4]]
            + [partial_solution.routes[0].nodes[-1]]
        )
        partial_solution.routes[0].calculate_metrics(problem)
        partial_solution.calculate_metrics()

    for operator in repair_manager.operators:
        print(f"Testing {operator.name}...")
        try:
            repaired = operator.apply(partial_solution.copy())
            assigned_count = len(partial_solution.unassigned_customers) - len(
                repaired.unassigned_customers
            )
            print(f"  Assigned {assigned_count} customers")
        except Exception as e:
            print(f"  Failed: {e}")


def run_benchmark_comparison():
    """Run benchmark comparison across different instance sizes"""

    print("\n=== BENCHMARK COMPARISON ===")

    sizes = ["small", "medium", "large"]
    iterations = 200
    results = {}

    for size in sizes:
        print(f"\n--- {size.upper()} INSTANCE ---")

        # Create instance
        problem = create_demo_instance(size)

        # Run optimization
        solver, solution = run_alns_optimization(problem, iterations)

        if solution is not None:
            # Analyze results
            analyzer = PerformanceAnalyzer(problem)
            analysis = analyzer.analyze_solution(solution)

            results[size] = {
                "customers": len(problem.customers),
                "ifs": len(problem.intermediate_facilities),
                "cost": solution.total_cost,
                "distance": solution.total_distance,
                "vehicles": analysis["num_vehicles"],
                "time": solution.total_time,
                "unassigned": len(solution.unassigned_customers),
                "convergence_iterations": len(solver.convergence_history)
                if hasattr(solver, "convergence_history")
                else 0,
            }

            print(
                f"Results: Cost={results[size]['cost']:.2f}, "
                f"Vehicles={results[size]['vehicles']}, "
                f"Unassigned={results[size]['unassigned']}"
            )

    # Summary comparison
    print(f"\n=== BENCHMARK SUMMARY ===")
    print(
        f"{'Size':<10} {'Customers':<10} {'Cost':<12} {'Vehicles':<10} {'Unassigned':<12} {'Conv Iters':<12}"
    )
    print("-" * 70)

    for size, data in results.items():
        print(
            f"{size:<10} {data['customers']:<10} {data['cost']:<12.2f} "
            f"{data['vehicles']:<10} {data['unassigned']:<12} {data['convergence_iterations']:<12}"
        )

    return results


def main():
    """Main demonstration function"""

    print("=" * 80)
    print("OT PROJECT: MUNICIPAL WASTE COLLECTION OPTIMIZATION WITH ALNS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    output_dir = "demo_outputs"
    instance_size = "medium"
    iterations = 300

    # Create demo instance
    print(f"\nCreating {instance_size} demonstration instance...")
    problem = create_demo_instance(instance_size)

    # Demonstrate operators
    demonstrate_operators(problem)

    # Run optimization
    solver, solution = run_alns_optimization(problem, iterations)

    if solution is not None:
        # Analyze and visualize
        analyze_and_visualize(problem, solution, solver, output_dir)

        # Run benchmark comparison
        benchmark_results = run_benchmark_comparison()

        # Final summary
        print(f"\n=== DEMONSTRATION COMPLETE ===")
        print(f"All outputs saved to: {os.path.abspath(output_dir)}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Success criteria check
        success_criteria = {
            "All customers assigned": len(solution.unassigned_customers) == 0,
            "Valid solution cost": solution.total_cost > 0,
            "Convergence tracked": hasattr(solver, "convergence_history")
            and len(solver.convergence_history) > 0,
            "Visualization created": os.path.exists(
                os.path.join(
                    output_dir, f"routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            ),
            "Solution exported": os.path.exists(
                os.path.join(
                    output_dir,
                    f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
            ),
        }

        print(f"\n=== SUCCESS CRITERIA ===")
        for criterion, met in success_criteria.items():
            status = "✓" if met else "✗"
            print(f"{status} {criterion}")

        overall_success = all(success_criteria.values())
        print(f"\nOverall Success: {'✓ PASSED' if overall_success else '✗ FAILED'}")

        return overall_success
    else:
        print("Optimization failed - no solution produced")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
