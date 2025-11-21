# Author: Harsh Sharma (231070064)
#
# This file serves as the main entry point for the application. It provides a
# command-line interface (CLI) for users to interact with the VRP solver.
# It handles argument parsing, orchestrates the different modules (data generation,
# solving, visualization), and provides several demonstration modes.
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

# Video creation imports (if available)
try:
    # Try importing from workspace root first, then current directory
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from simple_video_creator import SimpleVideoCreator
    VIDEO_CREATOR_AVAILABLE = True
except ImportError:
    try:
        from simple_video_creator import SimpleVideoCreator
        VIDEO_CREATOR_AVAILABLE = True
    except ImportError:
        VIDEO_CREATOR_AVAILABLE = False


class OptimizationVideoTracker:
    """
    Tracks the state of the optimization process at different iterations
    to generate a video of the solution's evolution.
    """
    
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.optimization_history = []
        self.current_best_cost = float('inf')
        
    def track_state(self, iteration: int, solution, current_cost: float):
        """Records the current state of the solution for a single frame of the video."""
        
        # Extract route coordinates for visualization
        routes_coords = []
        if solution and hasattr(solution, 'routes'):
            for route in solution.routes:
                route_coords = []
                for node in route.nodes:
                    route_coords.append((node.x, node.y))
                routes_coords.append(route_coords)
        
        state = {
            'iteration': iteration,
            'cost': current_cost,
            'best_cost': self.current_best_cost,
            'routes': routes_coords
        }
        
        # Update best cost found so far
        if current_cost < self.current_best_cost:
            self.current_best_cost = current_cost
            
        self.optimization_history.append(state)
        
    def create_video(self, output_filename: str = None) -> str:
        """Creates and saves an optimization video from the tracked history."""
        if not VIDEO_CREATOR_AVAILABLE:
            print("⚠️ Video creation not available - install requirements")
            return None
            
        if not self.optimization_history:
            print("⚠️ No optimization history to create video")
            return None
            
        try:
            # Prepare data in the format expected by the video creator
            customer_data = {}
            for customer in self.problem.customers:
                customer_data[(customer.x, customer.y)] = customer.demand
                
            intermediate_facs = [(ifac.x, ifac.y) for ifac in self.problem.intermediate_facilities]
            depot_location = (self.problem.depot.x, self.problem.depot.y)
            
            # Initialize video creator
            video_creator = SimpleVideoCreator()
            
            # Create route evolution video
            if output_filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"alns_optimization_{timestamp}.gif"
                
            video_path = video_creator.create_optimization_animation(
                optimization_history=self.optimization_history,
                customer_data=customer_data,
                depot_location=depot_location,
                intermediate_facilities=intermediate_facs,
                output_filename=output_filename
            )
            
            if video_path:
                print(f"🎬 Optimization video created: {video_path}")
                
                # Also create a video of the cost convergence
                costs = [state['cost'] for state in self.optimization_history]
                cost_filename = output_filename.replace('.gif', '_cost.gif')
                cost_path = video_creator.create_cost_animation(
                    costs=costs,
                    output_filename=cost_filename
                )
                if cost_path:
                    print(f"📊 Cost convergence video created: {cost_path}")
                    
            return video_path
            
        except Exception as e:
            print(f"⚠️ Video creation failed: {e}")
            return None


def create_sample_instance() -> ProblemInstance:
    """Creates a small, hardcoded problem instance for basic demonstration."""
    print("Creating sample problem instance...")

    problem = ProblemInstance("Sample Instance")
    problem.vehicle_capacity = 20
    problem.number_of_vehicles = 3
    problem.disposal_time = 2

    # Add depot, customers, and IFs
    depot = Location(0, 0, 0, 0, "depot")
    problem.depot = depot
    if1 = Location(100, 20, 20, 0, "if")
    problem.intermediate_facilities.append(if1)
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
    """Creates a more complex, randomly generated instance for demonstration."""
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


def run_basic_demonstration(create_video: bool = False, args=None):
    """Runs a simple demonstration with a small, fixed problem instance."""
    print("\n" + "=" * 60)
    print("BASIC DEMONSTRATION")
    print("=" * 60)

    problem = create_sample_instance()
    print(f"\nProblem: {problem}")

    # Initialize and run the ALNS solver
    solver = ALNS(problem)
    solver.max_iterations = 200

    # Setup video tracking if requested
    video_tracker = None
    if create_video:
        if VIDEO_CREATOR_AVAILABLE:
            video_tracker = OptimizationVideoTracker(problem)
            print("🎬 Video tracking enabled!")
        else:
            print("⚠️ Video tracking requested but video creator not available")

    if video_tracker:
        def _video_callback(iteration_idx, best_solution):
            if iteration_idx % 10 == 0 or iteration_idx == solver.max_iterations:
                current_cost = getattr(best_solution, 'total_cost', float('inf'))
                video_tracker.track_state(iteration_idx, best_solution, current_cost)
        solver.iteration_callback = _video_callback

    print(f"\nStarting ALNS optimization with {solver.max_iterations} iterations...")
    start_time = time.time()
    solution = solver.run(max_iterations=solver.max_iterations)
    end_time = time.time()

    # Create video if tracking was enabled
    if video_tracker:
        video_tracker.create_video()

    print(f"ALNS completed in {end_time - start_time:.2f} seconds")

    # Display basic results
    print("\n" + "-" * 40)
    print("OPTIMIZATION RESULTS")
    print("-" * 40)
    print(f"Final best solution cost: {solution.total_cost:.2f}")
    print(f"Routes: {len(solution.routes)}")
    for idx, route in enumerate(solution.routes):
        print(f"  Route {idx + 1}: {' -> '.join([f'{node.type[0].upper()}{node.id}' for node in route.nodes])}")

    return solution, problem, solver


def run_comprehensive_demonstration(
    live: bool = False, iterations: int = 500, create_video: bool = False, args=None
) -> Tuple[
    Optional[object], ProblemInstance, Optional[object], Optional[Dict[str, Any]]
]:
    """
    Runs a more comprehensive demonstration with a generated problem,
    detailed analysis, and optional live visualization.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)

    problem = create_comprehensive_instance()

    # Adjust vehicle count to ensure feasibility
    try:
        min_needed = int(problem.get_min_vehicles_needed())
        current_limit = problem.number_of_vehicles
        if current_limit == float('inf') or current_limit < min_needed or current_limit <= 0:
            problem.number_of_vehicles = min_needed
            print(f"Note: Adjusted vehicle count to minimum required: {problem.number_of_vehicles}")
    except Exception:
        pass

    # Check for basic problem feasibility before running
    feasible_flag, feasible_msg = problem.is_feasible()
    print(f"\nProblem: {problem}")
    if not feasible_flag:
        print(f"Problem feasibility check: {feasible_flag} ({feasible_msg})")
        return None, problem, None, None

    # Initialize solver and progress tracking
    solver = ALNS(problem)
    solver.max_iterations = int(iterations)
    ProgressTracker = create_progress_tracker()
    progress = ProgressTracker(solver.max_iterations)
    
    # Setup video tracking if requested
    video_tracker = None
    if create_video:
        if VIDEO_CREATOR_AVAILABLE:
            video_tracker = OptimizationVideoTracker(problem)
            print("🎬 Video tracking enabled!")
        else:
            print("⚠️ Video tracking requested but video creator not available")

    # Setup callbacks for progress updates and live plotting
    visualizer = None
    if live:
        try:
            visualizer = RouteVisualizer(problem, live=True)
            visualizer.start_live(title=f"Live - {problem.name}")
            def _iteration_callback(iteration_idx, best_solution):
                progress.update(iteration_idx)
                if video_tracker:
                    if iteration_idx % 10 == 0 or iteration_idx == solver.max_iterations:
                        current_cost = getattr(best_solution, 'total_cost', float('inf'))
                        video_tracker.track_state(iteration_idx, best_solution, current_cost)
                visualizer.update_live(best_solution, getattr(solver, "convergence_history", []))
            solver.iteration_callback = _iteration_callback
            print("🎬 Live plotting enabled.")
        except Exception:
            visualizer = None
    else:
        def _progress_callback(iteration_idx, best_solution):
            progress.update(iteration_idx)
            if video_tracker:
                if iteration_idx % 10 == 0 or iteration_idx == solver.max_iterations:
                    current_cost = getattr(best_solution, 'total_cost', float('inf'))
                    video_tracker.track_state(iteration_idx, best_solution, current_cost)
        solver.iteration_callback = _progress_callback

    # Run the optimization
    print(f"\n🚀 Starting ALNS optimization with {solver.max_iterations} iterations...")
    start_time = time.time()
    solution = solver.run(max_iterations=solver.max_iterations)
    end_time = time.time()
    print(f"ALNS completed in {end_time - start_time:.2f} seconds")

    if visualizer is not None:
        visualizer.stop_live()

    # Analyze and display detailed results
    analyzer = PerformanceAnalyzer(problem)
    analysis = analyzer.analyze_solution(solution)
    
    if video_tracker:
        video_tracker.create_video()

    print("\n" + "-" * 60)
    print("COMPREHENSIVE ANALYSIS")
    print("-" * 60)
    print(analyzer.generate_report(solution))

    return solution, problem, solver, analysis


def run_benchmark_demonstration():
    """Runs the solver on a suite of problems of increasing size and reports performance."""
    print("\n" + "=" * 60)
    print("BENCHMARK DEMONSTRATION")
    print("=" * 60)

    instances = [
        ("Small", 6, 1),
        ("Medium", 15, 2),
        ("Large", 25, 3),
    ]
    results = []

    for name, n_customers, n_ifs in instances:
        print(f"\n--- Benchmarking {name} Instance ---")
        problem = DataGenerator.generate_instance(
            f"{name} Benchmark", n_customers, n_ifs, seed=42
        )
        solver = ALNS(problem)
        solver.max_iterations = 300
        start_time = time.time()
        solution = solver.run(max_iterations=solver.max_iterations)
        end_time = time.time()

        results.append({
            "name": name, "cost": solution.total_cost, "time": end_time - start_time
        })

    # Display summary table
    print(f"\n" + "-" * 60)
    print("BENCHMARK SUMMARY")
    print("-" * 60)
    for result in results:
        print(f"  {result['name']:<10}: Cost={result['cost']:.2f}, Time={result['time']:.2f}s")

    return results


def main():
    """
    Main function that parses command-line arguments and runs the appropriate
    demonstration or solver mode.
    """
    parser = argparse.ArgumentParser(
        description="Municipal Waste Collection Route Optimization with ALNS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define CLI arguments
    parser.add_argument(
        "--demo",
        choices=["basic", "comprehensive", "benchmark"],
        help="Run a specific demonstration mode.",
    )
    parser.add_argument("--instance", help="Path to a problem instance file in JSON format.")
    parser.add_argument("--iterations", type=int, default=200, help="Number of ALNS iterations.")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots to files.")
    parser.add_argument("--live", action="store_true", help="Enable live plotting during optimization.")
    parser.add_argument("--save-results", action="store_true", help="Save final solution to a JSON file.")
    parser.add_argument("--video", action="store_true", help="Create a video of the optimization process.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed output.")

    args = parser.parse_args()

    # Run the selected mode
    if args.demo:
        if args.demo == "basic":
            run_basic_demonstration(create_video=args.video, args=args)
        elif args.demo == "comprehensive":
            run_comprehensive_demonstration(
                live=args.live, iterations=args.iterations, create_video=args.video, args=args
            )
        elif args.demo == "benchmark":
            run_benchmark_demonstration()
    elif args.instance:
        # Logic to load and run a specific instance file
        print(f"Loading instance from: {args.instance}")
        problem = DataGenerator.load_instance_from_file(args.instance)
        solver = ALNS(problem)
        solver.max_iterations = args.iterations
        solution = solver.run(max_iterations=solver.max_iterations)
        # Further processing...
    else:
        # Default action if no mode is specified
        print("No mode specified. Running comprehensive demonstration by default.")
        run_comprehensive_demonstration(
            live=args.live, iterations=args.iterations, create_video=args.video, args=args
        )

    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
