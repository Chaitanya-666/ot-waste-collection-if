#!/usr/bin/env python3
"""
Comprehensive demo for the VRP-IF ALNS project.

This script:
 - generates a synthetic instance (configurable)
 - runs the ALNS solver
 - saves a solution JSON and PNG plots (routes and convergence)
 - prints summary metrics to stdout

Usage:
    python demos/comprehensive_demo.py
    python demos/comprehensive_demo.py --customers 30 --ifs 3 --iterations 800 --save-plots

Note:
 The demo imports project modules from the `src` package. When running from the
 project root this script will add the `src` directory to `sys.path`.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Ensure the 'src' package is importable when running from the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import project modules (located in src/)
try:
    from src.data_generator import DataGenerator  # type: ignore
except Exception:
    # fallback: some installations may expose module names without package prefix
    from data_generator import DataGenerator  # type: ignore

try:
    from src.alns import ALNS  # type: ignore
except Exception:
    from alns import ALNS  # type: ignore

try:
    from src.utils import RouteVisualizer, PerformanceAnalyzer, save_solution_to_file  # type: ignore
except Exception:
    from utils import RouteVisualizer, PerformanceAnalyzer, save_solution_to_file  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive demo for VRP-IF ALNS")
    p.add_argument(
        "--customers",
        type=int,
        default=20,
        help="Number of customers in synthetic instance",
    )
    p.add_argument(
        "--ifs", type=int, default=2, help="Number of intermediate facilities"
    )
    p.add_argument(
        "--capacity", type=int, default=30, help="Vehicle capacity for the instance"
    )
    p.add_argument(
        "--area-size",
        type=int,
        default=120,
        help="Spatial area size (coordinates range)",
    )
    p.add_argument("--iterations", type=int, default=500, help="ALNS iterations")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--save-plots", action="store_true", help="Save route and convergence plots"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "outputs"),
        help="Directory for outputs",
    )
    return p.parse_args()


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    print("=== VRP-IF ALNS: Comprehensive demo ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(
        f"Instance: customers={args.customers}, IFs={args.ifs}, capacity={args.capacity}, area={args.area_size}"
    )
    print(f"ALNS iterations: {args.iterations}, seed: {args.seed}")
    print()

    # 1) Generate instance
    problem = DataGenerator.generate_instance(
        name=f"Demo_{args.customers}C_{args.ifs}IF",
        n_customers=args.customers,
        n_ifs=args.ifs,
        vehicle_capacity=args.capacity,
        area_size=args.area_size,
        demand_range=(1, 12),
        service_time_range=(1, 4),
        seed=args.seed,
        cluster_factor=0.4,
    )

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
        # If anything goes wrong here, continue — the subsequent sanity check will catch infeasibility.
        pass

    # sanity check
    feasible_flag, feasible_msg = problem.is_feasible()
    print(f"Problem feasibility check: {feasible_flag} ({feasible_msg})")
    if not feasible_flag:
        print("Instance is infeasible. Exiting demo.")
        return

    # 2) Setup solver
    solver = ALNS(problem)
    solver.max_iterations = args.iterations

    # 3) Run solver
    print("\nStarting ALNS optimization...")
    t0 = time.time()
    solution = solver.run(max_iterations=solver.max_iterations)
    elapsed = time.time() - t0
    print(f"ALNS finished in {elapsed:.2f}s")

    # 4) Print brief summary
    total_cost = getattr(
        solution, "total_cost", getattr(solution, "total_distance", None)
    )
    total_distance = getattr(solution, "total_distance", None)
    total_time = getattr(solution, "total_time", None)
    n_routes = len(solution.routes) if hasattr(solution, "routes") else 0
    n_unassigned = (
        len(solution.unassigned_customers)
        if hasattr(solution, "unassigned_customers")
        else None
    )

    print("\n=== SOLUTION SUMMARY ===")
    print(f"Total cost: {total_cost}")
    print(f"Total distance: {total_distance}")
    print(f"Total time: {total_time}")
    print(f"Routes found: {n_routes}")
    print(f"Unassigned customers: {n_unassigned}")
    print(
        f"Convergence history length: {len(solver.convergence_history) if hasattr(solver, 'convergence_history') else 'N/A'}"
    )

    # 5) Save solution JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sol_file = os.path.join(args.output_dir, f"solution_{ts}.json")
    try:
        save_solution_to_file(solution, sol_file)
    except Exception as e:
        print(f"Warning: could not save solution JSON: {e}")

    # 6) Plot and save visuals (if requested)
    if args.save_plots:
        try:
            visualizer = RouteVisualizer(problem)

            print("Generating route plot...")
            fig = visualizer.plot_solution(solution, title=f"Routes - {problem.name}")
            route_png = os.path.join(args.output_dir, f"routes_{ts}.png")
            fig.savefig(route_png, dpi=200, bbox_inches="tight")
            print(f"Saved route plot to: {route_png}")
            try:
                fig.clf()
            except Exception:
                pass

            print("Generating convergence plot...")
            conv_fig = visualizer.plot_convergence(
                getattr(solver, "convergence_history", []), title="ALNS Convergence"
            )
            conv_png = os.path.join(args.output_dir, f"convergence_{ts}.png")
            conv_fig.savefig(conv_png, dpi=200, bbox_inches="tight")
            print(f"Saved convergence plot to: {conv_png}")
            try:
                conv_fig.clf()
            except Exception:
                pass

        except Exception as e:
            print(f"Warning: plotting failed: {e}")
            print(
                "If you want plotting, ensure matplotlib is installed (pip install matplotlib)."
            )

    # 7) Detailed analysis
    try:
        analyzer = PerformanceAnalyzer(problem)
        analysis = analyzer.analyze_solution(solution)
        print("\n=== DETAILED ANALYSIS ===")
        print(f"Vehicles used: {analysis.get('num_vehicles')}")
        print(f"IF visits (total): {analysis.get('if_visits')}")
        print(
            f"Capacity utilization (per vehicle): {analysis.get('vehicle_utilization')}"
        )
        print(f"Route details (summary):")
        for rd in analysis.get("route_details", []):
            print(
                f"  V{rd['vehicle_id']}: dist={rd['distance']:.2f}, time={rd['time']:.2f}, customers={rd['customers_served']}, IFs={rd['if_visits']}, max_load={rd['max_load']}"
            )
    except Exception as e:
        print(f"Warning: performance analysis failed: {e}")

    print("\nDemo outputs directory:", os.path.abspath(args.output_dir))
    print("Demo complete.")


if __name__ == "__main__":
    main()
