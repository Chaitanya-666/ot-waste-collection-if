#!/usr/bin/env python3
"""
Enhanced CLI for the OT Waste Collection project

Provides higher-level command-line utilities for:
 - Running ALNS with configuration files
 - Constructing initial solutions with enhanced heuristics
 - Running the benchmarking suite
 - Validating solutions with detailed reports

This tool is intentionally non-intrusive: it uses existing modules (ALNS, data generator,
enhanced construction, benchmarking, and validator) and does not modify core algorithm code.
It focuses on orchestration, convenience, and producing human-readable outputs.

Usage examples:
  # Run ALNS solver using a config file
  python enhanced_cli.py solve --config config.json --iterations 500 --save-results

  # Construct an initial solution using enhanced constructions
  python enhanced_cli.py construct --strategy cluster_based --save-results

  # Run full benchmark suite
  python enhanced_cli.py benchmark --algorithms alns nearest_neighbor savings --runs 3 --out report.json

  # Validate a saved solution file (JSON created by utils.save_solution_to_file)
  python enhanced_cli.py validate --solution-file outputs/solution.json --report-file outputs/validation.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Prefer package-relative imports when possible; fall back to non-package imports.
try:
    from src.data_generator import DataGenerator  # type: ignore
    from src.alns import ALNS  # type: ignore
    from src.utils import save_solution_to_file, load_solution_from_file  # type: ignore
    from src.enhanced_construction import (
        EnhancedConstructionHeuristics,
        ConstructionStrategy,
    )  # type: ignore
    from src.benchmarking import BenchmarkingFramework, AlgorithmType  # type: ignore
    from src.enhanced_validator import EnhancedSolutionValidator  # type: ignore
except Exception:
    try:
        # fallback: assume running from package root where modules are top-level
        from data_generator import DataGenerator  # type: ignore
        from alns import ALNS  # type: ignore
        from utils import save_solution_to_file, load_solution_from_file  # type: ignore
        from enhanced_construction import (
            EnhancedConstructionHeuristics,
            ConstructionStrategy,
        )  # type: ignore
        from benchmarking import BenchmarkingFramework, AlgorithmType  # type: ignore
        from enhanced_validator import EnhancedSolutionValidator  # type: ignore
    except Exception as e:
        # If imports fail, we will handle this later with descriptive errors.
        ALNS = None  # type: ignore
        DataGenerator = None  # type: ignore
        save_solution_to_file = None  # type: ignore
        load_solution_from_file = None  # type: ignore
        EnhancedConstructionHeuristics = None  # type: ignore
        ConstructionStrategy = None  # type: ignore
        BenchmarkingFramework = None  # type: ignore
        AlgorithmType = None  # type: ignore
        EnhancedSolutionValidator = None  # type: ignore
        _import_error = e  # store for diagnostics

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("enhanced_cli")


def load_json_file(path: str) -> Dict[str, Any]:
    """Load JSON file and return dict."""
    with open(path, "r") as f:
        return json.load(f)


def apply_algorithm_config(solver: "ALNS", cfg: Dict[str, Any]) -> None:
    """Apply top-level algorithm configuration to an ALNS instance (non-destructive)."""
    if solver is None:
        return

    alg_cfg = cfg.get("algorithm", {})
    if "max_iterations" in alg_cfg:
        solver.max_iterations = int(alg_cfg["max_iterations"])
    if "seed" in alg_cfg:
        try:
            solver.seed = int(alg_cfg["seed"])
        except Exception:
            pass
    if "temperature" in alg_cfg:
        try:
            solver.temperature = float(alg_cfg["temperature"])
        except Exception:
            pass
    if "cooling_rate" in alg_cfg:
        try:
            solver.cooling_rate = float(alg_cfg["cooling_rate"])
        except Exception:
            pass
    if "learning_rate" in alg_cfg:
        try:
            solver.learning_rate = float(alg_cfg["learning_rate"])
        except Exception:
            pass
    if "adaptive_period" in alg_cfg:
        try:
            solver.adaptive_period = int(alg_cfg["adaptive_period"])
        except Exception:
            pass

    # Optionally apply operator weights if provided
    destroy_cfg = cfg.get("destroy_operators", {})
    repair_cfg = cfg.get("repair_operators", {})

    try:
        # Assign destroy weights
        for name, params in destroy_cfg.items():
            weight = params.get("weight")
            if (
                weight is not None
                and getattr(solver, "destroy_weights", None) is not None
            ):
                if name in solver.destroy_weights:
                    solver.destroy_weights[name] = float(weight)
    except Exception:
        logger.debug("Unable to apply destroy operator weights - skipping")

    try:
        # Assign repair weights
        for name, params in repair_cfg.items():
            weight = params.get("weight")
            # repair operators are stored by name in solver.repair_weights
            if (
                weight is not None
                and getattr(solver, "repair_weights", None) is not None
            ):
                if name in solver.repair_weights:
                    solver.repair_weights[name] = float(weight)
    except Exception:
        logger.debug("Unable to apply repair operator weights - skipping")


def run_solver_on_instance(
    problem: Any,
    cfg: Optional[Dict[str, Any]] = None,
    iterations: Optional[int] = None,
    save_results: bool = False,
    results_dir: str = "outputs",
) -> Any:
    """
    Run the ALNS solver on a given ProblemInstance.

    Returns the solution object (as produced by solver.run()).
    """
    if ALNS is None:
        raise RuntimeError(
            "ALNS implementation not available in imports. Check project layout."
        )

    solver = ALNS(problem)

    # Apply config (if provided)
    if cfg:
        apply_algorithm_config(solver, cfg)

    # Override iterations if explicitly provided
    if iterations is not None:
        solver.max_iterations = int(iterations)

    logger.info(f"Running ALNS for up to {solver.max_iterations} iterations...")
    start_time = time.time()
    solution = solver.run(max_iterations=solver.max_iterations)
    elapsed = time.time() - start_time
    logger.info(f"ALNS finished in {elapsed:.3f}s")

    # Optionally save results
    if save_results:
        if save_solution_to_file is None:
            logger.warning("save_solution_to_file not available; cannot save results.")
        else:
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"solution_{timestamp}.json")
            try:
                save_solution_to_file(solution, filename)
                logger.info(f"Solution saved to: {filename}")
            except Exception as e:
                logger.warning(f"Failed to save solution: {e}")

    return solution


def run_construction(
    problem: Any,
    strategy_name: Optional[str] = None,
    multi_start: int = 1,
    save_results: bool = False,
    results_dir: str = "outputs",
) -> Any:
    """Run enhanced construction heuristics and return the best constructed Solution."""
    if EnhancedConstructionHeuristics is None or ConstructionStrategy is None:
        raise RuntimeError(
            "Enhanced construction module not available in imports. Check project layout."
        )

    # Map strategy name to enum (if provided)
    strategy = None
    if strategy_name:
        try:
            strategy = ConstructionStrategy(strategy_name)
        except Exception:
            # Try to match by name ignoring case / underscores
            normalized = strategy_name.strip().lower().replace("-", "_")
            for s in ConstructionStrategy:
                if s.value == normalized or s.name.lower() == normalized:
                    strategy = s
                    break
    # Create heuristics object
    heur = EnhancedConstructionHeuristics(problem)

    best_result = None
    if multi_start <= 1:
        result = heur.construct_solution(strategy)
        best_result = result
    else:
        best_result = heur.multi_start_construction(num_starts=multi_start)

    # Save result if requested
    if save_results:
        if save_solution_to_file is None:
            logger.warning(
                "save_solution_to_file not available; cannot save constructed solution."
            )
        else:
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"constructed_{timestamp}.json")
            try:
                save_solution_to_file(best_result.solution, filename)
                logger.info(f"Constructed solution saved to: {filename}")
            except Exception as e:
                logger.warning(f"Failed to save constructed solution: {e}")

    return best_result


def run_benchmark(
    algorithms: List[str],
    runs: int = 1,
    out_file: Optional[str] = None,
    seed: int = 42,
) -> List[Any]:
    """Run benchmarking suite for the requested algorithms."""
    if BenchmarkingFramework is None or AlgorithmType is None:
        raise RuntimeError(
            "Benchmarking module not available in imports. Check project layout."
        )

    framework = BenchmarkingFramework(seed=seed)
    framework.create_standard_benchmarks()

    # Translate algorithm strings to AlgorithmType or available algorithm list
    alg_types = []
    for a in algorithms:
        try:
            # allow passing "alns" or "ALNS"
            alg_types.append(AlgorithmType(a.lower()))
        except Exception:
            # try mapping common names
            norm = a.strip().lower()
            mapping = {
                "alns": AlgorithmType.ALNS,
                "nearest_neighbor": AlgorithmType.NEAREST_NEIGHBOR,
                "savings": AlgorithmType.SAVINGS,
                "random": AlgorithmType.RANDOM,
            }
            if norm in mapping:
                alg_types.append(mapping[norm])
            else:
                logger.warning(f"Unknown algorithm '{a}' - skipping.")

    if not alg_types:
        # default: run a small set
        alg_types = [
            AlgorithmType.ALNS,
            AlgorithmType.NEAREST_NEIGHBOR,
            AlgorithmType.SAVINGS,
        ]

    results = framework.run_benchmark_suite(alg_types, iterations_per_algorithm=runs)

    # Optionally save benchmark report
    if out_file:
        try:
            report = framework.generate_benchmark_report(results)
            with open(out_file, "w") as f:
                f.write(report)
            logger.info(f"Benchmark report written to {out_file}")
        except Exception as e:
            logger.warning(f"Unable to write benchmark report to {out_file}: {e}")

    return results


def validate_solution_file(
    solution_file: str,
    report_file: Optional[str] = None,
    benchmark: bool = False,
) -> Any:
    """
    Validate a saved solution JSON file and optionally write a textual report.

    Expects a solution file format compatible with `utils.save_solution_to_file`.
    """
    if load_solution_from_file is None or EnhancedSolutionValidator is None:
        raise RuntimeError(
            "Validation utilities not available in imports. Check project layout."
        )

    logger.info(f"Loading solution from: {solution_file}")
    solution = load_solution_from_file(solution_file)
    if getattr(solution, "problem", None) is None:
        # If problem reference is missing from loaded solution, attempt to rebuild from metadata
        logger.debug(
            "Loaded solution missing problem reference. Attempting to reconstruct problem from solution data."
        )
    validator = EnhancedSolutionValidator(solution.problem)
    result = validator.validate_solution(
        solution, check_quality=True, benchmark=benchmark
    )
    report_text = validator.generate_validation_report(result)

    if report_file:
        try:
            os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
            with open(report_file, "w") as f:
                f.write(report_text)
            logger.info(f"Validation report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save validation report: {e}")
    else:
        # Print to stdout
        print(report_text)

    return result


def ensure_data_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="enhanced_cli.py",
        description="Enhanced CLI for OT Waste Collection - configuration, benchmarking, validation, construction helpers",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # Solve command
    solve_p = sub.add_parser(
        "solve", help="Run ALNS solver on a generated or provided instance"
    )
    solve_p.add_argument("--config", help="Path to configuration JSON file (optional)")
    solve_p.add_argument("--instance", help="Path to instance JSON file (optional)")
    solve_p.add_argument(
        "--iterations", type=int, help="Override iterations (max_iterations)"
    )
    solve_p.add_argument(
        "--save-results", action="store_true", help="Save solution to outputs/"
    )
    solve_p.add_argument(
        "--results-dir", default="outputs", help="Directory to save outputs"
    )

    # Construct command
    construct_p = sub.add_parser(
        "construct", help="Construct initial solution(s) using enhanced heuristics"
    )
    construct_p.add_argument(
        "--strategy", help="Construction strategy (e.g., cluster_based, greedy_nearest)"
    )
    construct_p.add_argument(
        "--multi-start", type=int, default=1, help="Multi-start attempts"
    )
    construct_p.add_argument(
        "--save-results", action="store_true", help="Save constructed solution"
    )
    construct_p.add_argument(
        "--results-dir", default="outputs", help="Directory to save outputs"
    )
    construct_p.add_argument("--instance", help="Path to instance JSON file (optional)")

    # Benchmark command
    bench_p = sub.add_parser("benchmark", help="Run benchmarking suite")
    bench_p.add_argument(
        "--algorithms",
        nargs="+",
        default=["alns", "nearest_neighbor", "savings"],
        help="List of algorithms to benchmark",
    )
    bench_p.add_argument("--runs", type=int, default=1, help="Runs per algorithm")
    bench_p.add_argument("--out", help="Output file for benchmark report (text)")

    # Validate command
    val_p = sub.add_parser("validate", help="Validate a saved solution JSON file")
    val_p.add_argument(
        "--solution-file", required=True, help="Path to a saved solution JSON file"
    )
    val_p.add_argument(
        "--report-file", help="Write textual validation report to this path"
    )
    val_p.add_argument(
        "--benchmark",
        action="store_true",
        help="Include benchmark metrics in validation",
    )

    # Misc
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for generation/algorithms",
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Quick import validation
    if ALNS is None or DataGenerator is None:
        logger.error(
            "Project imports not available. Ensure you're running from the project root and that src/ is importable."
        )
        try:
            logger.debug(f"Import error: {_import_error}")  # may exist
        except Exception:
            pass
        return 2

    seed = getattr(args, "seed", 42)

    try:
        if args.command == "solve":
            cfg = None
            if args.config:
                cfg = load_json_file(args.config)
            # If an instance file is provided, load it, else create a small default instance
            if args.instance:
                problem = DataGenerator.load_instance_from_file(args.instance)
            else:
                # Generate a default problem for quick experiments
                logger.info("No instance provided - generating a small demo instance.")
                problem = DataGenerator.generate_instance(
                    name="cli_demo",
                    n_customers=15,
                    n_ifs=2,
                    vehicle_capacity=25,
                    seed=seed,
                )
            sol = run_solver_on_instance(
                problem,
                cfg=cfg,
                iterations=getattr(args, "iterations", None),
                save_results=args.save_results,
                results_dir=args.results_dir,
            )
            # Print quick summary
            try:
                print(f"Solution cost: {sol.total_cost:.2f}")
                print(f"Routes: {len(sol.routes)}")
            except Exception:
                print(
                    "Solver completed - check saved outputs or enable --verbose for details."
                )

        elif args.command == "construct":
            # Load or generate instance
            if args.instance:
                problem = DataGenerator.load_instance_from_file(args.instance)
            else:
                problem = DataGenerator.generate_instance(
                    name="cli_construct_demo",
                    n_customers=20,
                    n_ifs=3,
                    vehicle_capacity=25,
                    seed=seed,
                )
            result = run_construction(
                problem,
                strategy_name=getattr(args, "strategy", None),
                multi_start=getattr(args, "multi_start", 1),
                save_results=getattr(args, "save_results", False),
                results_dir=getattr(args, "results_dir", "outputs"),
            )
            print(f"Constructed solution cost: {result.solution.total_cost:.2f}")
            print(f"Construction time: {result.construction_time:.3f}s")
            print(f"Strategy used: {result.strategy.value}")

        elif args.command == "benchmark":
            out = getattr(args, "out", None)
            results = run_benchmark(
                args.algorithms, runs=getattr(args, "runs", 1), out_file=out, seed=seed
            )
            print("Benchmarking complete.")
            if out:
                print(f"Report written to: {out}")

        elif args.command == "validate":
            res = validate_solution_file(
                solution_file=args.solution_file,
                report_file=getattr(args, "report_file", None),
                benchmark=getattr(args, "benchmark", False),
            )
            print(f"Validation status: {res.status.value}")

        else:
            parser.print_help()

    except Exception as e:
        logger.exception(f"An error occurred while executing the command: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
