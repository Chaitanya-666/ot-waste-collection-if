#!/usr/bin/env python3
"""
Enhanced CLI for the OT Waste Collection project (improved imports & config handling)

This CLI is an orchestration layer that uses core project modules (ALNS, data_generator,
utils, enhanced construction, benchmarking, and validator). It intentionally does not
modify core algorithmic code. Improvements in this version:

- More robust import resolution:
  * Tries package-relative `src.*` imports
  * Falls back to top-level imports
  * If imports still fail, attempts to add the repository's package directory to PYTHONPATH
    automatically (when run from the repo root)
  * Gives detailed error messages explaining how to run the CLI in the correct context

- Improved configuration/template handling:
  * Supports passing either:
      - a path to a JSON configuration file via `--config /path/to/config.json`
      - a template key matching entries inside the `config_templates` JSON blob shipped
        in the package
  * Loads `config_templates` automatically from the package dir if present
  * Config merging: top-level CLI flags (like --iterations) override config contents

- Friendly usage and error messages for common developer mistakes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Basic logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("enhanced_cli")


# ---------------------------------------------------------------------------
# Import resolution helpers
# ---------------------------------------------------------------------------
def try_imports():
    """
    Try multiple import strategies to import project modules used by the CLI.

    Returns a dict with module references (or None) and an `error` if import failed.
    """
    modules = {
        "DataGenerator": None,
        "ALNS": None,
        "save_solution_to_file": None,
        "load_solution_from_file": None,
        "EnhancedConstructionHeuristics": None,
        "ConstructionStrategy": None,
        "BenchmarkingFramework": None,
        "AlgorithmType": None,
        "EnhancedSolutionValidator": None,
        "import_error": None,
    }

    # First attempt: package-relative imports from `src.*`
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

        modules.update(
            {
                "DataGenerator": DataGenerator,
                "ALNS": ALNS,
                "save_solution_to_file": save_solution_to_file,
                "load_solution_from_file": load_solution_from_file,
                "EnhancedConstructionHeuristics": EnhancedConstructionHeuristics,
                "ConstructionStrategy": ConstructionStrategy,
                "BenchmarkingFramework": BenchmarkingFramework,
                "AlgorithmType": AlgorithmType,
                "EnhancedSolutionValidator": EnhancedSolutionValidator,
            }
        )
        return modules
    except Exception as e1:
        modules["import_error"] = e1
        # Continue to next attempt

    # Second attempt: top-level imports (when running from package dir)
    try:
        from data_generator import DataGenerator  # type: ignore
        from alns import ALNS  # type: ignore
        from utils import save_solution_to_file, load_solution_from_file  # type: ignore
        from enhanced_construction import (
            EnhancedConstructionHeuristics,
            ConstructionStrategy,
        )  # type: ignore
        from benchmarking import BenchmarkingFramework, AlgorithmType  # type: ignore
        from enhanced_validator import EnhancedSolutionValidator  # type: ignore

        modules.update(
            {
                "DataGenerator": DataGenerator,
                "ALNS": ALNS,
                "save_solution_to_file": save_solution_to_file,
                "load_solution_from_file": load_solution_from_file,
                "EnhancedConstructionHeuristics": EnhancedConstructionHeuristics,
                "ConstructionStrategy": ConstructionStrategy,
                "BenchmarkingFramework": BenchmarkingFramework,
                "AlgorithmType": AlgorithmType,
                "EnhancedSolutionValidator": EnhancedSolutionValidator,
            }
        )
        return modules
    except Exception as e2:
        # store the second error if first wasn't present
        if modules["import_error"] is None:
            modules["import_error"] = e2

    # Third attempt: try to detect repository structure and add package dir to sys.path.
    # If this file is located at .../ot-waste-collection-if/ot-waste-collection-if/enhanced_cli.py
    # the package dir we want to add is .../ot-waste-collection-if/ot-waste-collection-if
    try:
        current_file = Path(__file__).resolve()
        # Walk upward to find a directory that contains 'src' or looks like the package root.
        candidate = current_file.parent
        found = None
        for _ in range(6):
            if (candidate / "src").is_dir() or (candidate / "__init__.py").exists():
                found = candidate
                break
            candidate = candidate.parent
        if found:
            if str(found) not in sys.path:
                sys.path.insert(0, str(found))
                logger.debug(f"Inserted {found} into sys.path for imports")
            # Try the top-level imports again
            from data_generator import DataGenerator  # type: ignore
            from alns import ALNS  # type: ignore
            from utils import save_solution_to_file, load_solution_from_file  # type: ignore
            from enhanced_construction import (
                EnhancedConstructionHeuristics,
                ConstructionStrategy,
            )  # type: ignore
            from benchmarking import BenchmarkingFramework, AlgorithmType  # type: ignore
            from enhanced_validator import EnhancedSolutionValidator  # type: ignore

            modules.update(
                {
                    "DataGenerator": DataGenerator,
                    "ALNS": ALNS,
                    "save_solution_to_file": save_solution_to_file,
                    "load_solution_from_file": load_solution_from_file,
                    "EnhancedConstructionHeuristics": EnhancedConstructionHeuristics,
                    "ConstructionStrategy": ConstructionStrategy,
                    "BenchmarkingFramework": BenchmarkingFramework,
                    "AlgorithmType": AlgorithmType,
                    "EnhancedSolutionValidator": EnhancedSolutionValidator,
                }
            )
            return modules
    except Exception as e3:
        # augment import_error for diagnostics
        modules["import_error"] = modules.get("import_error") or e3

    return modules


# Perform import attempts once
_MODULES = try_imports()


# ---------------------------------------------------------------------------
# Configuration / template loading
# ---------------------------------------------------------------------------
def locate_config_templates() -> Optional[Path]:
    """
    Try to find the `config_templates` JSON file shipped in the package.
    Returns a Path or None.
    """
    # Common locations (relative to this file)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "config_templates",
        here / "config_templates.json",
        here.parent / "config_templates",
        here.parent / "config_templates.json",
        here.parent / "config_templates.json",
        Path("config_templates"),
        Path("config_templates.json"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def load_config_templates() -> Dict[str, Any]:
    """
    Load templates from the `config_templates` JSON file if available.
    Returns an empty dict if not found or on read error.
    """
    path = locate_config_templates()
    if not path:
        logger.debug("No config_templates file found in known locations.")
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                logger.warning(
                    f"config_templates at {path} does not contain a JSON object."
                )
                return {}
    except Exception as e:
        logger.warning(f"Failed to load config_templates from {path}: {e}")
        return {}


# Preload templates (will be empty if file not present)
_CONFIG_TEMPLATES = load_config_templates()


def resolve_config(config_arg: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Resolve user-provided config argument.
    - If config_arg is None -> return None
    - If config_arg is a path to a file -> load and return JSON
    - If config_arg matches a key in the templates -> return that template dict
    - Otherwise, return None and print helpful message
    """
    if not config_arg:
        return None

    p = Path(config_arg)
    # 1) If it's an existing file, load it
    if p.exists() and p.is_file():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config file {p}: {e}")
            return None

    # 2) If it's a template key
    if _CONFIG_TEMPLATES and config_arg in _CONFIG_TEMPLATES:
        logger.debug(
            f"Loaded configuration template '{config_arg}' from config_templates."
        )
        return _CONFIG_TEMPLATES[config_arg]

    # 3) Not found: attempt to treat it as a JSON string
    try:
        parsed = json.loads(config_arg)
        if isinstance(parsed, dict):
            logger.debug("Parsed --config argument as inline JSON.")
            return parsed
    except Exception:
        pass

    logger.error(
        f"Config argument '{config_arg}' not recognized. It must be: path/to/config.json, a template name from config_templates, or inline JSON."
    )
    if _CONFIG_TEMPLATES:
        logger.info(
            f"Available templates: {', '.join(list(_CONFIG_TEMPLATES.keys())[:10])}"
        )
    return None


# ---------------------------------------------------------------------------
# CLI operations (orchestration only)
# ---------------------------------------------------------------------------
def run_solver_on_instance(
    problem: Any,
    cfg: Optional[Dict[str, Any]] = None,
    iterations: Optional[int] = None,
    save_results: bool = False,
    results_dir: str = "outputs",
) -> Any:
    """
    Run the ALNS solver on a given ProblemInstance.
    Uses the imported ALNS implementation resolved earlier.
    """
    ALNS = _MODULES.get("ALNS")
    save_solution_to_file = _MODULES.get("save_solution_to_file")
    if ALNS is None:
        raise RuntimeError(
            "ALNS implementation not available. Ensure you run this script from the project root or install the package."
        )

    solver = ALNS(problem)

    # Apply cfg options to solver (non-destructive)
    if cfg:
        try:
            alg_cfg = cfg.get("algorithm", {})
            if "max_iterations" in alg_cfg:
                solver.max_iterations = int(alg_cfg["max_iterations"])
            if "seed" in alg_cfg:
                solver.seed = int(alg_cfg["seed"])
            if "temperature" in alg_cfg:
                solver.temperature = float(alg_cfg["temperature"])
            if "cooling_rate" in alg_cfg:
                solver.cooling_rate = float(alg_cfg["cooling_rate"])
            if "learning_rate" in alg_cfg:
                solver.learning_rate = float(alg_cfg["learning_rate"])
            if "adaptive_period" in alg_cfg:
                solver.adaptive_period = int(alg_cfg["adaptive_period"])
        except Exception:
            logger.debug("Some algorithm config keys could not be applied to solver.")

    if iterations is not None:
        solver.max_iterations = int(iterations)

    logger.info(f"Running ALNS for up to {solver.max_iterations} iterations...")
    t0 = time.time()
    solution = solver.run(max_iterations=solver.max_iterations)
    elapsed = time.time() - t0
    logger.info(
        f"ALNS finished in {elapsed:.3f}s — cost: {getattr(solution, 'total_cost', float('nan'))}"
    )

    # Save solution if requested
    if save_results:
        if save_solution_to_file is None:
            logger.warning(
                "save_solution_to_file utility not available; cannot save solution."
            )
        else:
            os.makedirs(results_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(results_dir, f"solution_{ts}.json")
            try:
                # Some save helpers accept (solution, problem, filename) others only (solution, filename)
                # Try both patterns gracefully.
                try:
                    save_solution_to_file(solution, out_path)
                except TypeError:
                    # maybe expects (solution, problem, filename)
                    save_solution_to_file(
                        solution, getattr(solution, "problem", None), out_path
                    )  # type: ignore
                logger.info(f"Solution saved to {out_path}")
            except Exception as e:
                logger.warning(f"Failed to save solution to {out_path}: {e}")

    return solution


def run_construction(
    problem: Any,
    strategy_name: Optional[str] = None,
    multi_start: int = 1,
    save_results: bool = False,
    results_dir: str = "outputs",
) -> Any:
    """
    Run enhanced construction heuristics and return the best constructed Solution.
    """
    EnhancedConstructionHeuristics = _MODULES.get("EnhancedConstructionHeuristics")
    ConstructionStrategy = _MODULES.get("ConstructionStrategy")
    save_solution_to_file = _MODULES.get("save_solution_to_file")

    if EnhancedConstructionHeuristics is None or ConstructionStrategy is None:
        raise RuntimeError("Enhanced construction module not available for import.")

    heur = EnhancedConstructionHeuristics(problem)

    strategy = None
    if strategy_name:
        # Accept either enum members or string names
        try:
            strategy = ConstructionStrategy(strategy_name)
        except Exception:
            normalized = strategy_name.strip().lower().replace("-", "_")
            for s in ConstructionStrategy:
                if s.value == normalized or s.name.lower() == normalized:
                    strategy = s
                    break

    if multi_start <= 1:
        result = heur.construct_solution(strategy)
    else:
        result = heur.multi_start_construction(num_starts=multi_start)

    logger.info(
        f"Constructed solution: cost={getattr(result.solution, 'total_cost', float('nan'))}, strategy={getattr(result, 'strategy', 'unknown')}"
    )

    if save_results and save_solution_to_file:
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"constructed_{ts}.json")
        try:
            save_solution_to_file(result.solution, out_path)
            logger.info(f"Constructed solution saved to {out_path}")
        except Exception as e:
            logger.warning(f"Failed to save constructed solution: {e}")

    return result


def run_benchmark(
    algorithms: List[str], runs: int = 1, out_file: Optional[str] = None, seed: int = 42
):
    """
    Run benchmarking suite using the project's BenchmarkingFramework.
    """
    BenchmarkingFramework = _MODULES.get("BenchmarkingFramework")
    AlgorithmType = _MODULES.get("AlgorithmType")
    if BenchmarkingFramework is None or AlgorithmType is None:
        raise RuntimeError("Benchmarking module not available for import.")

    framework = BenchmarkingFramework(seed=seed)
    framework.create_standard_benchmarks()

    alg_types = []
    for a in algorithms:
        norm = a.strip().lower()
        mapping = {
            "alns": AlgorithmType.ALNS,
            "nearest_neighbor": AlgorithmType.NEAREST_NEIGHBOR,
            "nearest": AlgorithmType.NEAREST_NEIGHBOR,
            "savings": AlgorithmType.SAVINGS,
            "random": AlgorithmType.RANDOM,
        }
        if norm in mapping:
            alg_types.append(mapping[norm])
        else:
            logger.warning(f"Unknown algorithm '{a}' — skipping.")

    if not alg_types:
        alg_types = [
            AlgorithmType.ALNS,
            AlgorithmType.NEAREST_NEIGHBOR,
            AlgorithmType.SAVINGS,
        ]

    results = framework.run_benchmark_suite(alg_types, iterations_per_algorithm=runs)

    if out_file:
        try:
            report = framework.generate_benchmark_report(results)
            os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
            with open(out_file, "w") as f:
                f.write(report)
            logger.info(f"Benchmark report written to {out_file}")
        except Exception as e:
            logger.warning(f"Unable to write benchmark report to {out_file}: {e}")

    return results


def validate_solution_file(
    solution_file: str, report_file: Optional[str] = None, benchmark: bool = False
):
    """
    Validate a saved solution JSON file and optionally write a textual report.
    """
    load_solution_from_file = _MODULES.get("load_solution_from_file")
    EnhancedSolutionValidator = _MODULES.get("EnhancedSolutionValidator")

    if load_solution_from_file is None or EnhancedSolutionValidator is None:
        raise RuntimeError("Validation utilities not available for import.")

    logger.info(f"Loading solution from {solution_file}")
    solution = load_solution_from_file(solution_file)
    if getattr(solution, "problem", None) is None:
        logger.debug("Loaded solution missing embedded problem reference (if any).")

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
            logger.info(f"Validation report written to {report_file}")
        except Exception as e:
            logger.warning(f"Failed to write validation report: {e}")
    else:
        print(report_text)

    return result


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="enhanced_cli.py",
        description="Enhanced CLI for OT Waste Collection (config/template aware)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # Solve
    solve_p = sub.add_parser(
        "solve", help="Run ALNS solver on a generated or provided instance"
    )
    solve_p.add_argument(
        "--config",
        help="Path to JSON config file or template name from config_templates",
    )
    solve_p.add_argument("--instance", help="Path to instance JSON file (optional)")
    solve_p.add_argument("--iterations", type=int, help="Override iterations")
    solve_p.add_argument(
        "--save-results", action="store_true", help="Save solution to outputs/"
    )
    solve_p.add_argument(
        "--results-dir", default="outputs", help="Directory to save outputs"
    )

    # Construct
    cs_p = sub.add_parser(
        "construct", help="Construct initial solution(s) using enhanced heuristics"
    )
    cs_p.add_argument(
        "--strategy", help="Construction strategy (e.g., cluster_based, greedy_nearest)"
    )
    cs_p.add_argument(
        "--multi-start", type=int, default=1, help="Number of multi-start attempts"
    )
    cs_p.add_argument(
        "--save-results", action="store_true", help="Save constructed solution"
    )
    cs_p.add_argument(
        "--results-dir", default="outputs", help="Directory to save outputs"
    )
    cs_p.add_argument("--instance", help="Path to instance JSON file (optional)")

    # Benchmark
    bench_p = sub.add_parser("benchmark", help="Run benchmarking suite")
    bench_p.add_argument(
        "--algorithms",
        nargs="+",
        default=["alns", "nearest_neighbor", "savings"],
        help="Algorithms to benchmark",
    )
    bench_p.add_argument("--runs", type=int, default=1, help="Runs per algorithm")
    bench_p.add_argument("--out", help="Output file for benchmark report (text)")

    # Validate
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

    # Quick import validation with helpful message
    if _MODULES.get("ALNS") is None or _MODULES.get("DataGenerator") is None:
        logger.error("Required project modules are not available for import.")
        logger.error(
            "Make sure you run this script from the project root, or set PYTHONPATH so the `src` package is importable."
        )
        logger.error("Example (from repo root):")
        logger.error(
            "  PYTHONPATH=ot-waste-collection-if/ot-waste-collection-if python3 enhanced_cli.py --help"
        )
        logger.debug(f"Import diagnostic: {_MODULES.get('import_error')}")
        return 2

    seed = getattr(args, "seed", 42)

    try:
        if args.command == "solve":
            cfg = resolve_config(args.config) if getattr(args, "config", None) else None

            # Load or generate problem
            if args.instance:
                try:
                    problem = _MODULES["DataGenerator"].load_instance_from_file(
                        args.instance
                    )  # type: ignore
                except Exception as e:
                    logger.error(f"Failed to load instance from {args.instance}: {e}")
                    return 1
            else:
                problem = _MODULES["DataGenerator"].generate_instance(
                    name="cli_demo",
                    n_customers=cfg.get("problem", {}).get("n_customers", 15)
                    if cfg
                    else 15,
                    n_ifs=cfg.get("problem", {}).get("n_ifs", 2) if cfg else 2,
                    vehicle_capacity=cfg.get("problem", {}).get("vehicle_capacity", 25)
                    if cfg
                    else 25,
                    seed=seed,
                )

            solution = run_solver_on_instance(
                problem,
                cfg=cfg,
                iterations=getattr(args, "iterations", None),
                save_results=getattr(args, "save_results", False),
                results_dir=getattr(args, "results_dir", "outputs"),
            )
            try:
                print(f"Solution cost: {solution.total_cost:.2f}")
                print(f"Routes: {len(solution.routes)}")
            except Exception:
                print(
                    "Solver completed — enable --verbose for more details or inspect saved outputs."
                )

        elif args.command == "construct":
            if args.instance:
                try:
                    problem = _MODULES["DataGenerator"].load_instance_from_file(
                        args.instance
                    )  # type: ignore
                except Exception as e:
                    logger.error(f"Failed to load instance from {args.instance}: {e}")
                    return 1
            else:
                problem = _MODULES["DataGenerator"].generate_instance(
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
            print(
                f"Constructed solution cost: {getattr(result.solution, 'total_cost', float('nan')):.2f}"
            )
            print(f"Strategy: {getattr(result, 'strategy', 'unknown')}")

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
                solution_file=getattr(args, "solution_file"),
                report_file=getattr(args, "report_file", None),
                benchmark=getattr(args, "benchmark", False),
            )
            print(f"Validation status: {res.status.value}")

        else:
            parser.print_help()

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
