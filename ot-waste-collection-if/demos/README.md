/mnt/windows-d/TY LABWORKS/OT LAB/project/ot-waste-collection-if/ot-waste-collection-if/demos/README.md#L1-260
# Demos — VRP-IF with ALNS (Complete Start-to-Finish Guide)

This directory contains demonstration scripts and documentation for the ALNS solver for the Vehicle Routing Problem with Intermediate Facilities (VRP-IF). This README provides complete start-to-finish guidance so someone new to the repository can:

- Set up the environment
- Generate or load instances
- Run the solver through the `main.py` entrypoint or demo scripts
- Save and inspect outputs (JSON, plots)
- Run tests and troubleshoot common problems
- Extend or contribute to the demos

Location
- Demo runner: `demos/comprehensive_demo.py`
- Demo outputs: `demos/outputs/` (created at runtime)
- CLI entrypoint (project root): `main.py`
- Requirements: `requirements.txt` in the project root

Overview
- The project implements an Adaptive Large Neighborhood Search (ALNS) framework for municipal waste collection routing where vehicles may visit intermediate facilities (IFs) to unload.
- Demos generate synthetic instances, solve them using ALNS, produce JSON solutions, and optionally generate route & convergence plots.

Prerequisites
- Python 3.8+ (3.10+ recommended)
- pip
- Optional: virtualenv or venv for an isolated environment

Quick setup
1. Create and activate a virtual environment (recommended):

```/dev/null/commands.sh#L1-3
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```/dev/null/commands.sh#L4-6
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If you cannot open graphical windows on the host (headless server), set the matplotlib backend before running demos:

```/dev/null/commands.sh#L7-9
export MPLBACKEND=Agg   # Linux/macOS (bash)
# Windows (PowerShell): $env:MPLBACKEND = 'Agg'
```

Running demos — quick commands

- Run the packaged comprehensive demo script (recommended for reproducible demo):

```/dev/null/commands.sh#L10-14
python3 demos/comprehensive_demo.py --customers 20 --ifs 2 --capacity 30 --iterations 500 --save-plots
```

- Use the `main.py` CLI entrypoint to run various modes:

```/dev/null/commands.sh#L15-20
python3 main.py --demo comprehensive --save-plots --save-results
python3 main.py --demo basic
python3 main.py --demo benchmark
```

- Solve from an existing instance JSON file:

```/dev/null/commands.sh#L21-24
python3 main.py --instance path/to/instance.json --iterations 300 --save-results
```

- Create a configuration template (if supported):

```/dev/null/commands.sh#L25-27
python3 main.py --create-config
```

CLI options (important ones)
- `--demo {basic,comprehensive,benchmark}` — run demonstration modes
- `--instance <file>` — load and solve an instance stored as JSON
- `--iterations N` — number of ALNS iterations (default: 200)
- `--live` — enable live plotting (interactive; requires a display)
- `--save-plots` — save route and convergence plots as PNGs
- `--save-results` — save solution JSON to disk
- `--create-config` — produce a configuration template JSON
- `--verbose` — print more output and solver internals

What the demo produces
- JSON solution file: `demos/outputs/solution_<timestamp>.json`
  - Contains routes, node sequences, loads, distances, and run metadata.
- Route plot: `demos/outputs/routes_<timestamp>.png`
  - Visual map of routes, depot, customers, and intermediate facilities.
- Convergence plot: `demos/outputs/convergence_<timestamp>.png`
  - Solver objective (cost) vs. iteration history.
- Console output: summary with total cost, distance/time, vehicles used, and route-level details.

Recommended run flow (start-to-finish)
1. Install dependencies and set MPL backend if headless.
2. Run a small demonstration to verify everything works:

```/dev/null/commands.sh#L28-32
python3 main.py --demo basic --iterations 100 --save-plots --save-results --verbose
```

3. Inspect outputs:
   - Open the generated JSON to view routes and metadata.
   - Open route & convergence PNGs (if generated).
4. Run a larger comprehensive demo:

```/dev/null/commands.sh#L33-37
python3 main.py --demo comprehensive --iterations 500 --save-plots --save-results
```

5. For benchmarking, use the `benchmark` demo mode to run multiple instance sizes and collect summary statistics.

Understanding outputs (JSON solution)
- The solution JSON contains:
  - `routes`: list of routes; each route is a sequence of node IDs (depots, customers, IFs).
  - `total_cost`, `total_distance`, `total_time`
  - `unassigned_customers` (should be empty for feasible instances)
  - `run_metadata`: seed, iterations, timestamp, runtime
- Use `demos/outputs` to store and organize runs by timestamp.

Testing the codebase
- Run unit tests (recommended to run from repository root or adjust PYTHONPATH):

Option A — change directory into package and run tests:

```/dev/null/commands.sh#L38-40
cd ot-waste-collection-if/ot-waste-collection-if
python3 -m unittest discover -s tests -p "test_*.py"
```

Option B — use PYTHONPATH to ensure `src` can be imported from tests (run from repo root):

```/dev/null/commands.sh#L41-44
PYTHONPATH=ot-waste-collection-if/ot-waste-collection-if python3 -m unittest discover \
    -s ot-waste-collection-if/ot-waste-collection-if/tests -p "test_*.py"
```

- Note: Tests exercise core modules: `problem`, `solution`, destroy/repair operators, and ALNS integration. If a test fails, examine stack traces and start with small `iterations` to reproduce faster.

Troubleshooting / common issues
- "Matplotlib backend error / plots don't appear"
  - Use `export MPLBACKEND=Agg` (or equivalent for your shell) to run headless plotting.
- "Customer demand exceeds vehicle capacity"
  - This means generated customer demand > vehicle capacity. Increase `--capacity` or adjust `demand_range` (if using generator).
- "Unassigned customers remain"
  - Instance may be infeasible with current vehicle count/capacity. Check `problem.is_feasible()` output. Increase `--iterations` or adjust instance parameters.
- "ImportError in tests"
  - Ensure you run tests from the correct working directory or set `PYTHONPATH` so test runner finds `src` modules.
- "Live plotting errors"
  - Live plotting requires a display (X11/Wayland) and `matplotlib` interactive backend. On headless servers, avoid `--live`.

Project structure (important files)
- `main.py` — enhanced CLI entrypoint (run demos, load instances, save results)
- `src/`
  - `problem.py` — ProblemInstance and Location definitions, distance/time utilities
  - `solution.py` — Route and Solution classes; metric calculations and feasibility checks
  - `alns.py` — ALNS algorithm implementation and orchestration
  - `destroy_operators.py` / `repair_operators.py` — operator implementations and managers
  - `utils.py` — visualization and performance analysis helpers
- `demos/` — demo drivers and example scripts
  - `demos/comprehensive_demo.py` — a script that builds an instance, runs solver, and saves outputs
  - `demos/outputs/` — demo artifacts generated at runtime
- `tests/` — unit tests (run as shown earlier)
- `requirements.txt` — dependency list

Tips for reproducible experiments
- Always set `--seed` (if available in demo scripts) to reproduce random instance and operator behavior.
- Save both JSON and plots for each run. Include the `run_metadata` (seed & parameters) in the JSON.
- For CI: run the unit tests with a small iteration count to validate functionality quickly.

Extending demos or adding notebooks
- Add a Jupyter notebook that:
  - Generates a medium sized instance
  - Runs `ALNS` with a fixed seed
  - Embeds route & convergence figures inline
  - Saves the solution JSON and prints analysis summary
- Put notebooks under `notebooks/` and include a short README describing reproduction steps.

Contributing
- Fork the repository, create a feature branch, and open a pull request with:
  - Clear description of changes
  - New or updated tests for new behavior
  - Updated demo or README instructions if behavior changed
- Keep commits focused and tests passing locally before PR.

Acceptance criteria for a successful demo
- No unhandled exceptions during the run
- All customers assigned for feasible instances
- No capacity violations in any route
- IF visits present where appropriate (when a vehicle exceeds capacity mid-route)
- Convergence plot shows objective improving (typically faster in early iterations)
- Re-running with the same seed reproduces the result

Example complete run (short)
1. Create venv and install requirements.
2. Run a small comprehensive demo and save plots:

```/dev/null/commands.sh#L45-49
python3 main.py --demo comprehensive --iterations 200 --save-plots --save-results --verbose
```

3. Inspect `demos/outputs/solution_<timestamp>.json`, `routes_<timestamp>.png`, and `convergence_<timestamp>.png`.

Contact / support
- For questions about design or help reproducing results, open an issue in the repo with:
  - OS and Python version
  - Command used
  - Smallest reproducible example (if possible)
  - Any traceback logs

Final notes
- This README aims to be a comprehensive start-to-finish guide for running demos and working with this project. If you would like, we can:
  - Add a short Jupyter notebook demonstrating a medium-sized experiment end-to-end.
  - Add a GitHub Actions workflow to run tests and generate a demo artifact on each push.
  - Produce a short slide deck documenting the problem, approach, and demo steps for presentations.