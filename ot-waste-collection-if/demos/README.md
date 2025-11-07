# Demos — VRP-IF with ALNS

This directory contains scripts and instructions to run reproducible demonstrations of the ALNS solver for the Vehicle Routing Problem with Intermediate Facilities (VRP-IF). The demo scripts produce:

- A synthetic problem instance (configurable).
- A solved route plan (JSON).
- Route and convergence plots (PNG).
- A short performance summary printed to stdout.

Location:
- Demo script: `demos/comprehensive_demo.py`
- Demo outputs: `demos/outputs/` (created at runtime)
- Requirements: `requirements.txt` in the project root

---

Quick prerequisites
- Python 3.8+ recommended (3.10+ preferred)
- Install dependencies (inside virtualenv is recommended):

```/dev/null/commands.sh#L1-3
pip install -r requirements.txt
```

If you cannot open graphical windows on the host (headless server), set the matplotlib backend before running the demo:

```/dev/null/commands.sh#L4-6
export MPLBACKEND=Agg   # Linux/macOS (bash)
set MPLBACKEND=Agg      # Windows (PowerShell: $env:MPLBACKEND = 'Agg')
```

---

How to run the comprehensive demo

From repository root, run the demo script:

```/dev/null/commands.sh#L7-12
python3 demos/comprehensive_demo.py --customers 20 --ifs 2 --capacity 30 --iterations 500 --save-plots
```

Arguments:
- `--customers`: number of customers to generate (default 20)
- `--ifs`: number of intermediate facilities (default 2)
- `--capacity`: vehicle capacity used for the instance
- `--iterations`: number of ALNS iterations
- `--save-plots`: save route & convergence plots to `demos/outputs/`
- `--output-dir`: customize output directory

You can also use the enhanced CLI entrypoint:

```/dev/null/commands.sh#L13-16
python3 main.py --demo comprehensive --save-plots --save-results
```

`main.py` exposes additional demonstration and benchmarking modes (see `--demo` option).

---

What the demo produces
- JSON solution file: `demos/outputs/solution_<timestamp>.json`
  - Contains routes, node sequences, loads, distances and run metadata.
- Route plot: `demos/outputs/routes_<timestamp>.png`
  - Visual map of routes, depot, customers and intermediate facilities.
- Convergence plot: `demos/outputs/convergence_<timestamp>.png`
  - Solver cost vs iteration history.
- Console output with a brief performance summary and detailed route metrics.

---

What to show in a live demo or recorded video
1. Run a demo (small or medium instance). Show the console output:
   - Total cost, total distance/time.
   - Number of routes and unassigned customers (should be 0 for feasible instances).
2. Open the route plot and point out:
   - Depot marker.
   - IF (Intermediate Facility) markers — where vehicles stop to empty.
   - A route where the vehicle visits an IF during its itinerary — emphasize load reset after IF.
3. Open the convergence plot:
   - Show how solution cost evolves.
   - Highlight the iteration where the best solution was found.
4. Open and inspect the solution JSON (or print one route):
   - Show node IDs and sequence (depot / customers / IF / depot).
   - Verify loads per route (ensure no capacity violations).
5. (Optional) Run a benchmark set and show a short table of cost/time/vehicles for several instances.

---

Acceptance criteria to claim success
- Functional:
  - All customers are assigned in demo instances (unassigned customers == 0).
  - No capacity violations in any route (per-route max load ≤ vehicle capacity).
  - IF visits appear where needed (routes that would exceed capacity have IFs).
- Algorithmic:
  - Convergence plot shows improvement (typically the cost reduces rapidly in early iterations).
  - Best solution is found before the final iteration (indicates effective operator behavior).
- Reproducibility:
  - Using the same `--seed` yields identical results for the same configuration (documented nondeterminism if randomized operators still vary).
- Robustness:
  - Demo runs without errors on the target machine with required packages installed.

---

Test the codebase
- Run the test suite to validate core components:

```/dev/null/commands.sh#L17-19
python3 -m unittest discover -s tests -p "test_*.py"
```

All tests should pass for core functions like create-instance, route feasibility, destroy/repair operator basic behavior, and a short ALNS run.

---

Troubleshooting / common issues
- Plotting fails / no display:
  - Use a headless backend: `export MPLBACKEND=Agg` (Linux/macOS) before running.
- “Customer demand exceeds vehicle capacity”:
  - This means the generated customer demand exceeded the vehicle capacity. Increase `capacity` or reduce `demand_range`.
- Long runs:
  - Reduce `--iterations` for quick demos (e.g., 100-300).
- Missing dependencies:
  - Ensure `numpy` and `matplotlib` are installed. Use the provided `requirements.txt`.
- Unassigned customers remain:
  - Try increasing `--iterations` or review the instance for feasibility (problem.is_feasible() check).

---

Recommended demonstration script / flow (5–8 minutes)
1. Show the project README briefly (one slide describing the problem).
2. Run a quick demo: small instance with plots saved (30–60s).
3. Display the route PNG and point out IF visits and routes.
4. Display the convergence PNG and mention adaptive operator behavior.
5. Open solution JSON to show route sequences and loads.
6. Show short automated test run output confirming correctness.

---

Files in `demos/`
- `comprehensive_demo.py` — main demo runner (configurable via CLI)
- `outputs/` — demo outputs (created at runtime)
- (You can add additional scripts, e.g. `demo_quick.py` for a 1-minute smoke test.)

---

Next recommended steps (if you want to finalize the project)
- Add a short Jupyter notebook that runs a medium-size instance and embeds the plots and analysis summary for reproducible experiments.
- Add a short slideshow or recorded walkthrough that follows the demo script above (helpful for presentations).
- Add a CI job to run the tests and produce the demo artifacts (plots + JSON) on each push.

---

If you'd like, I can:
- Create a short presentation outline (slide-by-slide).
- Add a small Jupyter notebook that runs the demo and embeds plots and metrics.
- Add a GitHub Actions workflow that runs tests and generates demo artifacts.

Tell me which addition you want next and I will prepare it.