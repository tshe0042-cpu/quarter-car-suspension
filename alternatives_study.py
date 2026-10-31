"""Alternative solutions: compare the chosen solver with credible alternatives.

Seven ways of solving the same run are compared on a smooth speed hump and on
a pothole, where the wheel leaves the road:

* Euler with a fixed step (the simplest method, notebook 7.3)
* RK4 with a fixed step (notebook 7.4)
* RK4 restarted at the road edges (the method the program uses)
* SciPy's adaptive RK45, at its default and at a tight tolerance
* SciPy's adaptive RK45 restarted at the road edges
* the exact solution of the linear model with ``scipy.signal.lsim``

Each fixed-step method uses the largest time step that follows the motion
within 1% (found as in ``time_step_study.py``). Every method is then measured
on accuracy against the reference solution, right-hand-side evaluations, run
time (fastest of 15 runs) and peak memory. Run time and memory are measured on
the computer that runs the script, so they will differ between computers.

Run from the project folder (takes a few minutes)::

    python alternatives_study.py

The table and figure are saved in the ``results`` folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quarter_car.compare import linear_solve, measure
from quarter_car.model import QuarterCar, damping_from_ratio
from quarter_car.road import Bump, Pothole
from quarter_car.scenarios import simulate_road
from quarter_car.solvers import euler_step, rk4_step
from quarter_car.timestep import (STEPS, adaptive_solve, reference_solution, required_step,
                                  run_errors, step_study)

RESULTS = Path(__file__).resolve().parent / "results"
CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=damping_from_ratio(0.3, 20e3, 300.0), Kw=200e3)
CASES = [
    ("Speed hump, 75 mm x 3.7 m, 30 km/h", Bump(0.075, 3.7), 30 / 3.6),
    ("Pothole, 50 mm x 0.6 m, 40 km/h", Pothole(0.05, 0.6), 40 / 3.6),
    ("Pothole, 50 mm x 0.6 m, 80 km/h", Pothole(0.05, 0.6), 80 / 3.6),
]
CHOSEN = "RK4, restarted at edges"
TOL = 0.01
REPEATS = 15                            # timed runs per method; the fastest is reported
EULER_STEPS = tuple(h for h in STEPS + (5e-5, 2e-5) if h < 7e-3)   # Euler is unstable above 7 ms

# Chart styling: the chosen method is highlighted and the others are grey.
HIGHLIGHT, OTHER, INK, SECONDARY, GRID, WARNING = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b"


def fixed_step_run(step, split, steps):
    """Return a method that finds its 1% time step, then times one run at that step."""
    def run(road, speed, reference):
        h, _ = required_step(step_study(CAR, road, speed, steps, step, split, reference), TOL)
        (t, Z, _), seconds, memory = measure(
            lambda: simulate_road(CAR, road, speed, h=h, step=step, split_at_edges=split), REPEATS)
        evals = (len(t) - 1) * (1 if step is euler_step else 4)
        return f"h = {1e3 * h:g} ms", t, Z, evals, seconds, memory
    return run


def adaptive_run(rtol, split):
    """Return a method that times SciPy's adaptive RK45 at the given tolerance."""
    def run(road, speed, reference):
        result, seconds, memory = measure(
            lambda: adaptive_solve(CAR, road, speed, rtol=rtol, atol=rtol * 1e-3, split_at_edges=split),
            REPEATS)
        t = np.arange(0.0, result.t[-1], 1e-4)       # judged every 0.1 ms using the dense output
        return f"rtol = {rtol:g}", t, result.states(t), result.n_evals, seconds, memory
    return run


def linear_run(road, speed, reference):
    """Time the exact solution of the linear model (no lift-off)."""
    (t, Z), seconds, memory = measure(lambda: linear_solve(CAR, road, speed, h=1e-3), REPEATS)
    return "h = 1 ms", t, Z, None, seconds, memory


METHODS = {
    "Euler, fixed step": fixed_step_run(euler_step, False, EULER_STEPS),
    "RK4, fixed step": fixed_step_run(rk4_step, False, STEPS),
    CHOSEN: fixed_step_run(rk4_step, True, STEPS),
    "RK45 adaptive": adaptive_run(1e-3, False),
    "RK45 adaptive, tight tolerance": adaptive_run(1e-6, False),
    "RK45 adaptive, restarted at edges": adaptive_run(1e-3, True),
    "Linear model, exact (lsim)": linear_run,
}


def run_comparison():
    """Run every method on every case and return the results as a list of dicts."""
    rows = []
    for case, road, speed in CASES:
        reference = reference_solution(CAR, road, speed)
        for method, run in METHODS.items():
            setting, t, Z, evals, seconds, memory = run(road, speed, reference)
            error, _ = run_errors(CAR, t, Z, reference)
            rows.append({"case": case, "method": method, "setting": setting, "error": error,
                         "evals": evals, "seconds": seconds, "memory": memory})
        print(f"  {case}: done")
    return rows


def save_tables(rows):
    """Save the results as a Markdown table and a CSV file, and print the table."""
    lines = ["| Case | Method | Setting | Error | RHS evaluations | Run time [ms] | Memory [kB] |",
             "|---|---|---|---|---|---|---|"]
    csv = ["case,method,setting,error_percent,rhs_evaluations,run_time_ms,memory_kB"]
    for row in rows:
        evals = "-" if row["evals"] is None else str(row["evals"])
        lines.append(f"| {row['case']} | {row['method']} | {row['setting']} | {100 * row['error']:.2f}% | "
                     f"{evals} | {1e3 * row['seconds']:.1f} | {row['memory'] / 1024:.0f} |")
        csv.append(f"{row['case']},{row['method']},{row['setting']},{100 * row['error']:.4f},"
                   f"{'' if row['evals'] is None else row['evals']},{1e3 * row['seconds']:.2f},"
                   f"{row['memory'] / 1024:.1f}")
    table = "\n".join(lines)
    (RESULTS / "alternatives.md").write_text(
        "Each method compared with the reference solution. Error is the largest error in body "
        "acceleration and tyre force, as a percentage of their peaks.\n\n" + table + "\n")
    (RESULTS / "alternatives.csv").write_text("\n".join(csv) + "\n")
    print(table)


def save_figure(rows):
    """Bar chart of run time for each method, with its error written on each bar."""
    cases = [case for case, _, _ in CASES[:2]]          # the speed hump and the 40 km/h pothole
    methods = list(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, case in zip(axes, cases):
        case_rows = {row["method"]: row for row in rows if row["case"] == case}
        times = [1e3 * case_rows[m]["seconds"] for m in methods]
        colours = [HIGHLIGHT if m == CHOSEN else OTHER for m in methods]
        ax.barh(methods, times, color=colours, height=0.6)
        for i, m in enumerate(methods):
            error = case_rows[m]["error"]
            passed = error <= TOL
            ax.text(times[i] * 1.08, i, f"{times[i]:.0f} ms, error {100 * error:.2g}%"
                    + ("" if passed else " (misses 1%)"),
                    va="center", fontsize=8, color=INK if passed else WARNING)
        ax.set_xscale("log")
        ax.set_xlim(1, max(times) * 12)
        ax.set_title(case, fontsize=10, color=INK)
        ax.set_xlabel("Run time [ms], log scale")
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.tick_params(colors=SECONDARY, labelsize=8)
    axes[0].invert_yaxis()                               # first method at the top
    fig.suptitle("Run time of each method (fastest of 15 runs), at its 1% setting where it has one",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "alternatives_run_time.png", dpi=200)
    plt.close(fig)


def main():
    """Run the comparison and save the table and figure."""
    RESULTS.mkdir(exist_ok=True)
    print("Running every method on every case...")
    rows = run_comparison()
    save_tables(rows)
    save_figure(rows)
    print(f"\nSaved the table and figure in {RESULTS}")


if __name__ == "__main__":
    main()
