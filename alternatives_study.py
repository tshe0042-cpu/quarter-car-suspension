"""Alternative solutions: compare the chosen solver with credible alternatives.

Runs the method comparison in ``quarter_car/compare.py`` on four set cases:
a smooth speed hump and a pothole (where the wheel leaves the road) at two
speeds, all with the default car, and a sharp bump at the top speed with a
soft, lightly damped car. Each method is measured on accuracy,
right-hand-side evaluations, run time (fastest of 15 runs) and peak memory.
Run time and memory are measured on the computer that runs the script, so
they will differ between computers. The program (``main.py``) runs the same
comparison on the user's own inputs.

Saves these results in the ``results`` folder:

* alternatives.md: every method's setting, error, work, run time and memory.
* alternatives_error_and_cost.png: the error and work of every method on
  every case.

Run from the project folder (takes a few minutes)::

    python alternatives_study.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from quarter_car.compare import CHOSEN, TOL, compare_methods
from quarter_car.model import QuarterCar, damping_from_ratio
from quarter_car.road import Bump, Pothole

RESULTS = Path(__file__).resolve().parent / "results"
CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=damping_from_ratio(0.3, 20e3, 300.0), Kw=200e3)
SOFT_CAR = QuarterCar(m1=40.0, m2=150.0, Ks=5e3, b=damping_from_ratio(0.05, 5e3, 150.0), Kw=100e3)
CASES = [   # (name, car, road, speed [m/s])
    ("Speed hump, 75 mm x 3.7 m, 30 km/h", CAR, Bump(0.075, 3.7), 30 / 3.6),
    ("Pothole, 50 mm x 0.6 m, 40 km/h", CAR, Pothole(0.05, 0.6), 40 / 3.6),
    ("Pothole, 50 mm x 0.6 m, 80 km/h", CAR, Pothole(0.05, 0.6), 80 / 3.6),
    ("Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car", SOFT_CAR, Bump(0.15, 0.3), 110 / 3.6),
]
REPEATS = 15                       # timed runs per method; the fastest is reported

# Chart styling for the error and cost figure.
SHAPES = ("o", "s", "^", "D")      # one marker shape per case
CHOSEN_COLOUR, OTHER_COLOUR, MISS_COLOUR = "#1baf7a", "#52514e", "#d03b3b"
CHOSEN_TINT, MISS_TINT = "#e4f4ed", "#fbefee"   # light backgrounds for the chosen row and the misses
INK, SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def save_tables(results):
    """Save all results as a Markdown table and print it."""
    lines = ["| Case | Method | Setting | Error | RHS evaluations | Run time [ms] | Memory [kB] |",
             "|---|---|---|---|---|---|---|"]
    for case, rows in results.items():
        for row in rows:
            evals = "-" if row["evals"] is None else str(row["evals"])
            lines.append(f"| {case} | {row['method']} | {row['setting']} | {100 * row['error']:.2f}% | "
                         f"{evals} | {1e3 * row['seconds']:.1f} | {row['memory'] / 1024:.0f} |")
    table = "\n".join(lines)
    (RESULTS / "alternatives.md").write_text(
        "Each method compared with the reference solution. Error is the largest error in body "
        "acceleration and tyre force, as a percentage of their peaks.\n\n" + table + "\n")
    print(table)


def save_error_and_cost_figure(results):
    """Error and work (right-hand-side evaluations) of every method on every case.

    Each method has one row, with one marker per case. The chosen method is
    highlighted. Errors to the right of the 1% line miss the accuracy target
    and are labelled with their value.
    """
    cases = list(results)
    methods = [row["method"] for row in results[cases[0]]]
    offsets = np.linspace(-0.27, 0.27, len(cases))     # each case on its own line within a row
    chosen = methods.index(CHOSEN)

    fig, (ax_error, ax_cost) = plt.subplots(1, 2, figsize=(8, 4.8), sharey=True)
    ax_error.axvspan(100 * TOL, 1e3, color=MISS_TINT, lw=0, zorder=0)
    for ax in (ax_error, ax_cost):         # the chosen method's band sits over the tint, under the grid
        ax.axhspan(chosen - 0.45, chosen + 0.45, color=CHOSEN_TINT, lw=0, zorder=0.1)
    ax_error.axvline(100 * TOL, color=MISS_COLOUR, lw=1)

    for offset, shape, case in zip(offsets, SHAPES, cases):
        for row in results[case]:
            y = methods.index(row["method"]) + offset
            marks = dict(color=CHOSEN_COLOUR if row["method"] == CHOSEN else OTHER_COLOUR,
                         ms=5.5, mec=SURFACE, mew=0.8)
            percent = 100 * row["error"]
            ax_error.plot(percent, y, shape, **marks)
            if row["error"] > TOL:                     # label every miss with its error
                text = f"{percent:.2g}%" if percent < 10 else f"{percent:.0f}%"
                ax_error.text(percent * 1.35, y, text, fontsize=6.5, color=INK, va="center")
            if row["evals"] is not None:
                ax_cost.plot(row["evals"], y, shape, **marks)
    ax_cost.text(30, methods.index("Linear model, exact (lsim)"), "Exact solution, no steps",
                 fontsize=7, color=SECONDARY, va="center")

    ax_error.set_xscale("log")
    ax_error.set_xlim(1e-3, 1e3)
    ax_error.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax_error.set_xlabel("Error compared with the reference [% of peak]")
    ax_error.set_title("Accuracy", fontsize=9, loc="left")
    ax_cost.set_xscale("log")
    ax_cost.set_xlim(20, 3e5)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_cost.set_xlabel("Right-hand-side evaluations")
    ax_cost.set_title("Work done", fontsize=9, loc="left")
    ax_error.set_ylim(len(methods) - 0.5, -0.5)               # first method at the top
    ax_error.set_yticks(range(len(methods)))
    ax_error.set_yticklabels([f"{m} (chosen)" if m == CHOSEN else m for m in methods])
    ax_error.get_yticklabels()[chosen].set_fontweight("bold")
    for ax in (ax_error, ax_cost):
        ax.grid(True, axis="x", which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.tick_params(colors=SECONDARY, labelsize=7.5)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.label.set_color(INK)
        ax.xaxis.label.set_fontsize(8)

    handles = [Line2D([], [], ls="none", marker=shape, color=OTHER_COLOUR, ms=5.5, mec=SURFACE, mew=0.8)
               for shape in SHAPES[:len(cases)]]
    handles += [Line2D([], [], color=MISS_COLOUR, lw=1), Patch(color=MISS_TINT, lw=0)]
    labels = cases + ["1% accuracy target", "Misses the target"]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle("Accuracy and work of each method on four test cases", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(RESULTS / "alternatives_error_and_cost.png", dpi=200)
    plt.close(fig)


def main():
    """Run the comparison for every case and save the table and figure."""
    RESULTS.mkdir(exist_ok=True)
    results = {}
    for case, car, road, speed in CASES:
        results[case] = compare_methods(car, road, speed, repeats=REPEATS)
        print(f"  {case}: done")
    save_tables(results)
    save_error_and_cost_figure(results)
    print(f"\nSaved the table and figure in {RESULTS}")


if __name__ == "__main__":
    main()
