"""Alternative solutions: compare the chosen solver with credible alternatives.

Runs the method comparison in ``quarter_car/compare.py`` on three set cases
with the default car: a smooth speed hump, and a pothole (where the wheel
leaves the road) at two speeds. Each method is measured on accuracy,
right-hand-side evaluations, run time (fastest of 15 runs) and peak memory.
Run time and memory are measured on the computer that runs the script, so
they will differ between computers. The program (``main.py``) runs the same
comparison on the user's own inputs.

Run from the project folder (takes a few minutes)::

    python alternatives_study.py

The table and figure are saved in the ``results`` folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from quarter_car.compare import compare_methods, plot_comparison
from quarter_car.model import QuarterCar, damping_from_ratio
from quarter_car.road import Bump, Pothole

RESULTS = Path(__file__).resolve().parent / "results"
CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=damping_from_ratio(0.3, 20e3, 300.0), Kw=200e3)
CASES = [
    ("Speed hump, 75 mm x 3.7 m, 30 km/h", Bump(0.075, 3.7), 30 / 3.6),
    ("Pothole, 50 mm x 0.6 m, 40 km/h", Pothole(0.05, 0.6), 40 / 3.6),
    ("Pothole, 50 mm x 0.6 m, 80 km/h", Pothole(0.05, 0.6), 80 / 3.6),
]
REPEATS = 15                       # timed runs per method; the fastest is reported


def save_tables(results):
    """Save all results as a Markdown table and a CSV file, and print the table."""
    lines = ["| Case | Method | Setting | Error | RHS evaluations | Run time [ms] | Memory [kB] |",
             "|---|---|---|---|---|---|---|"]
    csv = ["case,method,setting,error_percent,rhs_evaluations,run_time_ms,memory_kB"]
    for case, rows in results.items():
        for row in rows:
            evals = "-" if row["evals"] is None else str(row["evals"])
            lines.append(f"| {case} | {row['method']} | {row['setting']} | {100 * row['error']:.2f}% | "
                         f"{evals} | {1e3 * row['seconds']:.1f} | {row['memory'] / 1024:.0f} |")
            csv.append(f"{case},{row['method']},{row['setting']},{100 * row['error']:.4f},"
                       f"{'' if row['evals'] is None else row['evals']},{1e3 * row['seconds']:.2f},"
                       f"{row['memory'] / 1024:.1f}")
    table = "\n".join(lines)
    (RESULTS / "alternatives.md").write_text(
        "Each method compared with the reference solution. Error is the largest error in body "
        "acceleration and tyre force, as a percentage of their peaks.\n\n" + table + "\n")
    (RESULTS / "alternatives.csv").write_text("\n".join(csv) + "\n")
    print(table)


def save_figure(results):
    """Run-time bar charts for the speed hump and the 40 km/h pothole."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, case in zip(axes, list(results)[:2]):
        plot_comparison(results[case], ax, case)
    fig.suptitle("Run time of each method (fastest of 15 runs), at its 1% setting where it has one",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "alternatives_run_time.png", dpi=200)
    plt.close(fig)


def main():
    """Run the comparison for every case and save the table and figure."""
    RESULTS.mkdir(exist_ok=True)
    results = {}
    for case, road, speed in CASES:
        results[case] = compare_methods(CAR, road, speed, repeats=REPEATS)
        print(f"  {case}: done")
    save_tables(results)
    save_figure(results)
    print(f"\nSaved the table and figure in {RESULTS}")


if __name__ == "__main__":
    main()
