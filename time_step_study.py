"""Time step study: how fine must the time step be to capture sudden road changes?

Runs every road surface in the program with the default car, compares each
run with an accurate reference solution (see ``quarter_car/timestep.py``)
and finds the time step needed for the solver to follow the true motion
within 1%. It also compares the cost of Euler, RK4, RK4 restarted at the road
edges, and SciPy's adaptive RK45, and checks the program's own time step
(including the peak values it reports) on every road and speed.

Run from the project folder (takes a few minutes)::

    python time_step_study.py

Figures and tables are saved in the ``results`` folder.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quarter_car.model import QuarterCar, damping_from_ratio, natural_modes
from quarter_car.road import Bump, Pothole
from quarter_car.scenarios import road_input, simulate_road
from quarter_car.solvers import euler_step, rk4_step
from main import TIME_STEP
from quarter_car.timestep import (STEPS, adaptive_solve, max_stable_step, reference_solution,
                                  required_step, run_errors, step_study)

RESULTS = Path(__file__).resolve().parent / "results"
CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=damping_from_ratio(0.3, 20e3, 300.0), Kw=200e3)
ROADS = {
    "Speed hump, 75 mm x 3.7 m": Bump(0.075, 3.7),
    "Speed bump, 75 mm x 0.5 m": Bump(0.075, 0.5),
    "Small pothole, 25 mm x 0.3 m": Pothole(0.025, 0.3),
    "Medium pothole, 50 mm x 0.6 m": Pothole(0.05, 0.6),
    "Large pothole, 75 mm x 1.0 m": Pothole(0.075, 1.0),
}
SPEEDS_KMH = (20, 40, 60, 80)
TOL = 0.01                              # accuracy target: 1% error
FIGURE_STEPS = STEPS + (5e-5,)          # one extra fine step for the convergence plot
METHODS = {                             # name: (step function, restart at edges?)
    "Euler": (euler_step, False),
    "RK4": (rk4_step, False),
    "RK4, restarted at edges": (rk4_step, True),
}

# Chart styling: one colour and marker per method, used the same way in every figure.
COLOURS = {"Euler": "#2a78d6", "RK4": "#eb6834", "RK4, restarted at edges": "#1baf7a"}
MARKERS = {"Euler": "o", "RK4": "s", "RK4, restarted at edges": "^"}
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def style(ax):
    """Give an axes hairline gridlines, quiet axis lines and dark text."""
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def observed_order(steps, errors, floor=1e-8):
    """Observed order p = log(E1 / E2) / log(h1 / h2) from the two finest steps.

    Steps with errors below ``floor`` are skipped, because there the error is
    set by the accuracy of the reference and round-off, not by the step.
    """
    usable = np.asarray(errors) > floor
    h, e = np.asarray(steps)[usable], np.asarray(errors)[usable]
    if len(h) < 2:
        return float("nan")
    return float(np.log(e[-2] / e[-1]) / np.log(h[-2] / h[-1]))


def print_context():
    """Print the time scales that set the time step for smooth roads."""
    freqs, _ = natural_modes(CAR)
    print(f"Wheel hop: {freqs[1]:.1f} Hz, period {1e3 / freqs[1]:.0f} ms")
    print(f"Largest stable step: Euler {1e3 * max_stable_step(CAR, 'euler'):.2f} ms, "
          f"RK4 {1e3 * max_stable_step(CAR, 'rk4'):.1f} ms")


def check_reference():
    """Check the reference solution against RK4 with a 5 microsecond step."""
    road, speed = ROADS["Medium pothole, 50 mm x 0.6 m"], 40 / 3.6
    reference = reference_solution(CAR, road, speed)
    t, Z, _ = simulate_road(CAR, road, speed, h=5e-6, split_at_edges=True)
    history, peaks = run_errors(CAR, t, Z, reference)
    print(f"Reference check (medium pothole, 40 km/h): DOP853 used {reference.n_evals} evaluations; "
          f"RK4 with h = 5 us differs by {history:.1e} (history) and {peaks:.1e} (peaks)")


def convergence_figure():
    """Error against time step at 40 km/h for a smooth road change and a sudden one."""
    names = ["Speed hump, 75 mm x 3.7 m", "Medium pothole, 50 mm x 0.6 m"]
    speed = 40 / 3.6
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for ax, name in zip(axes, names):
        road = ROADS[name]
        reference = reference_solution(CAR, road, speed)
        orders = []
        for method, (step, split) in METHODS.items():
            steps = [h for h in FIGURE_STEPS if step is rk4_step or h < 7e-3]   # Euler is unstable above 7.2 ms
            result = step_study(CAR, road, speed, steps, step, split, reference)
            errors = np.maximum(result["history"], 1e-10)   # keep exact (zero) errors on the log axis
            ax.loglog(1e3 * np.array(steps), 100 * errors, color=COLOURS[method], marker=MARKERS[method],
                      ms=5, lw=1.8, label=method)
            orders.append(f"{method}: {observed_order(steps, result['history']):.1f}")
        ax.axhline(100 * TOL, color=MUTED, lw=1)
        ax.text(0.045, 100 * TOL * 1.4, "1% error", color=SECONDARY, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time step h [ms]")
        ax.text(0.97, 0.04, "Observed order\n" + "\n".join(orders), transform=ax.transAxes,
                fontsize=7, color=SECONDARY, ha="right", va="bottom")
        style(ax)
        print(f"  {name}: observed order " + ", ".join(orders))
    axes[0].set_ylabel("Largest error in body acceleration\nand tyre force [% of peak]")
    axes[0].set_ylim(1e-7, 1e3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3,
               fontsize=8, frameon=False)
    fig.suptitle("Error against time step at 40 km/h", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTS / "error_vs_time_step.png", dpi=200)
    plt.close(fig)


def required_step_table():
    """Time step for the solver to follow the motion within 1%, for every road and speed."""
    lines = ["| Road | Method | " + " | ".join(f"{v} km/h" for v in SPEEDS_KMH) + " |",
             "|---|---|" + "---|" * len(SPEEDS_KMH)]
    csv = ["road,method,speed_kmh,required_step_ms,rhs_evaluations"]
    for name, road in ROADS.items():
        cells = {"RK4": [], "RK4, restarted at edges": []}
        for kmh in SPEEDS_KMH:
            reference = reference_solution(CAR, road, kmh / 3.6)
            for method in cells:
                step, split = METHODS[method]
                result = step_study(CAR, road, kmh / 3.6, STEPS, step, split, reference)
                h, n_evals = required_step(result, TOL, measure="history")
                cells[method].append("< 0.1" if h is None else f"{1e3 * h:g}")
                csv.append(f"{name},{method},{kmh},{'' if h is None else 1e3 * h},{n_evals or ''}")
        for method, values in cells.items():
            lines.append(f"| {name} | {method} | " + " | ".join(values) + " |")
        print(f"  {name}: done")
    table = "\n".join(lines)
    (RESULTS / "required_time_step.md").write_text(
        "Largest time step [ms] for which the body acceleration and tyre force stay within 1% "
        "of the reference at every step\n\n"
        + table + "\n")
    (RESULTS / "required_time_step.csv").write_text("\n".join(csv) + "\n")
    print(table)


def cost_table():
    """Cost of reaching 1% accuracy on the medium pothole, for every method."""
    road = ROADS["Medium pothole, 50 mm x 0.6 m"]
    lines = ["| Method | Speed | Time step or tolerance | Error | RHS evaluations |", "|---|---|---|---|---|"]
    for kmh in (40, 80):
        speed = kmh / 3.6
        reference = reference_solution(CAR, road, speed)
        for method, (step, split) in METHODS.items():
            steps = [h for h in STEPS + (5e-5, 2e-5) if step is rk4_step or h < 7e-3]
            result = step_study(CAR, road, speed, steps, step, split, reference)
            h, n_evals = required_step(result, TOL, measure="history")
            index = list(result["steps"]).index(h)
            lines.append(f"| {method} | {kmh} km/h | h = {1e3 * h:g} ms | "
                         f"{100 * result['history'][index]:.2f}% | {n_evals} |")
        grid = np.arange(0.0, 0.3 + road.length / speed + 2.0, 1e-4)
        for split in (False, True):
            for rtol in (1e-3, 1e-6):
                run = adaptive_solve(CAR, road, speed, rtol=rtol, atol=rtol * 1e-3, split_at_edges=split)
                error, _ = run_errors(CAR, grid, run.states(grid), reference)   # on a 0.1 ms grid
                name = "RK45 (adaptive), restarted at edges" if split else "RK45 (adaptive)"
                lines.append(f"| {name} | {kmh} km/h | rtol = {rtol:g} | {100 * error:.2f}% | {run.n_evals} |")
    table = "\n".join(lines)
    (RESULTS / "cost_medium_pothole.md").write_text(
        "Cost of following the motion within 1% on the medium pothole (50 mm deep, 0.6 m long). "
        "Adaptive runs are checked every 0.1 ms using their dense output.\n\n" + table + "\n")
    print(table)


def skipped_pothole_figure():
    """Show the adaptive RK45 solver stepping straight over a pothole at 80 km/h."""
    road, speed = ROADS["Medium pothole, 50 mm x 0.6 m"], 80 / 3.6
    reference = reference_solution(CAR, road, speed)
    plain = adaptive_solve(CAR, road, speed, rtol=1e-6, atol=1e-9)
    restarted = adaptive_solve(CAR, road, speed, rtol=1e-6, atol=1e-9, split_at_edges=True)
    t = np.arange(0.2, 0.7, 1e-4)
    in_view = (plain.t >= t[0]) & (plain.t <= t[-1])

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(1e3 * t, 1e3 * road_input(road, speed)(t), color=MUTED, lw=1.5, label="Road under the tyre")
    ax.plot(1e3 * t, 1e3 * reference.states(t)[:, 0], color=COLOURS["Euler"], lw=1.8, label="Reference wheel motion")
    ax.plot(1e3 * t, 1e3 * plain.states(t)[:, 0], color=COLOURS["RK4"], lw=1.8,
            label=f"RK45, rtol 1e-6 ({plain.n_evals} evaluations)")
    ax.plot(1e3 * plain.t[in_view], 1e3 * plain.states(plain.t[in_view])[:, 0], "o", color=COLOURS["RK4"],
            ms=6, mec="#fcfcfb", mew=1.5, label="RK45 steps")
    ax.plot(1e3 * t, 1e3 * restarted.states(t)[:, 0], color=COLOURS["RK4, restarted at edges"], lw=1.8,
            ls=(0, (1, 2)), label=f"RK45 restarted at edges ({restarted.n_evals} evaluations)")
    after = np.searchsorted(plain.t, 0.3)          # first RK45 step time at or after the near edge
    jump = 1e3 * plain.t[after - 1], 1e3 * plain.t[after]
    ax.annotate(f"One RK45 step, {jump[0]:.0f} to {jump[1]:.0f} ms,\nnever samples the pothole",
                xy=(jump[1], 0), xytext=(jump[1] + 30, -18), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=SECONDARY, lw=0.8))
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Height [mm]")
    ax.set_title("80 km/h over a 50 mm deep, 0.6 m long pothole", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "adaptive_solver_skips_pothole.png", dpi=200)
    plt.close(fig)
    print(f"  RK45 (rtol 1e-6) stepped from {plain.t[after - 1]:.3f} s to {plain.t[after]:.3f} s, "
          f"straight over the pothole (0.300 to {0.3 + road.length / speed:.3f} s)")


def check_program_step():
    """Check the program's time step (RK4 restarted at edges) on every road and speed.

    Unlike the tables above, this also checks the peak values the program
    reports, which can be missed between steps if the step is too coarse.
    """
    worst_history, worst_peaks = 0.0, 0.0
    for name, road in ROADS.items():
        for kmh in SPEEDS_KMH + (110,):
            reference = reference_solution(CAR, road, kmh / 3.6)
            t, Z, _ = simulate_road(CAR, road, kmh / 3.6, h=TIME_STEP, split_at_edges=True)
            history, peaks = run_errors(CAR, t, Z, reference)
            worst_history, worst_peaks = max(worst_history, history), max(worst_peaks, peaks)
    text = (f"Program time step h = {1e3 * TIME_STEP:g} ms (RK4 restarted at edges), every road at "
            f"{', '.join(str(v) for v in SPEEDS_KMH + (110,))} km/h: worst motion error "
            f"{100 * worst_history:.2f}%, worst peak error {100 * worst_peaks:.2f}%")
    (RESULTS / "program_time_step_check.md").write_text(text + "\n")
    print(text)


def main():
    """Run every part of the study and save the results."""
    RESULTS.mkdir(exist_ok=True)
    parts = [("Context", print_context), ("Reference check", check_reference),
             ("Convergence figure", convergence_figure), ("Required time step", required_step_table),
             ("Cost of 1% accuracy", cost_table), ("Adaptive solver figure", skipped_pothole_figure),
             ("Check of the program's time step", check_program_step)]
    for title, part in parts:
        start = time.perf_counter()
        print(f"\n== {title}")
        part()
        print(f"   ({time.perf_counter() - start:.0f} s)")
    print(f"\nSaved figures and tables in {RESULTS}")


if __name__ == "__main__":
    main()
