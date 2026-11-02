"""Time step study: how fine must the time step be to capture sudden road changes?

Runs a set of bumps and potholes with a fixed test car, compares each run
with an accurate reference solution (see ``quarter_car/timestep.py``) and
saves these results in the ``results`` folder:

* error_vs_time_step.png: how the error falls as the time step is reduced.
* required_time_step.md: the time step each road and speed needs for 1%.
* restart_road_readings.png and restart_acceleration_error.png: RK4 with and
  without restarts at the edges on the hardest pothole, same 1 ms step.
* program_time_step_check.md: the program's 1 ms step on every road and speed.
* input_range_check.md: the program's 1 ms step at the extremes of every
  input range.

Run from the project folder (takes a few minutes)::

    python time_step_study.py
"""

import itertools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quarter_car.model import QuarterCar, damping_from_ratio, natural_modes
from quarter_car.road import Bump, Pothole
from quarter_car.scenarios import body_acceleration, road_input, simulate_road
from quarter_car.solvers import euler_step, rk4_step
from main import TIME_STEP, build_car, build_road
from quarter_car.inputs import CAR_INPUTS, ROAD_INPUTS, SPEED_INPUT
from quarter_car.timestep import (STEPS, max_stable_step, reference_solution, required_step, run_errors,
                                  step_study)

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
    for name, road in ROADS.items():
        cells = {"RK4": [], "RK4, restarted at edges": []}
        for kmh in SPEEDS_KMH:
            reference = reference_solution(CAR, road, kmh / 3.6)
            for method in cells:
                step, split = METHODS[method]
                result = step_study(CAR, road, kmh / 3.6, STEPS, step, split, reference)
                h, _ = required_step(result, TOL, measure="history")
                cells[method].append("< 0.1" if h is None else f"{1e3 * h:g}")
        for method, values in cells.items():
            lines.append(f"| {name} | {method} | " + " | ".join(values) + " |")
        print(f"  {name}: done")
    table = "\n".join(lines)
    (RESULTS / "required_time_step.md").write_text(
        "Largest time step [ms] for which the body acceleration and tyre force stay within 1% "
        "of the reference at every step\n\n"
        + table + "\n")
    print(table)


def restart_figures():
    """Show RK4 with and without restarts at the edges on the hardest pothole.

    The hardest pothole is the deepest and shortest one at the top speed. Both
    runs use the program's 1 ms step, so the only difference is the restart.
    Without restarting, the step that ends on the near edge and the step that
    crosses the far edge each read the road on both sides of the edge, which
    puts an error of about 2% into the body acceleration. Restarting keeps
    every step on one side of an edge. Saves two figures: where each method
    reads the road, and the error in body acceleration that results.
    """
    road, speed, h = Pothole(0.15, 0.3), 110 / 3.6, TIME_STEP
    reference = reference_solution(CAR, road, speed)
    road_height = road_input(road, speed)
    near_edge, far_edge = 0.3, 0.3 + road.length / speed         # times the tyre meets each edge [s]
    title = "110 km/h over a 150 mm deep, 0.3 m long pothole"
    ring = "#fcfcfb"
    methods = {   # restarted at edges?: (name, colour, marker style)
        False: ("not restarted", COLOURS["RK4"], dict(marker="s", ms=7.5, mfc=ring, mec=COLOURS["RK4"], mew=1.4)),
        True: ("restarted at edges", COLOURS["RK4, restarted at edges"], dict(marker="^", ms=5.5, mec=ring, mew=0.8)),
    }
    runs = {restarted: simulate_road(CAR, road, speed, h=h, split_at_edges=restarted)[:2] for restarted in methods}

    # Figure 1: the moments each method reads the road around the pothole.
    zoom = (0.297, 0.313)                                         # time window shown [s]
    t_zoom = np.arange(zoom[0], zoom[1], 1e-6)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(1e3 * t_zoom, 1e3 * road_height(t_zoom), color=MUTED, lw=1.5, label="Road under the tyre")
    for restarted, (name, colour, shape) in methods.items():
        t = runs[restarted][0]
        nudge = 1e-12 if restarted else 0.0     # a restarted run reads just inside each segment
        reads = np.concatenate([t[:-1] + nudge, (t[:-1] + t[1:]) / 2, t[1:] - nudge])   # start, middle, end
        reads = reads[(reads >= zoom[0]) & (reads <= zoom[1])]
        ax.plot(1e3 * reads, 1e3 * road_height(reads), ls="none", color=colour, **shape,
                label=f"RK4 readings, 1 ms step, {name}")
    label = "Step that reads both sides of an edge (not restarted)"
    t = runs[False][0]
    for start, end in zip(t[:-1], t[1:]):
        if zoom[0] <= start < zoom[1] and np.ptp(road_height(np.array([start, (start + end) / 2, end]))) > 0:
            ax.axvspan(1e3 * start, 1e3 * end, color=COLOURS["RK4"], alpha=0.18, lw=0, label=label)
            label = "_nolegend_"                 # one legend entry for all shaded steps
    ax.set_title(f"Where RK4 reads the road, {title}", fontsize=10)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Road height [mm]")
    ax.set_xlim(1e3 * zoom[0], 1e3 * zoom[1])
    ax.set_ylim(-175, 30)
    ax.legend(fontsize=7.5, frameon=False, loc="center")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "restart_road_readings.png", dpi=200)
    plt.close(fig)

    # Figure 2: the error in body acceleration that results.
    view = (0.28, 0.46)                                           # time window shown [s]
    t_run = np.arange(0.0, far_edge + 2.0, 1e-5)
    accel_peak = np.abs(body_acceleration(CAR, reference.states(t_run))).max()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axvspan(1e3 * near_edge, 1e3 * far_edge, color=MUTED, alpha=0.15, lw=0)
    ax.text(1e3 * far_edge + 1.5, 0.95, "Pothole", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=SECONDARY, va="top")
    for restarted, (name, colour, _) in methods.items():
        t, Z = runs[restarted]
        error, _ = run_errors(CAR, t, Z, reference)
        error_text = "< 0.01%" if error < 1e-4 else f"{100 * error:.1f}%"
        accel_error = body_acceleration(CAR, Z) - body_acceleration(CAR, reference.states(t))
        ax.plot(1e3 * t, 100 * accel_error / accel_peak, color=colour, lw=1.8,
                label=f"RK4, 1 ms step, {name} (error {error_text})")
        print(f"  RK4, 1 ms step, {name}: error {error_text}")
    ax.set_title(f"Error in body acceleration, {title}", fontsize=10)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Error [% of peak]")
    ax.set_xlim(1e3 * view[0], 1e3 * view[1])
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "restart_acceleration_error.png", dpi=200)
    plt.close(fig)


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


def check_input_ranges():
    """Check the program's time step at the extremes of every input range.

    Every combination of the lowest and highest body mass, spring stiffness,
    damping ratio and tyre stiffness (16 cars) is run over six extreme roads:
    the deepest bump and pothole at their shortest and the top speed, at their
    longest and the lowest speed, and the shallowest of each at the top speed.
    A run passes if both the motion and the peaks are within 1%.
    """
    size, length, speed = (limits[3:] for limits in (*ROAD_INPUTS, SPEED_INPUT))   # (lowest, highest)
    roads = [("pothole", size[1], length[0], speed[1]), ("bump", size[1], length[0], speed[1]),
             ("pothole", size[1], length[1], speed[0]), ("bump", size[1], length[1], speed[0]),
             ("pothole", size[0], length[0], speed[1]), ("bump", size[0], length[1], speed[1])]
    rows = []
    for car_limits in itertools.product(*[(low, high) for *_, low, high in CAR_INPUTS]):
        values = {key: value for (key, *_), value in zip(CAR_INPUTS, car_limits)}
        car = build_car(values)
        for kind, height, road_length, kmh in roads:
            road = build_road(kind, {"height_or_depth": height, "length": road_length})
            reference = reference_solution(car, road, kmh / 3.6)
            t, Z, _ = simulate_road(car, road, kmh / 3.6, h=TIME_STEP, split_at_edges=True)
            history, peaks = run_errors(car, t, Z, reference)
            rows.append((max(history, peaks), history, peaks, car_limits,
                         f"{height:g} mm x {road_length:g} m {kind}", kmh))
    passed = sum(worst <= TOL for worst, *_ in rows)
    summary = (f"Program time step h = {1e3 * TIME_STEP:g} ms (RK4 restarted at edges) at the extremes of "
               f"every input range: {len(rows) // len(roads)} cars (every combination of the lowest and "
               f"highest body mass, spring stiffness, damping ratio and tyre stiffness) over {len(roads)} "
               f"extreme roads. Runs with both the motion and the peaks within 1% of the reference: "
               f"{passed} of {len(rows)}. Worst error: {100 * max(rows)[0]:.2f}%.")
    lines = ["| Body mass [kg] | Spring [kN/m] | Damping ratio | Tyre [kN/m] | Road | Speed [km/h] "
             "| Motion error | Peak error |", "|---|---|---|---|---|---|---|---|"]
    for _, history, peaks, (mass, spring, zeta, tyre), road_name, kmh in sorted(rows, reverse=True)[:5]:
        lines.append(f"| {mass:g} | {spring:g} | {zeta:g} | {tyre:g} | {road_name} | {kmh:g} | "
                     f"{100 * history:.2f}% | {100 * peaks:.2f}% |")
    (RESULTS / "input_range_check.md").write_text(summary + "\n\nThe five worst runs:\n\n"
                                                  + "\n".join(lines) + "\n")
    print(summary)


def main():
    """Run every part of the study and save the results."""
    RESULTS.mkdir(exist_ok=True)
    parts = [("Context", print_context), ("Reference check", check_reference),
             ("Convergence figure", convergence_figure), ("Required time step", required_step_table),
             ("Restart figures", restart_figures),
             ("Check of the program's time step", check_program_step),
             ("Check at the extremes of the input ranges", check_input_ranges)]
    for title, part in parts:
        start = time.perf_counter()
        print(f"\n== {title}")
        part()
        print(f"   ({time.perf_counter() - start:.0f} s)")
    print(f"\nSaved figures and tables in {RESULTS}")


if __name__ == "__main__":
    main()
