"""Optimisation and performance: analyse and improve the simulation's performance.

Uses one demanding case throughout (the default car over a 50 mm deep, 0.6 m
long pothole at 40 km/h, where the wheel leaves the road) and compares the
original code and method with the final version:

1. Profiling (``cProfile``) shows where the time goes.
2. Code improvements guided by the profile (a plain-Python path for single
   road heights, and the equations of motion written out term by term
   instead of ``A @ z``), checked to give the same results.
3. A FLOP count and arithmetic intensity estimate (notebook 2.1) explain the
   performance: each RK4 step needs only about 100 to 200 FLOPs, so the time
   is set by Python's overhead per operation, not by arithmetic or memory.
4. The time step trade-off between accuracy and run time, used to find the
   largest step that keeps both the motion and the peak values within 1%.
5. Baseline against final design, separating the effect of the code changes
   from the effect of restarting the solver at the road edges.
6. Repeatability, scalability and numerical stability.

The original code is kept in this file only as the baseline for comparison.
Run from the project folder (takes about a minute)::

    python performance_study.py

Tables are saved in ``results/performance.md`` and the trade-off figure in
``results/performance_tradeoff.png``.
"""

import cProfile
import pstats
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quarter_car.compare import measure
from quarter_car.model import G, QuarterCar, damping_from_ratio, make_rhs, state_matrices
from quarter_car.road import Pothole
from quarter_car.scenarios import edge_times, road_input
from quarter_car.solvers import rk4_step, simulate, simulate_in_segments
from quarter_car.timestep import max_stable_step, reference_solution, run_errors

RESULTS = Path(__file__).resolve().parent / "results"
CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=damping_from_ratio(0.3, 20e3, 300.0), Kw=200e3)
ROAD, SPEED = Pothole(0.05, 0.6), 40 / 3.6       # the case used throughout
PROGRAM_STEP = 1e-3                              # time step used by main.py [s]
REPEATS = 15                                     # timed runs; the fastest is reported
STEPS = (10e-3, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4)    # time steps for the trade-off [s]

# FLOPs counted by hand from the code (additions, subtractions, multiplications, divisions).
RK4_COMBINE_FLOPS = 60     # forming the stage states and the weighted average in rk4_step
CLAMP_FLOPS = 2            # keeping the time inside a segment (start + margin, end - margin)
ROAD_FLOPS = 2             # speed * (t - lead_in_time); a pothole height needs no arithmetic
RHS_FLOPS = {"original": 28 + 4 + 4 + 2 + ROAD_FLOPS + CLAMP_FLOPS,   # A @ z, B * r, sum, lift-off check
             "improved": 5 + 2 + 2 + 2 + ROAD_FLOPS + CLAMP_FLOPS}    # suspension, tyre, two accelerations

# Chart styling.
FINAL, BASELINE, INK, SECONDARY, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ---- The original code, kept only as the baseline -------------------------------

def original_pothole_profile(road, distance):
    """Pothole height as first written: always uses NumPy arrays, even for one number."""
    s = np.asarray(distance, dtype=float)
    r = np.where((s >= 0.0) & (s < road.length), -road.depth, 0.0)
    return r if r.ndim else float(r)


def original_road_input(road, speed, lead_in_time=0.3):
    """Road input as first written: always uses NumPy arrays, even for one time."""
    def r(t):
        return original_pothole_profile(road, speed * (np.asarray(t, dtype=float) - lead_in_time))
    return r


def original_rhs(car, road):
    """Right-hand side as first written: the matrix product ``A @ z + B r``."""
    A, B, _ = state_matrices(car)
    static_load = (car.m1 + car.m2) * G

    def rhs(t, z):
        r = road(t)
        dz = A @ z + B * r
        tyre_change = car.Kw * (r - z[0])
        if tyre_change < -static_load:
            dz[2] -= (tyre_change + static_load) / car.m1
        return dz

    return rhs


# ---- Running the case ------------------------------------------------------------

def run(code, h=PROGRAM_STEP, restart=True, t_after=2.0):
    """Simulate the case with the original or improved code. Returns times and states."""
    if code == "original":
        rhs = original_rhs(CAR, original_road_input(ROAD, SPEED))
    else:
        rhs = make_rhs(CAR, road_input(ROAD, SPEED))
    t_end = 0.3 + ROAD.length / SPEED + t_after
    if restart:
        t, Z, _ = simulate_in_segments(rk4_step, rhs, np.zeros(4), t_end, h, edge_times(ROAD, SPEED))
    else:
        t, Z, _ = simulate(rk4_step, rhs, np.zeros(4), t_end, h)
    return t, Z


def profile_top(code, n=6):
    """Profile one run and return the n functions that used the most time."""
    profiler = cProfile.Profile()
    profiler.enable()
    run(code)
    profiler.disable()
    stats = pstats.Stats(profiler).stats              # {(file, line, name): (calls, _, own time, ...)}
    total = sum(entry[2] for entry in stats.values())
    rows = sorted(stats.items(), key=lambda item: item[1][2], reverse=True)[:n]
    return [(f"{Path(key[0]).name}:{key[2]}", entry[1], 1e3 * entry[2], entry[2] / total)
            for key, entry in rows], 1e3 * total


def flops_per_step(code):
    """FLOPs in one RK4 step: four right-hand-side calls plus combining the stages."""
    return 4 * RHS_FLOPS[code] + RK4_COMBINE_FLOPS


def machine_rates():
    """Measured FLOP rates of a large and a tiny NumPy calculation on this computer."""
    big = np.random.default_rng(0).random((1000, 1000))
    _, seconds, _ = measure(lambda: big @ big, 3)
    large_rate = 2 * 1000**3 / seconds                 # FLOP/s of a 1000 x 1000 matrix product
    A, _, _ = state_matrices(CAR)
    z = np.ones(4)
    loops = 20000
    start = time.perf_counter()
    for _ in range(loops):
        A @ z
    small_rate = 28 * loops / (time.perf_counter() - start)   # FLOP/s of one 4 x 4 matrix-vector product
    return large_rate, small_rate


# ---- The study -------------------------------------------------------------------

def main():
    """Run every part of the study and save the tables and figure."""
    RESULTS.mkdir(exist_ok=True)
    reference = reference_solution(CAR, ROAD, SPEED)
    out = ["# Optimisation and performance", "",
           f"Case: default car, {ROAD.label}, {3.6 * SPEED:.0f} km/h (the wheel leaves the road). "
           f"Run times are the fastest of {REPEATS} runs on this computer.", ""]

    # 1. Profiling
    print("Profiling...")
    out += ["## 1. Profile of one run at the program's 1 ms step (cProfile, own time per function)", ""]
    for code in ("original", "improved"):
        rows, total = profile_top(code)
        out += [f"**{code.capitalize()} code** (profiled total {total:.0f} ms):", "",
                "| Function | Calls | Own time [ms] | Share |", "|---|---|---|---|"]
        out += [f"| {name} | {calls} | {ms:.1f} | {100 * share:.0f}% |" for name, calls, ms, share in rows]
        out.append("")

    # 4. Time step trade-off (run first: it sets the steps compared in section 2)
    print("Running the time step trade-off...")
    curves = {}
    for code, restart in (("original", False), ("improved", True)):
        curves[code] = []
        for h in STEPS:
            (t, Z), seconds, _ = measure(lambda: run(code, h, restart), 5)
            curves[code].append((h, max(run_errors(CAR, t, Z, reference)), seconds))
    largest_ok = {code: max(h for h, error, _ in points if error <= 0.01) for code, points in curves.items()}

    # 2. Baseline against final design
    print("Timing baseline and final versions...")
    h_base, h_final = largest_ok["original"], largest_ok["improved"]
    versions = [
        ("A. Baseline", "Fixed-step RK4", "original", h_base, False),
        ("B. Code improved only", "Fixed-step RK4", "improved", h_base, False),
        ("C. Method improved only", "RK4 restarted at edges", "original", h_final, True),
        ("D. Final, largest step within 1%", "RK4 restarted at edges", "improved", h_final, True),
        ("E. Final, program setting", "RK4 restarted at edges", "improved", PROGRAM_STEP, True),
    ]
    out += ["## 2. Baseline against final design", "",
            "Each step is the largest that keeps both the motion and the peak values within 1% "
            "(from section 4), except E, the 1 ms step the program uses.", "",
            "| Version | Method | Code | Step | Motion error | Peak error | Steps | FLOPs per run "
            "| Run time [ms] | Achieved FLOP/s | Memory [kB] |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    timings = {}
    for name, method, code, h, restart in versions:
        (t, Z), seconds, memory = measure(lambda: run(code, h, restart), REPEATS)
        motion, peaks = run_errors(CAR, t, Z, reference)
        steps = len(t) - 1
        flops = steps * flops_per_step(code)
        timings[name[0]] = seconds
        out.append(f"| {name} | {method} | {code} | {1e3 * h:g} ms | {100 * motion:.3f}% | "
                   f"{100 * peaks:.3f}% | {steps} | {flops / 1e3:.0f} k | {1e3 * seconds:.1f} | "
                   f"{flops / seconds / 1e6:.1f} M | {memory / 1024:.0f} |")
    _, Z_orig = run("original")
    t_new, Z_new = run("improved")
    difference = np.abs(Z_new - Z_orig).max() / np.abs(Z_orig).max()
    out += ["", f"- Code: the improved code gives the same results as the original (largest difference "
            f"{difference:.1e} of the largest state value, which is round-off) and is "
            f"{timings['A'] / timings['B']:.1f} times faster (A against B).",
            f"- Method: restarting at the road edges allows a {h_final / h_base:g} times larger step for "
            f"the same accuracy, making it {timings['A'] / timings['C']:.1f} times faster (A against C).",
            f"- Both together: {timings['A'] / timings['D']:.1f} times faster than the baseline at the "
            "same accuracy (A against D).",
            f"- The program uses 1 ms (E) for smooth animation frames and sharper peaks: still "
            f"{timings['A'] / timings['E']:.1f} times faster than the baseline, with a much smaller error.",
            ""]

    # 3. FLOPs and arithmetic intensity
    print("Measuring machine FLOP rates...")
    large_rate, small_rate = machine_rates()
    bytes_per_call = {"original": 8 * (16 + 4 + 4 + 4 + 1), "improved": 8 * (4 + 4 + 5 + 1)}
    out += ["## 3. FLOPs and arithmetic intensity", "",
            "| Code | FLOPs per right-hand side | FLOPs per RK4 step | Bytes per right-hand side "
            "| Arithmetic intensity [FLOP/byte] |", "|---|---|---|---|---|"]
    for code in ("original", "improved"):
        out.append(f"| {code} | {RHS_FLOPS[code]} | {flops_per_step(code)} | {bytes_per_call[code]} | "
                   f"{RHS_FLOPS[code] / bytes_per_call[code]:.2f} |")
    out += ["", f"- A 1000 x 1000 matrix product runs at {large_rate / 1e9:.1f} GFLOP/s on this computer, "
            f"but one 4 x 4 matrix-vector product runs at only {small_rate / 1e6:.0f} MFLOP/s: for tiny "
            "arrays the time is set by the overhead of each NumPy call, not by the arithmetic.",
            "- The simulation is in the same regime: the achieved FLOP/s in section 2 is thousands of "
            "times below what the computer can do. The data (about 100 to 250 bytes) stays in the "
            "fastest cache, so memory bandwidth does not limit it either. Speed-ups therefore come "
            "from doing fewer Python and NumPy operations, not fewer FLOPs.", ""]

    # 4. Time step trade-off table
    out += ["## 4. Time step trade-off (error = larger of motion and peak error)", "",
            "| Step | Baseline error | Baseline time [ms] | Final error | Final time [ms] |",
            "|---|---|---|---|---|"]
    for (h, e0, s0), (_, e1, s1) in zip(curves["original"], curves["improved"]):
        out.append(f"| {1e3 * h:g} ms | {100 * e0:.3f}% | {1e3 * s0:.1f} | {100 * e1:.3f}% | {1e3 * s1:.1f} |")
    out.append("")

    # 6. Repeatability, scalability and stability
    print("Checking repeatability and scalability...")
    _, Z_again = run("improved")
    times = []
    for _ in range(30):
        start = time.perf_counter()
        run("improved")
        times.append(time.perf_counter() - start)
    times = 1e3 * np.array(times)
    out += ["## 5. Repeatability", "",
            f"- Results: two runs with the same inputs are {'identical' if np.array_equal(Z_new, Z_again) else 'different'}.",
            f"- Run time over 30 runs: fastest {times.min():.1f} ms, median {np.median(times):.1f} ms, "
            f"slowest {times.max():.1f} ms (spread {100 * times.std() / times.mean():.0f}% of the mean), "
            "which is why the fastest of several runs is reported.", "",
            "## 6. Scalability (program setting, 1 ms steps)", "",
            "| Simulated time [s] | Steps | Run time [ms] | Time per simulated second [ms] | Memory [kB] |",
            "|---|---|---|---|---|"]
    for t_after in (1.0, 2.0, 4.0, 8.0, 16.0):
        (t, _), seconds, memory = measure(lambda: run("improved", t_after=t_after), 5)
        out.append(f"| {t[-1]:.1f} | {len(t) - 1} | {1e3 * seconds:.1f} | {1e3 * seconds / t[-1]:.1f} | "
                   f"{memory / 1024:.0f} |")
    limit = max_stable_step(CAR, "rk4")
    out += ["", "## 7. Numerical stability", "",
            f"- Largest stable RK4 step for this car: {1e3 * limit:.1f} ms (Euler: "
            f"{1e3 * max_stable_step(CAR, 'euler'):.1f} ms). The program's 1 ms step is "
            f"{limit / PROGRAM_STEP:.0f} times smaller, so accuracy, not stability, sets the step.", ""]

    (RESULTS / "performance.md").write_text("\n".join(out))
    print("\n".join(out))
    save_figure(curves)
    print(f"\nSaved results/performance.md and results/performance_tradeoff.png in {RESULTS}")


def save_figure(curves):
    """Error against run time for each time step, baseline against final design."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    styles = {"original": ("Baseline: fixed-step RK4, original code", BASELINE, "s"),
              "improved": ("Final: RK4 restarted at edges, improved code", FINAL, "o")}
    for code, points in curves.items():
        label, colour, marker = styles[code]
        h, error, seconds = (np.array(values) for values in zip(*points))
        ax.loglog(1e3 * seconds, 100 * error, color=colour, marker=marker, ms=6, lw=1.8, label=label)
        for hi, ei, si in zip(h, error, seconds):
            ax.annotate(f"{1e3 * hi:g} ms", (1e3 * si, 100 * ei), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=SECONDARY)
    program = [p for p in curves["improved"] if p[0] == PROGRAM_STEP][0]
    ax.plot(1e3 * program[2], 100 * program[1], "o", ms=13, mfc="none", mec=INK, mew=1.2)
    ax.annotate("program setting", (1e3 * program[2], 100 * program[1]), textcoords="offset points",
                xytext=(10, -14), fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, lw=1)
    ax.text(ax.get_xlim()[0] * 1.1, 1.15, "1% error", fontsize=8, color=SECONDARY)
    ax.set_xlabel("Run time [ms], log scale")
    ax.set_ylabel("Error [% of peak], log scale")
    ax.set_title(f"Accuracy against run time for each time step\n{ROAD.label}, "
                 f"{3.6 * SPEED:.0f} km/h", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=SECONDARY, labelsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "performance_tradeoff.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
