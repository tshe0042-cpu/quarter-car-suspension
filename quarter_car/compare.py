"""Compare the chosen solver with alternative approaches, for any car, road and speed.

Seven ways of solving the same run are compared:

* Euler with a fixed step (notebook 7.3)
* RK4 with a fixed step (notebook 7.4)
* RK4 restarted at the road edges (the method the program uses)
* SciPy's adaptive RK45, at its default and at a tight tolerance
* SciPy's adaptive RK45 restarted at the road edges
* the exact solution of the linear model with ``scipy.signal.lsim``, which is
  fast but cannot model the wheel leaving the road, because lift-off makes
  the equations non-linear

For each fixed-step method, the largest time step that follows the motion
within 1% is found first (as in ``time_step_study.py``). Every method is then
measured on accuracy against an accurate reference solution, right-hand-side
evaluations, run time and peak memory (notebook 2.1).
"""

import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lsim

from quarter_car.model import state_matrices
from quarter_car.scenarios import road_input, simulate_road
from quarter_car.solvers import euler_step, rk4_step
from quarter_car.timestep import (STEPS, adaptive_solve, max_stable_step, reference_solution,
                                  required_step, run_errors, step_study)

CHOSEN = "RK4, restarted at edges"
TOL = 0.01                                                    # accuracy target: 1% error
EULER_STEPS = (5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5)  # Euler time steps tried [s]

# Chart colours: the chosen method is highlighted and the others are grey.
HIGHLIGHT, OTHER, INK, SECONDARY, GRID, WARNING = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b"


def linear_solve(car, road, speed, h=1e-3, lead_in_time=0.3, t_after=2.0):
    """Solve the linear model (tyre always on the road) with ``scipy.signal.lsim``.

    ``lsim`` steps the state-space model with the matrix exponential and
    treats the road input as straight lines between samples, so for the
    linear model its only error comes from sampling the road. It ignores
    lift-off: if the road drops away, the tyre "pulls" the wheel down.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    h : float, optional
        Spacing of the road samples and output times [s]. Default 1 ms.
    lead_in_time, t_after : float, optional
        As for ``scenarios.simulate_road``.

    Returns
    -------
    t : ndarray, shape (n,)
        Times [s].
    Z : ndarray, shape (n, 4)
        States ``[x, y, x_dot, y_dot]`` at each time.
    """
    A, B, _ = state_matrices(car)
    t_end = lead_in_time + road.length / speed + t_after
    t = np.linspace(0.0, t_end, int(round(t_end / h)) + 1)
    r = road_input(road, speed, lead_in_time)(t)
    _, _, Z = lsim((A, B[:, None], np.eye(4), np.zeros((4, 1))), r, t)
    return t, Z


def measure(func, repeats=7):
    """Time a calculation and find its peak memory use.

    The run time is the fastest of several runs, as Python's ``timeit``
    recommends: slower runs are caused by other programs using the computer,
    not by the code itself. Memory is measured in one extra run with
    ``tracemalloc``, because tracing slows the code down.

    Parameters
    ----------
    func : callable
        Calculation to measure, called with no arguments.
    repeats : int, optional
        Number of timed runs. Default 7 (as ``%timeit`` uses).

    Returns
    -------
    result
        What ``func`` returned.
    seconds : float
        Fastest run time [s].
    peak_bytes : int
        Largest memory allocated by Python during one run [bytes].
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    func()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, min(times), peak_bytes


def _fixed_step(step, split, steps):
    """Method that finds its 1% time step, then measures one run at that step."""
    def run(car, road, speed, reference, repeats, t_after):
        if step is euler_step:                                # only try steps where Euler is stable
            limit = max_stable_step(car, "euler")
            steps_tried = [h for h in steps if h < 0.95 * limit] or [0.5 * limit]
        else:
            steps_tried = list(steps)
        result = step_study(car, road, speed, steps_tried, step, split, reference, t_after)
        h, _ = required_step(result, TOL)
        setting = f"h = {1e3 * h:g} ms" if h is not None else f"h = {1e3 * steps_tried[-1]:g} ms (finest tried)"
        h = h if h is not None else steps_tried[-1]
        (t, Z, _), seconds, memory = measure(
            lambda: simulate_road(car, road, speed, h=h, step=step, split_at_edges=split,
                                  t_after=t_after), repeats)
        evals = (len(t) - 1) * (1 if step is euler_step else 4)
        return setting, t, Z, evals, seconds, memory
    return run


def _adaptive(rtol, split):
    """Method that measures SciPy's adaptive RK45 at the given tolerance."""
    def run(car, road, speed, reference, repeats, t_after):
        result, seconds, memory = measure(
            lambda: adaptive_solve(car, road, speed, rtol=rtol, atol=rtol * 1e-3,
                                   split_at_edges=split, t_after=t_after), repeats)
        t = np.arange(0.0, result.t[-1], 1e-4)               # judged every 0.1 ms using the dense output
        return f"rtol = {rtol:g}", t, result.states(t), result.n_evals, seconds, memory
    return run


def _linear(car, road, speed, reference, repeats, t_after):
    """Method that measures the exact solution of the linear model (no lift-off)."""
    (t, Z), seconds, memory = measure(lambda: linear_solve(car, road, speed, t_after=t_after), repeats)
    return "h = 1 ms", t, Z, None, seconds, memory


METHODS = {
    "Euler, fixed step": _fixed_step(euler_step, False, EULER_STEPS),
    "RK4, fixed step": _fixed_step(rk4_step, False, STEPS),
    CHOSEN: _fixed_step(rk4_step, True, STEPS),
    "RK45 adaptive": _adaptive(1e-3, False),
    "RK45 adaptive, tight tolerance": _adaptive(1e-6, False),
    "RK45 adaptive, restarted at edges": _adaptive(1e-3, True),
    "Linear model, exact (lsim)": _linear,
}


def compare_methods(car, road, speed, repeats=5, t_after=2.0):
    """Run every method in :data:`METHODS` on one road and speed, and measure it.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    repeats : int, optional
        Timed runs per method; the fastest is kept. Default 5.
    t_after : float, optional
        Time simulated after the road feature [s]. Default 2.0.

    Returns
    -------
    list of dict
        One entry per method with ``method``, ``setting``, ``error`` (largest
        error in body acceleration and tyre force as a fraction of their
        peaks), ``passed`` (error within 1%), ``evals`` (right-hand-side
        evaluations, None for lsim), ``seconds`` (run time) and ``memory``
        (peak memory [bytes]).
    """
    reference = reference_solution(car, road, speed, t_after=t_after)
    rows = []
    with np.errstate(over="ignore", invalid="ignore"):       # an unstable run may overflow
        for method, run in METHODS.items():
            setting, t, Z, evals, seconds, memory = run(car, road, speed, reference, repeats, t_after)
            error, _ = run_errors(car, t, Z, reference)
            error = error if np.isfinite(error) else float("inf")
            rows.append({"method": method, "setting": setting, "error": error, "passed": error <= TOL,
                         "evals": evals, "seconds": seconds, "memory": memory})
    return rows


def comparison_table(rows):
    """Text table of the results of :func:`compare_methods`, for printing.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`compare_methods`.

    Returns
    -------
    str
        One line per method with its setting, error, run time and memory.
    """
    lines = [f"  {'Method':35s} {'Setting':26s} {'Error':>9s} {'Run time':>10s} {'Memory':>9s}"]
    for row in rows:
        note = "" if row["passed"] else "  misses 1%"
        lines.append(f"  {row['method']:35s} {row['setting']:26s} {100 * row['error']:8.2f}% "
                     f"{1e3 * row['seconds']:7.1f} ms {row['memory'] / 1024:6.0f} kB{note}")
    return "\n".join(lines)


def plot_comparison(rows, ax, title):
    """Draw a bar chart of run time for each method, labelled with its error and memory.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`compare_methods`.
    ax : matplotlib Axes
        Where to draw.
    title : str
        Title for the chart.
    """
    names = [row["method"] for row in rows]
    times = np.array([1e3 * row["seconds"] for row in rows])
    colours = [HIGHLIGHT if name == CHOSEN else OTHER for name in names]
    ax.barh(names, times, color=colours, height=0.6)
    for i, row in enumerate(rows):
        error = "unstable" if not np.isfinite(row["error"]) else f"error {100 * row['error']:.2g}%"
        label = f"{times[i]:.0f} ms, {row['memory'] / 1024:.0f} kB, {error}"
        ax.text(times[i] * 1.08, i, label + ("" if row["passed"] else " (misses 1%)"),
                va="center", fontsize=8, color=INK if row["passed"] else WARNING)
    ax.set_xscale("log")
    ax.set_xlim(max(0.1, 0.5 * times.min()), times.max() * 12)
    ax.set_ylim(len(rows) - 0.5, -0.5)                        # first method at the top
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("Run time [ms], log scale")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=SECONDARY, labelsize=8)


def comparison_figure(rows, title):
    """Make a window showing :func:`plot_comparison` for one run.

    The window appears when ``plt.show()`` is next called (for example by the
    animation), so both windows open together.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`compare_methods`.
    title : str
        Description of the run, used in the title.

    Returns
    -------
    matplotlib Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))
    plot_comparison(rows, ax, f"Solution methods compared: {title}")
    fig.text(0.01, 0.01, "Each method at the largest step or tolerance setting that keeps the error under "
             "1% where it has one; run time is the fastest of several runs. The animation uses 1 ms steps.",
             fontsize=8, color=SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
