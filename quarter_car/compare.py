"""Tools for comparing the chosen solver with alternative approaches.

* :func:`linear_solve` solves the linear state-space model exactly with
  ``scipy.signal.lsim``. It is fast and needs no time step study, but it
  cannot model the wheel leaving the road, because lift-off makes the
  equations non-linear.
* :func:`measure` finds the run time and peak memory of a calculation, so
  methods can be compared on cost as well as accuracy (notebook 2.1).
"""

import time
import tracemalloc

import numpy as np
from scipy.signal import lsim

from quarter_car.model import state_matrices
from quarter_car.scenarios import road_input


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
