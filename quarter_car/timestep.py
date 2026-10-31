"""Tools for the time step study: how fine must the time step be?

Each run is compared with an accurate reference solution. The error is
measured in the outputs that matter, in two ways:

* history error: the largest difference in body acceleration (comfort) and
  tyre force (safety) at the run's own time steps, as a fraction of their
  peak values. This shows how accurately the solver follows the true motion.
* peak error: the largest relative error in the peak body acceleration,
  suspension travel and tyre force. A coarse step can miss a short peak even
  when the solver is accurate at the points it does calculate.

The reference uses SciPy's adaptive 8th-order Runge-Kutta solver (DOP853)
with a very tight tolerance, restarted at every sudden change in the road.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from quarter_car.model import G, make_rhs
from quarter_car.scenarios import (body_acceleration, edge_times, road_input, simulate_road,
                                   suspension_travel, tyre_force)
from quarter_car.solvers import euler_step, rk4_step

EVALS_PER_STEP = {euler_step: 1, rk4_step: 4}   # right-hand-side evaluations per step
STEPS = (10e-3, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4)   # time steps tried [s], coarse to fine


@dataclass(frozen=True)
class AdaptiveRun:
    """Result of :func:`adaptive_solve`.

    Attributes
    ----------
    t : ndarray
        Times of the steps the solver accepted [s].
    states : callable
        ``states(times)`` returns the states ``[x, y, x_dot, y_dot]`` at any
        times in the run, shape (len(times), 4), from the solver's dense output.
    n_evals : int
        Number of right-hand-side evaluations used.
    """

    t: np.ndarray
    states: object
    n_evals: int


def adaptive_solve(car, road, speed, method="RK45", rtol=1e-3, atol=1e-6, split_at_edges=False,
                   lead_in_time=0.3, t_after=2.0):
    """Simulate a road run with SciPy's adaptive ``solve_ivp``.

    The adaptive solver picks its own step sizes from an error estimate, but
    it only sees the road at the times where it evaluates the right-hand side.
    Without ``split_at_edges`` it can therefore step straight over a short
    pothole on an otherwise flat road.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    method : str, optional
        ``solve_ivp`` method, e.g. "RK45" (default) or "DOP853".
    rtol, atol : float, optional
        Relative and absolute tolerances. Defaults 1e-3 and 1e-6 (SciPy's defaults).
    split_at_edges : bool, optional
        If True, restart the solver at each sudden change in the road.
    lead_in_time, t_after : float, optional
        As for ``scenarios.simulate_road``.

    Returns
    -------
    AdaptiveRun
        Step times, a function for the states at any time, and the cost.
    """
    rhs = make_rhs(car, road_input(road, speed, lead_in_time))
    t_end = lead_in_time + road.length / speed + t_after
    breaks = edge_times(road, speed, lead_in_time) if split_at_edges else []
    times = [0.0, *sorted(b for b in breaks if 0.0 < b < t_end), t_end]

    pieces, t_steps, z, n_evals = [], [np.array([0.0])], np.zeros(4), 0
    for start, end in zip(times[:-1], times[1:]):
        margin = 1e-9 * (end - start)      # keep each segment on its own side of a break

        def f_inside(t, z, start=start, end=end, margin=margin):
            return rhs(min(max(t, start + margin), end - margin), z)

        sol = solve_ivp(f_inside, (start, end), z, method=method, rtol=rtol, atol=atol,
                        dense_output=True)
        pieces.append((start, end, sol.sol))
        t_steps.append(sol.t[1:])
        z, n_evals = sol.y[:, -1], n_evals + sol.nfev

    def states(at):
        at = np.asarray(at, dtype=float)
        out = np.empty((at.size, 4))
        for start, end, dense in pieces:
            inside = (at >= start) & (at <= end)
            if inside.any():
                out[inside] = dense(at[inside]).T
        return out

    return AdaptiveRun(np.concatenate(t_steps), states, n_evals)


@dataclass(frozen=True)
class Reference:
    """Accurate reference solution for one road and speed.

    Attributes
    ----------
    states : callable
        ``states(times)`` returns the reference states at any times, shape (n, 4).
    road : callable
        ``road(times)`` returns the road height under the tyre [m].
    peaks : ndarray, shape (3,)
        True peak body acceleration [m/s^2], suspension travel [m] and tyre
        force [N], found on a 10 microsecond grid.
    n_evals : int
        Cost of the reference solution.
    """

    states: object
    road: object
    peaks: np.ndarray
    n_evals: int


def peak_outputs(car, Z, r):
    """Peak body acceleration [m/s^2], suspension travel [m] and tyre force [N].

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    Z : ndarray, shape (n, 4)
        States at each time.
    r : ndarray, shape (n,)
        Road height under the tyre at each time [m].

    Returns
    -------
    ndarray, shape (3,)
        Largest size of the body acceleration and suspension travel, and the
        largest tyre force.
    """
    return np.array([np.abs(body_acceleration(car, Z)).max(),
                     np.abs(suspension_travel(Z)).max(),
                     tyre_force(car, Z, r).max()])


def reference_solution(car, road, speed, lead_in_time=0.3, t_after=2.0):
    """Accurate reference: DOP853 with ``rtol = 1e-11``, restarted at every edge.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    lead_in_time, t_after : float, optional
        As for ``scenarios.simulate_road``.

    Returns
    -------
    Reference
        Reference states, road input, true peaks and cost.
    """
    run = adaptive_solve(car, road, speed, method="DOP853", rtol=1e-11, atol=1e-13,
                         split_at_edges=True, lead_in_time=lead_in_time, t_after=t_after)
    r = road_input(road, speed, lead_in_time)
    fine = np.arange(0.0, run.t[-1], 1e-5)
    peaks = peak_outputs(car, run.states(fine), r(fine))
    return Reference(run.states, r, peaks, run.n_evals)


def run_errors(car, t, Z, reference):
    """Errors of a run compared with the reference solution.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    t : ndarray, shape (n,)
        Times of the run [s].
    Z : ndarray, shape (n, 4)
        States of the run at those times.
    reference : Reference
        From :func:`reference_solution` for the same road and speed.

    Returns
    -------
    history : float
        Largest error in body acceleration and tyre force at the run's own
        times, as a fraction of the peak body acceleration and the peak change
        in tyre force from the static load.
    peaks : float
        Largest relative error in the three peak outputs (see :func:`peak_outputs`).
    """
    Z_ref, r = reference.states(t), reference.road(t)
    accel, accel_ref = body_acceleration(car, Z), body_acceleration(car, Z_ref)
    force, force_ref = tyre_force(car, Z, r), tyre_force(car, Z_ref, r)
    static_load = (car.m1 + car.m2) * G
    history = max(np.abs(accel - accel_ref).max() / np.abs(accel_ref).max(),
                  np.abs(force - force_ref).max() / np.abs(force_ref - static_load).max())
    peaks = np.max(np.abs(peak_outputs(car, Z, r) - reference.peaks) / reference.peaks)
    return float(history), float(peaks)


def step_study(car, road, speed, steps=STEPS, step=rk4_step, split_at_edges=False, reference=None,
               t_after=2.0):
    """Errors and cost of fixed-step runs for a list of time steps.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    steps : sequence of float, optional
        Time steps to try [s]. Default :data:`STEPS`.
    step : callable, optional
        ``solvers.euler_step`` or ``solvers.rk4_step`` (default).
    split_at_edges : bool, optional
        Whether to restart the solver at each sudden change in the road.
    reference : Reference, optional
        Reuse an existing reference; one is calculated if not given.
    t_after : float, optional
        Time simulated after the road feature [s], as for
        ``scenarios.simulate_road``. Default 2.0.

    Returns
    -------
    dict
        ``steps`` [s], ``history`` and ``peaks`` errors (fractions, see
        :func:`run_errors`), ``error`` (the larger of the two) and
        ``n_evals`` (right-hand-side evaluations of each run), as arrays.
    """
    if reference is None:
        reference = reference_solution(car, road, speed, t_after=t_after)
    history, peaks, n_evals = [], [], []
    for h in steps:
        t, Z, _ = simulate_road(car, road, speed, h=h, step=step, split_at_edges=split_at_edges,
                                t_after=t_after)
        e_history, e_peaks = run_errors(car, t, Z, reference)
        history.append(e_history)
        peaks.append(e_peaks)
        n_evals.append((len(t) - 1) * EVALS_PER_STEP[step])
    history, peaks = np.array(history), np.array(peaks)
    return {"steps": np.array(steps), "history": history, "peaks": peaks,
            "error": np.maximum(history, peaks), "n_evals": np.array(n_evals)}


def required_step(result, tol=0.01, measure="history"):
    """Largest time step whose error, and that of every finer step tried, is within ``tol``.

    Parameters
    ----------
    result : dict
        Output of :func:`step_study`, with steps ordered coarse to fine.
    tol : float, optional
        Allowed error as a fraction. Default 0.01 (1%).
    measure : {"history", "peaks", "error"}, optional
        Which error to judge by (see :func:`step_study`). Default "history",
        the accuracy of the computed motion.

    Returns
    -------
    h : float or None
        Required time step [s], or None if even the finest step fails.
    n_evals : int or None
        Cost of the run at that step, or None.
    """
    ok = result[measure] <= tol
    if not ok[-1]:
        return None, None
    first = len(ok) - 1                       # start at the finest step...
    while first > 0 and ok[first - 1]:        # ...and move to coarser steps while they still pass
        first -= 1
    return float(result["steps"][first]), int(result["n_evals"][first])


def max_stable_step(car, method="rk4", h_max=0.1, n=100_000):
    """Largest stable fixed step for Euler or RK4 on the linear model.

    A step multiplies each mode by the method's stability function
    ``R(h * lambda)``, where ``lambda`` is an eigenvalue of ``A`` (notebook 7.4):
    ``R = 1 + z`` for Euler and ``R = 1 + z + z^2/2 + z^3/6 + z^4/24`` for RK4.
    The step is stable while ``|R| <= 1`` for every eigenvalue.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    method : {"euler", "rk4"}, optional
        Which method. Default "rk4".
    h_max : float, optional
        Largest step to check [s]. Default 0.1.
    n : int, optional
        Number of steps checked between 0 and ``h_max``. Default 100 000.

    Returns
    -------
    float
        Largest stable step [s] (to within ``h_max / n``).
    """
    from quarter_car.model import state_matrices
    eigenvalues = np.linalg.eigvals(state_matrices(car)[0])
    h = np.linspace(h_max / n, h_max, n)
    z = h[:, None] * eigenvalues[None, :]
    if method == "euler":
        R = 1 + z
    elif method == "rk4":
        R = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    else:
        raise ValueError(f"method must be 'euler' or 'rk4', got {method!r}")
    stable = np.all(np.abs(R) <= 1.0, axis=1)
    return float(h[np.argmin(stable) - 1]) if not stable.all() else h_max
