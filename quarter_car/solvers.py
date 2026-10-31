"""Fixed-step time integrators for first-order ODE systems ``dz/dt = f(t, z)``.

Each step function advances the whole state vector by one step and returns
the new state together with the number of right-hand-side evaluations it used,
so that methods can be compared on cost as well as accuracy.

Explicit methods such as these are only stable when the step size is below a
limit set by the fastest mode of the system (for the quarter car, wheel hop).

:func:`simulate` uses one fixed step for the whole run.
:func:`simulate_in_segments` also restarts at known sudden changes in the
input, such as the edges of a pothole, so that no step crosses one.
"""

import numpy as np


def euler_step(f, t, z, h):
    """Advance the state by one explicit Euler step.

    Uses only the slope at the start of the step::

        z_new = z + h * f(t, z)

    Local error is O(h^2) and global error is O(h).

    Parameters
    ----------
    f : callable
        Right-hand side ``f(t, z)`` returning ``dz/dt`` with the same shape as ``z``.
    t : float
        Time at the start of the step [s].
    z : ndarray
        State at the start of the step.
    h : float
        Step size [s].

    Returns
    -------
    z_new : ndarray
        State at time ``t + h``.
    n_evals : int
        Number of right-hand-side evaluations used (always 1).
    """
    return z + h * f(t, z), 1


def rk4_step(f, t, z, h):
    """Advance the state by one classical fourth-order Runge-Kutta step.

    Samples the slope four times (at the start, twice at the midpoint, and at
    the end of the step) and combines them with weights 1, 2, 2, 1::

        k1 = f(t, z)
        k2 = f(t + h/2, z + h/2 * k1)
        k3 = f(t + h/2, z + h/2 * k2)
        k4 = f(t + h, z + h * k3)
        z_new = z + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    Local error is O(h^5) and global error is O(h^4).

    Parameters
    ----------
    f : callable
        Right-hand side ``f(t, z)`` returning ``dz/dt`` with the same shape as ``z``.
    t : float
        Time at the start of the step [s].
    z : ndarray
        State at the start of the step.
    h : float
        Step size [s].

    Returns
    -------
    z_new : ndarray
        State at time ``t + h``.
    n_evals : int
        Number of right-hand-side evaluations used (always 4).
    """
    k1 = f(t, z)
    k2 = f(t + h / 2, z + h / 2 * k1)
    k3 = f(t + h / 2, z + h / 2 * k2)
    k4 = f(t + h, z + h * k3)
    return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 4


def simulate(step, f, z0, t_end, h, t0=0.0):
    """Integrate ``dz/dt = f(t, z)`` from ``t0`` to ``t_end`` with a fixed step.

    If ``t_end - t0`` is not a whole number of steps, the final step is
    shortened so the solution finishes exactly at ``t_end``.

    Parameters
    ----------
    step : callable
        A step function such as :func:`euler_step` or :func:`rk4_step`.
    f : callable
        Right-hand side ``f(t, z)``, for example from ``model.make_rhs``.
    z0 : array_like
        Initial state at ``t0``.
    t_end : float
        Final time [s], must be greater than ``t0``.
    h : float
        Step size [s], must be positive.
    t0 : float, optional
        Start time [s]. Default is 0.

    Returns
    -------
    t : ndarray, shape (n + 1,)
        Times of the accepted steps [s], from ``t0`` to ``t_end``.
    Z : ndarray, shape (n + 1, len(z0))
        State at each time; row ``i`` is the state at ``t[i]``.
    n_evals : int
        Total number of right-hand-side evaluations, a measure of cost.

    Raises
    ------
    ValueError
        If ``h`` is not positive and finite, or ``t_end`` is not after ``t0``.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be a positive finite number, got {h}")
    if not np.isfinite(t_end) or t_end <= t0:
        raise ValueError(f"t_end must be greater than t0, got t0={t0}, t_end={t_end}")

    z0 = np.asarray(z0, dtype=float)
    n_steps = int(np.ceil((t_end - t0) / h - 1e-9))   # tolerance absorbs round-off
    t = t0 + h * np.arange(n_steps + 1)
    t[-1] = t_end                                     # final step may be shorter

    Z = np.empty((n_steps + 1, z0.size))
    Z[0] = z0
    n_evals = 0
    for i in range(n_steps):
        Z[i + 1], calls = step(f, t[i], Z[i], t[i + 1] - t[i])
        n_evals += calls
    return t, Z, n_evals


def simulate_in_segments(step, f, z0, t_end, h, breaks, t0=0.0):
    """Integrate with a fixed step, restarting the solver at each break time.

    A step that crosses a sudden change in the input (such as the edge of a
    pothole) samples the input on both sides of the change, so it loses most
    of its accuracy. Splitting the run at the times of the sudden changes
    means no step ever crosses one. Each segment is integrated separately
    with :func:`simulate`, starting from the state where the previous segment
    finished.

    At a break time the input has two values, one either side of the change.
    Inside each segment the time passed to ``f`` is kept just inside the
    segment, so every segment only sees the input on its own side.

    Parameters
    ----------
    step : callable
        A step function such as :func:`euler_step` or :func:`rk4_step`.
    f : callable
        Right-hand side ``f(t, z)``. Its time dependence must come only from
        the input (as in ``model.make_rhs``).
    z0 : array_like
        Initial state at ``t0``.
    t_end : float
        Final time [s], must be greater than ``t0``.
    h : float
        Step size [s], must be positive. The last step of each segment is
        shortened so the segment ends exactly at its break time.
    breaks : sequence of float
        Times [s] where the input changes suddenly. Times outside
        ``(t0, t_end)`` are ignored.
    t0 : float, optional
        Start time [s]. Default is 0.

    Returns
    -------
    t : ndarray
        Times of the accepted steps [s]. Every break time is included once.
    Z : ndarray, shape (len(t), len(z0))
        State at each time.
    n_evals : int
        Total number of right-hand-side evaluations.

    Raises
    ------
    ValueError
        If ``h`` is not positive and finite, or ``t_end`` is not after ``t0``.
    """
    if not np.isfinite(t_end) or t_end <= t0:
        raise ValueError(f"t_end must be greater than t0, got t0={t0}, t_end={t_end}")
    inner = [b for b in np.unique(breaks) if t0 < b < t_end]
    times = [t0, *inner, t_end]

    t_parts, Z_parts = [np.array([t0])], [np.asarray(z0, dtype=float)[None, :]]
    n_evals = 0
    for start, end in zip(times[:-1], times[1:]):
        margin = 1e-9 * (end - start)      # tiny, so the input on this side of each break is used

        def f_inside(t, z, start=start, end=end, margin=margin):
            return f(min(max(t, start + margin), end - margin), z)

        t, Z, calls = simulate(step, f_inside, Z_parts[-1][-1], end, h, t0=start)
        t_parts.append(t[1:])              # t[0] is the end of the previous segment
        Z_parts.append(Z[1:])
        n_evals += calls
    return np.concatenate(t_parts), np.vstack(Z_parts), n_evals
