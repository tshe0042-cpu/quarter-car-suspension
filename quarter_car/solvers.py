"""Fixed-step time integrators for first-order ODE systems ``dz/dt = f(t, z)``.

Each step function advances the whole state vector by one step and returns
the new state together with the number of right-hand-side evaluations it used,
so that methods can be compared on cost as well as accuracy.

Explicit methods such as these are only stable when the step size is below a
limit set by the fastest mode of the system (for the quarter car, wheel hop).
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
