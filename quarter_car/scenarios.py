"""Standard load cases: run the quarter car over a road input and post-process it."""

import numpy as np

from quarter_car.model import make_rhs
from quarter_car.road import half_sine_bump
from quarter_car.solvers import rk4_step, simulate

G = 9.81  # gravitational acceleration [m/s^2]


def simulate_bump(car, height, length, speed, lead_in_time=0.3, t_after=2.0,
                  h=1e-3, step=rk4_step):
    """Simulate the car driving over a half-sine bump at constant speed.

    The car starts at rest in static equilibrium on a flat road, reaches the
    bump after ``lead_in_time`` and is simulated for ``t_after`` once it has
    left the bump, so the body has time to settle.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    height : float
        Peak bump height [m].
    length : float
        Bump length along the road [m].
    speed : float
        Vehicle speed [m/s].
    lead_in_time : float, optional
        Time before the tyre reaches the bump [s]. Default 0.3.
    t_after : float, optional
        Time simulated after the tyre leaves the bump [s]. Default 2.0.
    h : float, optional
        Solver step size [s]. Default 1 ms.
    step : callable, optional
        Step function from ``quarter_car.solvers``. Default RK4.

    Returns
    -------
    t : ndarray, shape (n + 1,)
        Times [s].
    Z : ndarray, shape (n + 1, 4)
        States ``[x, y, x_dot, y_dot]`` at each time.
    r : ndarray, shape (n + 1,)
        Road height under the tyre at each time [m].

    Raises
    ------
    ValueError
        If ``lead_in_time`` is negative, ``t_after`` is not positive, or the
        bump or solver inputs are invalid.
    """
    if not np.isfinite(lead_in_time) or lead_in_time < 0:
        raise ValueError(f"lead_in_time must be >= 0, got {lead_in_time}")
    if not np.isfinite(t_after) or t_after <= 0:
        raise ValueError(f"t_after must be > 0, got {t_after}")

    road = half_sine_bump(height, length, speed, start_distance=speed * lead_in_time)
    t_end = lead_in_time + length / speed + t_after
    t, Z, _ = simulate(step, make_rhs(car, road), np.zeros(4), t_end, h)
    return t, Z, road(t)


def tyre_force(car, Z, r):
    """Total vertical force between the tyre and the road.

    Static load plus the dynamic tyre spring force,
    ``(m1 + m2) g + Kw (r - x)``. A negative value means the wheel would
    leave the road, which the linear model cannot represent, so results
    beyond that point are not valid.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    Z : ndarray, shape (n, 4)
        States ``[x, y, x_dot, y_dot]``.
    r : ndarray, shape (n,)
        Road height under the tyre [m].

    Returns
    -------
    ndarray, shape (n,)
        Tyre contact force [N], positive when the tyre presses on the road.
    """
    return (car.m1 + car.m2) * G + car.Kw * (r - Z[:, 0])
