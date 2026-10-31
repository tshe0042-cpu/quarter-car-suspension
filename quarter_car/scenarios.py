"""Run the quarter car over a road at constant speed and work out the key results."""

import numpy as np

from quarter_car.model import G, make_rhs
from quarter_car.solvers import rk4_step, simulate, simulate_in_segments


def road_input(road, speed, lead_in_time=0.3):
    """Turn a road profile into the road input ``r(t)`` seen by the tyre.

    The car travels at a constant speed and reaches the start of the road
    feature at ``t = lead_in_time``, so at time ``t`` the tyre is at distance
    ``speed * (t - lead_in_time)`` from the start of the feature.

    Parameters
    ----------
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s], must be > 0.
    lead_in_time : float, optional
        Time before the tyre reaches the feature [s]. Default 0.3.

    Returns
    -------
    callable
        ``r(t)`` giving the road height [m] under the tyre at time ``t`` [s].
        Accepts a float or an array of times.

    Raises
    ------
    ValueError
        If ``speed`` is not positive or ``lead_in_time`` is negative.
    """
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError(f"speed must be a positive finite number, got {speed}")
    if not np.isfinite(lead_in_time) or lead_in_time < 0:
        raise ValueError(f"lead_in_time must be >= 0, got {lead_in_time}")

    def r(t):
        return road.profile(speed * (np.asarray(t, dtype=float) - lead_in_time))

    return r


def edge_times(road, speed, lead_in_time=0.3):
    """Times [s] when the tyre reaches each sudden change in the road surface.

    Parameters
    ----------
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    lead_in_time : float, optional
        Time before the tyre reaches the feature [s]. Default 0.3.

    Returns
    -------
    list of float
        One time for each entry in ``road.edges``.
    """
    return [lead_in_time + edge / speed for edge in road.edges]


def simulate_road(car, road, speed, h=1e-3, step=rk4_step, lead_in_time=0.3, t_after=2.0,
                  split_at_edges=True, lift_off=True):
    """Simulate the car driving over a road at constant speed.

    The car starts at rest on a flat road, reaches the road feature after
    ``lead_in_time`` and is simulated for ``t_after`` once it has left it, so
    the body has time to settle.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s]. Divide km/h by 3.6.
    h : float, optional
        Solver step size [s]. Default 1 ms.
    step : callable, optional
        Step function from ``quarter_car.solvers``. Default RK4.
    lead_in_time : float, optional
        Time before the tyre reaches the road feature [s]. Default 0.3.
    t_after : float, optional
        Time simulated after the tyre leaves the feature [s]. Default 2.0.
    split_at_edges : bool, optional
        If True (default), restart the solver at every sudden change in the
        road (see ``solvers.simulate_in_segments``), so no step crosses one.
        ``time_step_study.py`` shows this lets a pothole run use a step about
        10 times larger for the same accuracy. False uses one fixed step for
        the whole run.
    lift_off : bool, optional
        Whether the wheel can leave the road (see ``model.make_rhs``). Default True.

    Returns
    -------
    t : ndarray, shape (n,)
        Times [s].
    Z : ndarray, shape (n, 4)
        States ``[x, y, x_dot, y_dot]`` at each time.
    r : ndarray, shape (n,)
        Road height under the tyre at each time [m].

    Raises
    ------
    ValueError
        If ``speed``, ``h`` or ``t_after`` is not positive, or ``lead_in_time``
        is negative.
    """
    if not np.isfinite(t_after) or t_after <= 0:
        raise ValueError(f"t_after must be > 0, got {t_after}")
    r = road_input(road, speed, lead_in_time)
    rhs = make_rhs(car, r, lift_off)
    t_end = lead_in_time + road.length / speed + t_after
    if split_at_edges:
        breaks = edge_times(road, speed, lead_in_time)
        t, Z, _ = simulate_in_segments(step, rhs, np.zeros(4), t_end, h, breaks)
    else:
        t, Z, _ = simulate(step, rhs, np.zeros(4), t_end, h)
    return t, Z, r(t)


def tyre_force(car, Z, r):
    """Total vertical force between the tyre and the road.

    Static load plus the tyre spring force, ``(m1 + m2) g + Kw (r - x)``,
    but never below zero, because the tyre cannot pull on the road. Zero means
    the wheel is in the air.

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
        Tyre contact force [N].
    """
    return np.maximum((car.m1 + car.m2) * G + car.Kw * (r - Z[:, 0]), 0.0)


def body_acceleration(car, Z):
    """Body (sprung mass) acceleration, the main ride comfort measure.

    Calculated from the body's equation of motion,
    ``y_ddot = (Ks (x - y) + b (x_dot - y_dot)) / m2``, so no numerical
    differentiation of the displacement is needed. The body equation does not
    involve the tyre, so this holds whether or not the wheel is on the road.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    Z : ndarray, shape (n, 4)
        States ``[x, y, x_dot, y_dot]``.

    Returns
    -------
    ndarray, shape (n,)
        Body acceleration [m/s^2], positive upward.
    """
    x, y, x_dot, y_dot = Z.T
    return (car.Ks * (x - y) + car.b * (x_dot - y_dot)) / car.m2


def suspension_travel(Z):
    """Suspension deflection ``y - x``, the change in spring length from rest.

    Parameters
    ----------
    Z : ndarray, shape (n, 4)
        States ``[x, y, x_dot, y_dot]``.

    Returns
    -------
    ndarray, shape (n,)
        Suspension deflection [m], positive when the suspension extends.
    """
    return Z[:, 1] - Z[:, 0]


def road_summary(car, t, Z, r):
    """Key comfort and safety results of a run.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    t : ndarray, shape (n,)
        Times [s].
    Z : ndarray, shape (n, 4)
        States ``[x, y, x_dot, y_dot]``.
    r : ndarray, shape (n,)
        Road height under the tyre [m].

    Returns
    -------
    dict
        ``peak_body_displacement`` [m], ``peak_body_acceleration`` [m/s^2],
        ``peak_suspension_travel`` [m]
        (the peaks are the largest size, up or down), ``min_tyre_force`` [N],
        ``max_tyre_force`` [N], ``lift_off_time`` [s] (first time the wheel
        leaves the road, None if it never does) and ``airborne_time`` [s]
        (total time in the air).
    """
    force = tyre_force(car, Z, r)
    accel = body_acceleration(car, Z)
    airborne = force <= 0.0
    return {
        "peak_body_displacement": float(np.abs(Z[:, 1]).max()),
        "peak_body_acceleration": float(np.abs(accel).max()),
        "peak_suspension_travel": float(np.abs(suspension_travel(Z)).max()),
        "min_tyre_force": float(force.min()),
        "max_tyre_force": float(force.max()),
        "lift_off_time": float(t[np.argmax(airborne)]) if airborne.any() else None,
        "airborne_time": float(np.sum(np.diff(t)[airborne[:-1]])),
    }
