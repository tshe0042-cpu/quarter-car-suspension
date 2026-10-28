"""Road profiles: road height as a function of position along the road, or of time.

Road heights are in metres and measured upward from the flat road surface.
The tyre is treated as a point contact, so it follows the profile exactly. Real
tyres partly "envelope" very short bumps, so this over-predicts the input for
bumps shorter than the tyre contact patch (about 0.15 to 0.2 m).
"""

import numpy as np


def half_sine_profile(position, height, length, start):
    """Road height of a half-sine bump at the given positions along the road.

    On the bump the height is ``height * sin(pi * s / length)``, where ``s`` is
    the distance past the start of the bump. Off the bump the road is flat (0).

    Parameters
    ----------
    position : float or array_like
        Position along the road [m].
    height : float
        Peak height of the bump [m].
    length : float
        Length of the bump along the road [m].
    start : float
        Position where the bump begins [m].

    Returns
    -------
    float or ndarray
        Road height [m] at each position (a float if ``position`` is a scalar).
    """
    s = np.asarray(position, dtype=float) - start
    on_bump = (s >= 0.0) & (s <= length)
    r = np.where(on_bump, height * np.sin(np.pi * s / length), 0.0)
    return r if r.ndim else float(r)


def half_sine_bump(height, length, speed, start_distance=0.0):
    """Create the road input ``road(t)`` for a car crossing a half-sine bump.

    The car is at position 0 at ``t = 0`` and travels at a constant speed, so
    the road height under the tyre at time ``t`` is the profile at ``speed * t``.
    The bump's start and end are kinks (sudden changes in slope), which lower
    the observed order of fixed-step solvers unless steps land on them.

    Parameters
    ----------
    height : float
        Peak height of the bump [m], must be >= 0.
    length : float
        Length of the bump along the road [m], must be > 0.
    speed : float
        Vehicle speed [m/s], must be > 0. Divide km/h by 3.6.
    start_distance : float, optional
        Distance from the car's starting position to the start of the bump [m].
        Default is 0.

    Returns
    -------
    callable
        ``road(t)`` giving the road height [m] under the tyre at time ``t`` [s].
        Accepts a float or an array of times.

    Raises
    ------
    ValueError
        If any input is outside the ranges above or is not finite.
    """
    checks = {"height": (height, height >= 0), "length": (length, length > 0),
              "speed": (speed, speed > 0), "start_distance": (start_distance, start_distance >= 0)}
    for name, (value, ok) in checks.items():
        if not np.isfinite(value) or not ok:
            raise ValueError(f"{name} is outside its allowed range, got {value}")

    def road(t):
        return half_sine_profile(speed * np.asarray(t, dtype=float), height, length, start_distance)

    return road
