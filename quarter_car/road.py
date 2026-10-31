"""Road surfaces for the quarter car to drive over.

Each road gives its height above the flat road [m] as a function of distance
along the road [m], measured from where the road feature starts. The road is
flat before the feature (negative distance) and after it.

Two kinds of road cover smooth and sudden changes in the surface:

* :class:`Bump`: a half-sine bump such as a speed hump. The height changes
  smoothly; only the slope changes suddenly, at each end.
* :class:`Pothole`: a hole with vertical edges, so the height changes suddenly
  at each edge.

Both roads have the same interface: ``profile(distance)`` gives the height,
``length`` is the length of the feature, ``edges`` lists the distances where
the surface changes suddenly, and ``label`` describes the road in words.

``profile`` has a fast path for a single number. The solvers ask for the road
height one time at a time (thousands of times per run), and NumPy's array
machinery is slow for single numbers, so plain Python arithmetic is used
then. Arrays (for plots and the animation) still use NumPy.

Assumption: the tyre touches the road at a single point and follows the
profile exactly. Real tyres touch the road over a contact patch about 0.15 to
0.2 m long, which smooths out shorter features, so short or sharp features
(such as pothole edges) are harsher in the model than in reality.
"""

import math
from dataclasses import dataclass

import numpy as np


def _check_positive(**values):
    """Raise ValueError unless every keyword value is a positive finite number."""
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value}")


@dataclass(frozen=True)
class Bump:
    """A half-sine bump, such as a speed hump or a speed bump.

    The height is ``height * sin(pi * s / length)`` for ``0 <= s <= length``
    and zero elsewhere, where ``s`` is the distance from the start of the bump.
    The height changes smoothly, but the slope changes suddenly at both ends.

    Parameters
    ----------
    height : float
        Peak height [m], must be > 0.
    length : float
        Length along the road [m], must be > 0.

    Raises
    ------
    ValueError
        If ``height`` or ``length`` is not a positive finite number.
    """

    height: float
    length: float

    def __post_init__(self):
        _check_positive(height=self.height, length=self.length)

    @property
    def edges(self):
        """Distances [m] where the surface changes suddenly: the slope, at each end."""
        return (0.0, self.length)

    @property
    def label(self):
        """Description for titles, e.g. '75 mm high, 3.7 m long bump'."""
        return f"{1e3 * self.height:.0f} mm high, {self.length:g} m long bump"

    def profile(self, distance):
        """Road height at the given distance from the start of the bump.

        Parameters
        ----------
        distance : float or array_like
            Distance along the road from the start of the bump [m].

        Returns
        -------
        float or ndarray
            Road height [m]; a float if ``distance`` is a single number.
        """
        if np.isscalar(distance):                 # fast path for one number (used by the solvers)
            if 0.0 <= distance <= self.length:
                return self.height * math.sin(math.pi * distance / self.length)
            return 0.0
        s = np.asarray(distance, dtype=float)
        on_bump = (s >= 0.0) & (s <= self.length)
        r = np.where(on_bump, self.height * np.sin(np.pi * s / self.length), 0.0)
        return r if r.ndim else float(r)


@dataclass(frozen=True)
class Pothole:
    """A pothole with vertical edges.

    The road is ``depth`` lower for ``0 <= s < length`` and flat elsewhere, so
    the height changes suddenly at both edges: the worst case for a solver
    with a fixed time step.

    Parameters
    ----------
    depth : float
        Depth below the road surface [m], must be > 0.
    length : float
        Length along the road [m], must be > 0.

    Raises
    ------
    ValueError
        If ``depth`` or ``length`` is not a positive finite number.
    """

    depth: float
    length: float

    def __post_init__(self):
        _check_positive(depth=self.depth, length=self.length)

    @property
    def edges(self):
        """Distances [m] where the surface changes suddenly: the two edges."""
        return (0.0, self.length)

    @property
    def label(self):
        """Description for titles, e.g. '50 mm deep, 0.6 m long pothole'."""
        return f"{1e3 * self.depth:.0f} mm deep, {self.length:g} m long pothole"

    def profile(self, distance):
        """Road height at the given distance from the near edge of the pothole.

        Parameters
        ----------
        distance : float or array_like
            Distance along the road from the near edge [m].

        Returns
        -------
        float or ndarray
            Road height [m] (negative inside the pothole); a float if
            ``distance`` is a single number.
        """
        if np.isscalar(distance):                 # fast path for one number (used by the solvers)
            return -self.depth if 0.0 <= distance < self.length else 0.0
        s = np.asarray(distance, dtype=float)
        r = np.where((s >= 0.0) & (s < self.length), -self.depth, 0.0)
        return r if r.ndim else float(r)
