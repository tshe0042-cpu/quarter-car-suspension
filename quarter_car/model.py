"""Quarter-car suspension model: parameters, state-space matrices and right-hand side.

The state vector uses the same order as the hand workings::

    z = [x, y, x_dot, y_dot]

where ``x`` is the wheel (unsprung mass ``m1``) displacement [m] and ``y`` is the
body (sprung mass ``m2``) displacement [m]. Upward is positive.

Assumptions
-----------
* Displacements are measured from static equilibrium, so gravity and the static
  spring preload cancel and do not appear in the equations while the tyre is
  on the road.
* The suspension spring and damper are linear, with no bump stops or travel
  limits.
* The tyre is a linear spring with no damping. It can push on the road but not
  pull on it, so if the road drops away faster than the wheel can follow (for
  example at the edge of a pothole) the tyre force falls to zero and the wheel
  leaves the road until it lands again. While the tyre is on the road the
  model is linear, so the state-space matrices and transfer function apply.
* Passive suspension only (no actuator force).
"""

from dataclasses import dataclass

import numpy as np

G = 9.81  # gravitational acceleration [m/s^2]


@dataclass(frozen=True)
class QuarterCar:
    """Physical parameters of a passive quarter-car model.

    Parameters
    ----------
    m1 : float
        Unsprung mass (wheel, tyre, hub and brake) [kg].
    m2 : float
        Sprung mass, i.e. the body mass carried by one wheel [kg].
    Ks : float
        Suspension spring stiffness [N/m].
    b : float
        Damper coefficient [N s/m].
    Kw : float
        Tyre vertical stiffness [N/m].

    Raises
    ------
    ValueError
        If a mass or stiffness is not a positive finite number, or if ``b``
        is negative or not finite.
    """

    m1: float
    m2: float
    Ks: float
    b: float
    Kw: float

    def __post_init__(self):
        for name in ("m1", "m2", "Ks", "Kw"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {value}")
        if not np.isfinite(self.b) or self.b < 0:
            raise ValueError(f"b must be a finite number >= 0, got {self.b}")


def damping_from_ratio(zeta, Ks, m2):
    """Convert a damping ratio into a damper coefficient.

    Uses the body-only (single degree of freedom) definition
    ``b = 2 * zeta * sqrt(Ks * m2)``. The true modal damping of the two-mass
    system differs slightly; use :func:`natural_modes` to see the actual values.

    Parameters
    ----------
    zeta : float
        Damping ratio [-], must be >= 0.
    Ks : float
        Suspension spring stiffness [N/m].
    m2 : float
        Sprung (body) mass [kg].

    Returns
    -------
    float
        Damper coefficient ``b`` [N s/m].

    Raises
    ------
    ValueError
        If ``zeta`` is negative or not finite.
    """
    if not np.isfinite(zeta) or zeta < 0:
        raise ValueError(f"zeta must be a finite number >= 0, got {zeta}")
    return 2.0 * zeta * np.sqrt(Ks * m2)


def state_matrices(car):
    """Build the state-space matrices for ``z_dot = A z + B r`` and ``y = C z``.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.

    Returns
    -------
    A : ndarray, shape (4, 4)
        State matrix [1/s and 1/s^2 entries].
    B : ndarray, shape (4,)
        Road input vector; only the wheel acceleration row is non-zero.
    C : ndarray, shape (4,)
        Output row vector that selects the body displacement ``y``.
    """
    m1, m2, Ks, b, Kw = car.m1, car.m2, car.Ks, car.b, car.Kw
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],                       # d(x)/dt = x_dot
        [0.0, 0.0, 0.0, 1.0],                       # d(y)/dt = y_dot
        [-(Ks + Kw) / m1, Ks / m1, -b / m1, b / m1],  # wheel acceleration
        [Ks / m2, -Ks / m2, b / m2, -b / m2],         # body acceleration
    ])
    B = np.array([0.0, 0.0, Kw / m1, 0.0])
    C = np.array([0.0, 1.0, 0.0, 0.0])
    return A, B, C


def make_rhs(car, road, lift_off=True):
    """Create the right-hand side function ``f(t, z)`` used by every solver.

    The equations of motion are those from the free-body diagrams::

        m1 x_ddot = Ks (y - x) + b (y_dot - x_dot) + tyre force change
        m2 y_ddot = -(Ks (y - x) + b (y_dot - x_dot))

    While the tyre is on the road the tyre force change is ``Kw (r - x)`` and
    the model is linear, ``dz/dt = A z + B r`` (a test checks the two forms
    agree). The total tyre force is the static load plus this change,
    ``(m1 + m2) g + Kw (r - x)``. A tyre can push on the road but cannot pull
    on it, so when the total would be negative the wheel has left the road
    and the tyre force is set to zero instead.

    The equations are written out term by term with plain Python numbers
    rather than as the matrix product ``A @ z``: for a 4 x 4 system the time
    is set by the overhead of each NumPy call, not by the arithmetic, so this
    is several times faster (see ``performance_study.py``).

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : callable
        ``road(t)`` returns the road height ``r`` [m] at time ``t`` [s]. It must
        accept any time, because solvers such as RK4 evaluate half steps.
    lift_off : bool, optional
        If True (default), the wheel can leave the road as described above.
        If False, the tyre is treated as stuck to the road, which gives the
        purely linear model of :func:`state_matrices` and
        :func:`transfer_function`.

    Returns
    -------
    callable
        ``f(t, z)`` returning ``dz/dt`` as an array of shape (4,). This matches
        the ``fun(t, y)`` contract of ``scipy.integrate.solve_ivp``.
    """
    m1, m2, Ks, b, Kw = car.m1, car.m2, car.Ks, car.b, car.Kw
    static_load = (m1 + m2) * G                  # tyre force when the car is at rest [N]

    def rhs(t, z):
        x, y, x_dot, y_dot = np.asarray(z, dtype=float).tolist()   # plain floats are faster here
        suspension = Ks * (y - x) + b * (y_dot - x_dot)   # spring and damper force on the wheel [N]
        tyre = Kw * (road(t) - x)                         # change in tyre force from rest [N]
        if lift_off and tyre < -static_load:              # total tyre force would be negative,
            tyre = -static_load                           # so the wheel is in the air: force is zero
        return np.array([x_dot, y_dot, (suspension + tyre) / m1, -suspension / m2])

    return rhs


def transfer_function(car):
    """Road-to-body transfer function ``Y(s)/R(s) = num(s) / den(s)``.

    Derived independently from the equations of motion by Laplace transform, so
    it can be used to check the state-space model.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.

    Returns
    -------
    num : ndarray
        Numerator coefficients, highest power of ``s`` first.
    den : ndarray
        Denominator coefficients ``D(s)``, highest power of ``s`` first.
    """
    m1, m2, Ks, b, Kw = car.m1, car.m2, car.Ks, car.b, car.Kw
    num = np.array([Kw * b, Kw * Ks])
    den = np.array([
        m1 * m2,
        (m1 + m2) * b,
        m2 * (Ks + Kw) + m1 * Ks,
        Kw * b,
        Ks * Kw,
    ])
    return num, den


def natural_modes(car):
    """Natural frequencies and damping ratios of the two modes, from ``eig(A)``.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.

    Returns
    -------
    freqs_hz : ndarray, shape (2,)
        Undamped natural frequency of each mode [Hz], lowest first
        (body bounce, then wheel hop).
    zetas : ndarray, shape (2,)
        Modal damping ratio of each mode [-], in the same order.

    Raises
    ------
    ValueError
        If a mode is overdamped (real eigenvalues), because a single
        frequency and damping ratio are not defined for it in this form.
    """
    A, _, _ = state_matrices(car)
    eigenvalues = np.linalg.eigvals(A)
    upper = eigenvalues[eigenvalues.imag > 0]   # one eigenvalue from each complex pair
    if upper.size != 2:
        raise ValueError("Expected two underdamped modes, but at least one mode is overdamped.")
    omega_n = np.abs(upper)
    zetas = -upper.real / omega_n
    order = np.argsort(omega_n)
    return omega_n[order] / (2.0 * np.pi), zetas[order]
