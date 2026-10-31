"""Tests for quarter_car.solvers. Run from the repo root with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.model import QuarterCar, make_rhs
from quarter_car.solvers import euler_step, rk4_step, simulate, simulate_in_segments

# 1-DOF mass-spring-damper from notebook 7.5, which has an exact solution.
M, K, C = 5.0, 200.0, 8.0          # [kg], [N/m], [N s/m]
X0, V0 = 0.05, 0.0                 # initial position [m] and velocity [m/s]


def one_dof_rhs(t, z):
    """dz/dt for z = [x, v] of the 1-DOF mass-spring-damper."""
    x, v = z
    return np.array([v, (-C * v - K * x) / M])


def one_dof_exact(t):
    """Exact underdamped displacement x(t) for the 1-DOF system above."""
    wn = np.sqrt(K / M)
    zeta = C / (2 * np.sqrt(K * M))
    wd = wn * np.sqrt(1 - zeta**2)
    return np.exp(-zeta * wn * t) * (X0 * np.cos(wd * t)
                                     + (V0 + zeta * wn * X0) / wd * np.sin(wd * t))


def max_position_error(step, h, t_end=2.0):
    """Largest displacement error over the run, compared with the exact solution."""
    t, Z, _ = simulate(step, one_dof_rhs, [X0, V0], t_end, h)
    return np.max(np.abs(Z[:, 0] - one_dof_exact(t)))


def test_euler_first_step_matches_hand_calculation():
    """Body lifted 50 mm, h = 5 ms: matches the worked step done by hand."""
    car = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)
    f = make_rhs(car, road=lambda t: 0.0)
    z0 = np.array([0.0, 0.05, 0.0, 0.0])       # [x, y, x_dot, y_dot]
    z1, n_evals = euler_step(f, 0.0, z0, 0.005)
    assert np.allclose(z1, [0.0, 0.05, 0.125, -0.05 * 20e3 / 300.0 * 0.005])
    assert n_evals == 1


def test_rk4_is_exact_for_a_cubic():
    """With f depending only on t, RK4 reduces to Simpson's rule, exact for cubics."""
    t, Z, _ = simulate(rk4_step, lambda t, z: np.array([t**3]), [0.0], 1.0, 0.25)
    assert Z[-1, 0] == pytest.approx(0.25, abs=1e-14)


@pytest.mark.parametrize("step, expected_order, h", [
    (euler_step, 1.0, 0.002),
    (rk4_step, 4.0, 0.02),
])
def test_observed_order_of_accuracy(step, expected_order, h):
    """Halving h should cut the error by 2^p, where p is the method's order."""
    p = np.log2(max_position_error(step, h) / max_position_error(step, h / 2))
    assert p == pytest.approx(expected_order, abs=0.15)


def test_simulate_finishes_exactly_at_t_end():
    """0.25 s is not a whole number of 0.1 s steps, so the last step is shortened."""
    t, Z, n_evals = simulate(rk4_step, one_dof_rhs, [X0, V0], 0.25, 0.1)
    assert t[-1] == 0.25
    assert len(t) == 4 and Z.shape == (4, 2)
    assert n_evals == 3 * 4                    # 3 steps x 4 evaluations


@pytest.mark.parametrize("h, t_end", [(0.0, 1.0), (-0.01, 1.0), (np.nan, 1.0), (0.01, 0.0)])
def test_simulate_rejects_bad_inputs(h, t_end):
    with pytest.raises(ValueError):
        simulate(euler_step, one_dof_rhs, [X0, V0], t_end, h)


def step_input_rhs(t, z):
    """dz/dt = 0 before t = 0.33 s and 1 after it: a sudden change in the input."""
    return np.array([1.0 if t >= 0.33 else 0.0])


def test_crossing_a_sudden_change_loses_accuracy():
    """Exact answer at t = 1 is 1 - 0.33 = 0.67. A fixed RK4 step crossing the change misses it."""
    _, Z, _ = simulate(rk4_step, step_input_rhs, [0.0], 1.0, 0.1)
    assert abs(Z[-1, 0] - 0.67) > 1e-3


def test_restarting_at_the_change_is_exact():
    """Restarting at t = 0.33 s means no step crosses the change, so RK4 is exact again."""
    t, Z, n_evals = simulate_in_segments(rk4_step, step_input_rhs, [0.0], 1.0, 0.1, breaks=[0.33])
    assert Z[-1, 0] == pytest.approx(0.67, abs=1e-12)
    assert 0.33 in t
    assert n_evals == (len(t) - 1) * 4


def test_segments_join_into_one_increasing_time_series():
    t, Z, _ = simulate_in_segments(euler_step, step_input_rhs, [0.0], 1.0, 0.1,
                                   breaks=[0.33, 0.33, 0.5, -1.0, 2.0])   # repeats and outside times ignored
    assert np.all(np.diff(t) > 0)
    assert t[0] == 0.0 and t[-1] == 1.0
    assert {0.33, 0.5} <= set(t)
    assert Z.shape == (len(t), 1)


def test_segments_match_simulate_when_there_are_no_breaks():
    t1, Z1, _ = simulate(rk4_step, one_dof_rhs, [X0, V0], 1.0, 0.01)
    t2, Z2, _ = simulate_in_segments(rk4_step, one_dof_rhs, [X0, V0], 1.0, 0.01, breaks=[])
    assert np.allclose(t1, t2) and np.allclose(Z1, Z2)


@pytest.mark.parametrize("h, t_end", [(0.0, 1.0), (0.01, 0.0)])
def test_segments_reject_bad_inputs(h, t_end):
    with pytest.raises(ValueError):
        simulate_in_segments(rk4_step, one_dof_rhs, [X0, V0], t_end, h, breaks=[0.5])
