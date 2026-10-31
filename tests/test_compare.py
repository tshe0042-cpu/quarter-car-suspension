"""Tests for quarter_car.compare. Run with: python -m pytest -q"""

import numpy as np

from quarter_car.compare import linear_solve, measure
from quarter_car.model import G, QuarterCar
from quarter_car.road import Bump, Pothole
from quarter_car.scenarios import road_input, simulate_road

CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)


def test_linear_solution_matches_rk4_when_the_wheel_stays_on_the_road():
    """Two independent methods must agree when the model is linear (no lift-off)."""
    bump, speed = Bump(0.075, 3.7), 30 / 3.6
    t_lin, Z_lin = linear_solve(CAR, bump, speed, h=1e-3)
    t_rk4, Z_rk4 = simulate_road(CAR, bump, speed, h=1e-3)[:2]
    y_rk4 = np.interp(t_lin, t_rk4, Z_rk4[:, 1])
    assert np.abs(Z_lin[:, 1] - y_rk4).max() < 1e-4 * np.abs(y_rk4).max()


def test_linear_solution_lets_the_tyre_pull_on_a_pothole():
    """The linear model has no lift-off, so its tyre force goes negative in a pothole."""
    pothole, speed = Pothole(0.05, 0.6), 40 / 3.6
    t, Z = linear_solve(CAR, pothole, speed)
    force = (CAR.m1 + CAR.m2) * G + CAR.Kw * (road_input(pothole, speed)(t) - Z[:, 0])
    assert force.min() < -1000.0


def test_measure_returns_the_result_time_and_memory():
    result, seconds, peak_bytes = measure(lambda: np.ones(100_000).sum(), repeats=3)
    assert result == 100_000
    assert seconds > 0
    assert peak_bytes >= 100_000 * 8          # the array alone is 800 kB
