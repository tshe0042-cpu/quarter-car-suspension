"""Tests for quarter_car.energy. Run with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.energy import energy_balance, stored_energy, trapezoid
from quarter_car.model import QuarterCar, damping_from_ratio
from quarter_car.road import Bump, Pothole
from quarter_car.scenarios import road_input, simulate_road

CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=damping_from_ratio(0.3, 20e3, 300.0), Kw=200e3)


def test_trapezoid_is_exact_for_a_straight_line():
    """The trapezoid rule has degree of exactness 1, even with uneven spacing."""
    t = np.array([0.0, 0.5, 2.0])
    assert trapezoid(t, 3 * t + 1) == pytest.approx(8.0)       # integral of 3t + 1 from 0 to 2


def test_trapezoid_is_second_order():
    """Halving the spacing cuts the error by 4 (observed order 2, as in notebook 6.2)."""
    def error(n):
        t = np.linspace(0.0, np.pi, n + 1)
        return abs(trapezoid(t, np.sin(t)) - 2.0)
    assert np.log2(error(16) / error(32)) == pytest.approx(2.0, abs=0.05)


def test_trapezoid_handles_a_jump_at_a_sample():
    """A step from 0 to 1 at t = 1 integrates to exactly 1 over 0 to 2 when the values
    just before and just after the jump are both given; using one value is out by h/2."""
    t = np.linspace(0.0, 2.0, 11)                       # t = 1 is a sample, h = 0.2
    after = np.where(t >= 1.0, 1.0, 0.0)                # value just after each time
    before = np.where(t > 1.0, 1.0, 0.0)                # value just before each time
    assert trapezoid(t, after, before) == pytest.approx(1.0)
    assert trapezoid(t, after) == pytest.approx(1.1)


def test_stored_energy_is_zero_at_rest():
    assert stored_energy(CAR, np.zeros((1, 4)))[0] == 0.0
    assert stored_energy(CAR, np.array([[0.0, 0.0, 1.0, 0.0]]))[0] == pytest.approx(0.5 * CAR.m1)


@pytest.mark.parametrize("road, speed_kmh, tolerance", [
    (Bump(0.075, 3.7), 30, 1e-4),       # smooth speed hump
    (Pothole(0.05, 0.6), 40, 2e-3),     # sudden edges, and the wheel leaves the road
])
def test_energy_balance_closes(road, speed_kmh, tolerance):
    """Tyre work in must equal the stored energy change plus the damper energy."""
    speed = speed_kmh / 3.6
    t, Z, _ = simulate_road(CAR, road, speed, h=1e-3)
    balance = energy_balance(CAR, t, Z, road_input(road, speed))
    assert balance["damper_energy"] > 0
    assert abs(balance["stored_change"]) < 0.01 * balance["damper_energy"]   # car has settled by the end
    assert balance["relative_residual"] < tolerance


def test_energy_balance_error_falls_as_h_squared():
    """The residual comes from the trapezoid rule, so halving h should cut it by about 4."""
    road, speed = Pothole(0.05, 0.6), 40 / 3.6

    def residual(h):
        t, Z, _ = simulate_road(CAR, road, speed, h=h)
        return abs(energy_balance(CAR, t, Z, road_input(road, speed))["residual"])

    assert np.log2(residual(1e-3) / residual(5e-4)) == pytest.approx(2.0, abs=0.3)
