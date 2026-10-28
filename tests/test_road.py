"""Tests for quarter_car.road and quarter_car.scenarios. Run with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.model import QuarterCar, make_rhs
from quarter_car.road import half_sine_bump, half_sine_profile
from quarter_car.scenarios import (G, body_acceleration, bump_summary, simulate_bump,
                                   suspension_travel, tyre_force)

CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)


def test_profile_is_flat_off_the_bump():
    r = half_sine_profile([0.0, 1.9, 3.1, 10.0], height=0.075, length=1.0, start=2.0)
    assert np.allclose(r, 0.0)


def test_profile_peaks_at_the_middle_and_is_continuous_at_the_ends():
    start, length, height = 2.0, 1.0, 0.075
    assert half_sine_profile(start + length / 2, height, length, start) == pytest.approx(height)
    ends = half_sine_profile([start, start + length], height, length, start)
    assert np.allclose(ends, 0.0)


def test_bump_converts_time_to_position():
    """At 10 m/s the middle of a bump from 5 to 7 m is reached at t = 0.6 s."""
    road = half_sine_bump(height=0.1, length=2.0, speed=10.0, start_distance=5.0)
    assert road(0.6) == pytest.approx(0.1)
    assert road(0.0) == 0.0


@pytest.mark.parametrize("bad", [
    {"height": -0.01}, {"length": 0.0}, {"speed": 0.0}, {"speed": np.nan}, {"start_distance": -1.0},
])
def test_bump_rejects_bad_inputs(bad):
    args = {"height": 0.075, "length": 1.0, "speed": 5.0, "start_distance": 0.0, **bad}
    with pytest.raises(ValueError):
        half_sine_bump(**args)


def test_simulate_bump_starts_at_rest_and_settles():
    """Body starts at rest and has settled to under 1% of the bump height by the end."""
    height = 0.075
    t, Z, r = simulate_bump(CAR, height, length=3.7, speed=30 / 3.6, t_after=3.0)
    assert np.allclose(Z[0], 0.0)
    assert Z.shape == (len(t), 4) and r.shape == t.shape
    assert abs(Z[-1, 1]) < 0.01 * height


def test_tyre_force_at_rest_equals_static_weight():
    Z = np.zeros((1, 4))
    assert tyre_force(CAR, Z, np.zeros(1))[0] == pytest.approx((CAR.m1 + CAR.m2) * G)


def test_short_sharp_bump_flags_wheel_lift_off():
    """A 50 mm bump only 0.6 m long at 40 km/h throws the wheel off the road."""
    t, Z, r = simulate_bump(CAR, height=0.05, length=0.6, speed=40 / 3.6)
    assert tyre_force(CAR, Z, r).min() < 0


def test_body_acceleration_matches_the_model():
    """The comfort output must equal the body row of the state equations."""
    rng = np.random.default_rng(1)
    Z = rng.normal(scale=0.05, size=(5, 4))
    f = make_rhs(CAR, road=lambda t: 0.0)
    expected = np.array([f(0.0, z)[3] for z in Z])
    assert np.allclose(body_acceleration(CAR, Z), expected)


def test_suspension_travel_is_body_minus_wheel():
    Z = np.array([[0.01, 0.03, 0.0, 0.0]])
    assert suspension_travel(Z)[0] == pytest.approx(0.02)


def test_bump_summary_for_a_gentle_hump():
    t, Z, r = simulate_bump(CAR, height=0.075, length=3.7, speed=30 / 3.6)
    summary = bump_summary(CAR, t, Z, r)
    assert summary["lift_off_time"] is None
    assert summary["min_tyre_force"] > 0
    assert summary["peak_body_displacement"] == pytest.approx(Z[:, 1].max())
