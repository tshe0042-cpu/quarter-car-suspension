"""Tests for quarter_car.scenarios. Run with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.model import G, QuarterCar, make_rhs
from quarter_car.road import Bump, Pothole
from quarter_car.scenarios import (body_acceleration, edge_times, road_input, road_summary,
                                   simulate_road, suspension_travel, tyre_force)

CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)
STATIC_LOAD = (CAR.m1 + CAR.m2) * G


def test_road_input_converts_time_to_distance():
    """At 10 m/s the middle of a 2 m bump starting at t = 0.3 s is reached at t = 0.4 s."""
    r = road_input(Bump(height=0.1, length=2.0), speed=10.0, lead_in_time=0.3)
    assert r(0.4) == pytest.approx(0.1)
    assert r(0.0) == 0.0
    assert edge_times(Bump(height=0.1, length=2.0), 10.0, 0.3) == pytest.approx([0.3, 0.5])


@pytest.mark.parametrize("speed, lead_in_time", [(0.0, 0.3), (np.nan, 0.3), (10.0, -0.1)])
def test_road_input_rejects_bad_inputs(speed, lead_in_time):
    with pytest.raises(ValueError):
        road_input(Bump(0.1, 2.0), speed, lead_in_time)


def test_starts_at_rest_and_settles():
    """Body starts at rest and has settled to under 1% of the bump height by the end."""
    t, Z, r = simulate_road(CAR, Bump(0.075, 3.7), speed=30 / 3.6, t_after=3.0)
    assert np.allclose(Z[0], 0.0)
    assert Z.shape == (len(t), 4) and r.shape == t.shape
    assert abs(Z[-1, 1]) < 0.01 * 0.075


def test_run_lasts_lead_in_plus_crossing_plus_settling_time():
    t, _, _ = simulate_road(CAR, Pothole(0.05, 0.6), speed=10.0, lead_in_time=0.3, t_after=1.0)
    assert t[-1] == pytest.approx(0.3 + 0.06 + 1.0)


def test_restarting_at_edges_puts_a_step_on_every_edge():
    pothole, speed = Pothole(0.05, 0.6), 40 / 3.6
    t, _, _ = simulate_road(CAR, pothole, speed, h=7e-3, split_at_edges=True)
    for edge in edge_times(pothole, speed):
        assert np.min(np.abs(t - edge)) < 1e-12


def test_tyre_force_at_rest_equals_static_load():
    assert tyre_force(CAR, np.zeros((1, 4)), np.zeros(1))[0] == pytest.approx(STATIC_LOAD)


def test_tyre_force_is_never_negative():
    Z = np.array([[0.1, 0.0, 0.0, 0.0]])          # wheel 100 mm above a flat road
    assert tyre_force(CAR, Z, np.zeros(1))[0] == 0.0


def test_pothole_throws_the_wheel_off_the_road():
    """A 50 mm drop is deeper than the 17 mm static tyre deflection, so the
    wheel leaves the road; the tyre force must stay at zero, not go negative."""
    t, Z, r = simulate_road(CAR, Pothole(0.05, 0.6), speed=40 / 3.6)
    summary = road_summary(CAR, t, Z, r)
    assert summary["min_tyre_force"] == 0.0
    assert summary["lift_off_time"] == pytest.approx(0.3, abs=2e-3)
    assert 0.0 < summary["airborne_time"] < 0.2
    assert summary["max_tyre_force"] > 2 * STATIC_LOAD     # hard landing on the far edge


def test_gentle_bump_keeps_the_wheel_on_the_road():
    t, Z, r = simulate_road(CAR, Bump(0.075, 3.7), speed=30 / 3.6)
    summary = road_summary(CAR, t, Z, r)
    assert summary["lift_off_time"] is None
    assert summary["airborne_time"] == 0.0
    assert summary["min_tyre_force"] > 0
    assert summary["peak_body_displacement"] == pytest.approx(np.abs(Z[:, 1]).max())


def test_body_acceleration_matches_the_model():
    """The comfort output must equal the body row of the equations of motion."""
    Z = np.random.default_rng(1).normal(scale=0.05, size=(5, 4))
    f = make_rhs(CAR, road=lambda t: 0.0)
    expected = np.array([f(0.0, z)[3] for z in Z])
    assert np.allclose(body_acceleration(CAR, Z), expected)


def test_suspension_travel_is_body_minus_wheel():
    Z = np.array([[0.01, 0.03, 0.0, 0.0]])
    assert suspension_travel(Z)[0] == pytest.approx(0.02)


def test_road_input_gives_the_same_height_for_single_times_and_arrays():
    r = road_input(Pothole(0.05, 0.6), speed=10.0, lead_in_time=0.3)
    t = np.linspace(0.0, 1.0, 1001)
    assert np.array_equal([r(float(time)) for time in t], r(t))
