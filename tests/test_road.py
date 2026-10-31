"""Tests for quarter_car.road. Run with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.road import Bump, Pothole


def test_bump_is_flat_off_the_bump():
    bump = Bump(height=0.075, length=1.0)
    assert np.allclose(bump.profile([-1.0, -0.01, 1.01, 10.0]), 0.0)


def test_bump_peaks_in_the_middle_and_is_continuous_at_the_ends():
    bump = Bump(height=0.075, length=1.0)
    assert bump.profile(0.5) == pytest.approx(0.075)
    assert np.allclose(bump.profile([0.0, 1.0]), 0.0, atol=1e-15)


def test_pothole_is_a_sudden_drop_and_rise():
    """The height jumps at both edges: flat, then the full depth, then flat."""
    pothole = Pothole(depth=0.05, length=0.6)
    assert pothole.profile(-1e-9) == 0.0
    assert pothole.profile(0.0) == -0.05
    assert pothole.profile(0.599) == -0.05
    assert pothole.profile(0.6) == 0.0
    assert pothole.edges == (0.0, 0.6)


def test_a_single_distance_gives_a_float():
    for road in (Bump(0.075, 1.0), Pothole(0.05, 0.6)):
        assert isinstance(road.profile(0.3), float)


def test_labels_describe_the_road():
    assert Bump(0.075, 3.7).label == "75 mm high, 3.7 m long bump"
    assert Pothole(0.05, 0.6).label == "50 mm deep, 0.6 m long pothole"


@pytest.mark.parametrize("make_road", [
    lambda: Bump(height=-0.01, length=1.0),
    lambda: Bump(height=0.05, length=0.0),
    lambda: Pothole(depth=np.nan, length=1.0),
    lambda: Pothole(depth=0.05, length=-1.0),
])
def test_bad_roads_are_rejected(make_road):
    with pytest.raises(ValueError):
        make_road()


def test_single_number_fast_path_matches_the_array_path():
    """profile() uses plain Python for a single number; it must give the same heights."""
    s = np.linspace(-0.5, 4.5, 2001)
    s = np.concatenate([s, [0.0, 0.6, 3.7]])          # include the exact edges
    for road in (Bump(0.075, 3.7), Pothole(0.05, 0.6)):
        one_at_a_time = np.array([road.profile(float(value)) for value in s])
        assert np.allclose(one_at_a_time, road.profile(s), rtol=0.0, atol=1e-15)
