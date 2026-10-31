"""Tests for quarter_car.compare. Run with: python -m pytest -q"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from quarter_car.compare import (CHOSEN, METHODS, compare_methods, comparison_table, linear_solve,
                                 measure, plot_comparison)
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


@pytest.fixture(scope="module")
def comparison():
    """Method comparison for a short pothole run, shared by the tests below."""
    return compare_methods(CAR, Pothole(0.05, 0.6), 80 / 3.6, repeats=1, t_after=0.3)


def test_every_method_is_compared(comparison):
    assert [row["method"] for row in comparison] == list(METHODS)
    for row in comparison:
        assert row["seconds"] > 0 and row["memory"] > 0


def test_chosen_method_meets_1_percent_and_linear_model_does_not(comparison):
    """On a pothole the wheel leaves the road, which only the linear model gets wrong."""
    rows = {row["method"]: row for row in comparison}
    assert rows[CHOSEN]["passed"]
    assert not rows["Linear model, exact (lsim)"]["passed"]


def test_table_and_chart_have_one_entry_per_method(comparison):
    assert len(comparison_table(comparison).splitlines()) == len(METHODS) + 1    # plus the header
    ax = Figure().add_subplot()
    plot_comparison(comparison, ax, "Pothole at 80 km/h")
    assert len(ax.patches) == len(METHODS)
