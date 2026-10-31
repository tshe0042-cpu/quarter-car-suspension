"""Tests for quarter_car.timestep. Run with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.model import QuarterCar, make_rhs, natural_modes
from quarter_car.road import Pothole
from quarter_car.scenarios import edge_times, simulate_road
from quarter_car.solvers import rk4_step, simulate
from quarter_car.timestep import (adaptive_solve, max_stable_step, reference_solution,
                                  required_step, run_errors)

CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)
POTHOLE, SPEED = Pothole(depth=0.05, length=0.6), 40 / 3.6


@pytest.fixture(scope="module")
def reference():
    """Reference solution for the pothole, shared by the tests below."""
    return reference_solution(CAR, POTHOLE, SPEED)


def test_reference_agrees_with_a_very_fine_rk4_run(reference):
    """Two independent solvers (DOP853 and RK4 with h = 20 us) must agree."""
    t, Z, _ = simulate_road(CAR, POTHOLE, SPEED, h=2e-5, split_at_edges=True)
    assert max(run_errors(CAR, t, Z, reference)) < 1e-6


def test_pothole_edges_need_a_small_fixed_step(reference):
    """A 5 ms step that crosses the pothole edges is more than 1% out ..."""
    t, Z, _ = simulate_road(CAR, POTHOLE, SPEED, h=5e-3)
    assert max(run_errors(CAR, t, Z, reference)) > 0.01


def test_restarting_at_the_edges_allows_a_larger_step(reference):
    """... but the same step restarted at the edges is well within 1%."""
    t, Z, _ = simulate_road(CAR, POTHOLE, SPEED, h=2e-3, split_at_edges=True)
    assert max(run_errors(CAR, t, Z, reference)) < 0.01


def test_adaptive_solver_restarted_at_edges_steps_on_every_edge():
    run = adaptive_solve(CAR, POTHOLE, SPEED, split_at_edges=True)
    for edge in edge_times(POTHOLE, SPEED):
        assert np.min(np.abs(run.t - edge)) < 1e-12
    assert run.states(np.array([0.0])).shape == (1, 4)


def test_required_step_needs_every_finer_step_to_pass():
    steps = np.array([8, 4, 2, 1, 0.5]) * 1e-3
    result = {"steps": steps, "history": np.array([0.5, 0.005, 0.02, 0.004, 0.001]),
              "peaks": np.full(5, 0.5), "n_evals": np.array([10, 20, 40, 80, 160])}
    assert required_step(result, tol=0.01) == (1e-3, 80)       # 4 ms passes but 2 ms does not
    assert required_step(result, tol=0.01, measure="peaks") == (None, None)   # no peak passes
    result["history"] = np.full(5, 0.001)
    assert required_step(result, tol=0.01) == (8e-3, 10)       # everything passes


def test_euler_stability_limit_matches_theory():
    """Explicit Euler is stable for a mode while h < 2 zeta / omega_n (|1 + h lambda| <= 1)."""
    freqs, zetas = natural_modes(CAR)
    expected = np.min(2 * zetas / (2 * np.pi * freqs))
    assert max_stable_step(CAR, "euler") == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("factor, grows", [(0.9, False), (1.1, True)])
def test_rk4_stability_limit_separates_decay_from_growth(factor, grows):
    """Just below the RK4 limit a disturbance dies away; just above it, it grows."""
    h = factor * max_stable_step(CAR, "rk4")
    f = make_rhs(CAR, road=lambda t: 0.0, lift_off=False)
    _, Z, _ = simulate(rk4_step, f, [0.01, 0.0, 0.0, 0.0], 300 * h, h)
    assert (np.abs(Z[-1]).max() > np.abs(Z[0]).max()) == grows


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        max_stable_step(CAR, "rk2")
