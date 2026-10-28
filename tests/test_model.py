"""Tests for quarter_car.model. Run from the repo root with: python -m pytest -q"""

import numpy as np
import pytest

from quarter_car.model import (
    QuarterCar,
    damping_from_ratio,
    make_rhs,
    natural_modes,
    state_matrices,
    transfer_function,
)

# Typical passenger-car values used only for testing.
PARAMS = dict(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)
CAR = QuarterCar(**PARAMS)


def test_rhs_returns_four_states():
    f = make_rhs(CAR, road=lambda t: 0.0)
    assert f(0.0, np.zeros(4)).shape == (4,)


def test_rest_stays_at_rest():
    """With a flat road and no motion, nothing should accelerate."""
    f = make_rhs(CAR, road=lambda t: 0.0)
    assert np.allclose(f(0.0, np.zeros(4)), 0.0)


def test_raised_road_equilibrium():
    """On a road raised by h, x = y = h with zero velocity is an equilibrium."""
    h = 0.05
    f = make_rhs(CAR, road=lambda t: h)
    assert np.allclose(f(0.0, np.array([h, h, 0.0, 0.0])), 0.0)


def test_poles_match_eigenvalues():
    """Roots of the transfer-function denominator must equal the eigenvalues of A.

    The two come from separate derivations, so this checks the A matrix.
    """
    A, _, _ = state_matrices(CAR)
    _, den = transfer_function(CAR)
    poles = np.sort_complex(np.roots(den))
    eigenvalues = np.sort_complex(np.linalg.eigvals(A))
    assert np.allclose(poles, eigenvalues)


def test_typical_mode_frequencies():
    """Typical cars have body bounce near 1 to 1.5 Hz and wheel hop near 10 to 15 Hz."""
    freqs_hz, _ = natural_modes(CAR)
    assert 1.0 < freqs_hz[0] < 1.5
    assert 10.0 < freqs_hz[1] < 15.0


def test_damping_ratio_conversion():
    expected = 2 * 0.3 * np.sqrt(20e3 * 300.0)
    assert damping_from_ratio(0.3, Ks=20e3, m2=300.0) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [
    {"m1": -40.0},
    {"m2": 0.0},
    {"Ks": np.nan},
    {"Kw": np.inf},
    {"b": -1.0},
])
def test_invalid_parameters_raise(bad):
    params = {**PARAMS, **bad}
    with pytest.raises(ValueError):
        QuarterCar(**params)


def test_negative_damping_ratio_raises():
    with pytest.raises(ValueError):
        damping_from_ratio(-0.1, Ks=20e3, m2=300.0)
