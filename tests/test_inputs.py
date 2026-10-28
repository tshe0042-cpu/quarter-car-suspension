"""Tests for quarter_car.inputs and main.build_car. Run with: python -m pytest -q"""

import pytest

from main import WHEEL_MASS, build_car
from quarter_car.inputs import INPUTS, ask_float, ask_inputs, ask_yes_no
from quarter_car.model import damping_from_ratio


def fake_input(*answers):
    """Return an input function that replays the given answers in order."""
    replies = iter(answers)
    return lambda _prompt: next(replies)


def test_enter_accepts_the_default():
    assert ask_float("Speed", "km/h", 30.0, 5.0, 80.0, fake_input("")) == 30.0


def test_valid_number_is_accepted():
    assert ask_float("Speed", "km/h", 30.0, 5.0, 80.0, fake_input("45")) == 45.0


def test_invalid_answers_are_rejected_until_a_valid_one(capsys):
    """Text, NaN and out-of-range values are all rejected with a message."""
    value = ask_float("Speed", "km/h", 30.0, 5.0, 80.0,
                      fake_input("fast", "nan", "200", "-5", "60"))
    assert value == 60.0
    messages = capsys.readouterr().out
    assert "not a number" in messages
    assert messages.count("between 5 and 80") == 3


@pytest.mark.parametrize("answers, expected", [
    (("y",), True), (("No",), False), (("maybe", "yes"), True),
])
def test_yes_no(answers, expected):
    assert ask_yes_no("Again?", fake_input(*answers)) is expected


def test_all_defaults():
    values = ask_inputs(fake_input(*[""] * len(INPUTS)))
    assert values == {key: default for key, _, _, default, _, _ in INPUTS}


def test_build_car_converts_units():
    values = {"body_mass": 300.0, "spring_stiffness": 20.0, "damping_ratio": 0.3,
              "tyre_stiffness": 200.0}
    car = build_car(values)
    assert (car.m1, car.m2, car.Ks, car.Kw) == (WHEEL_MASS, 300.0, 20e3, 200e3)
    assert car.b == pytest.approx(damping_from_ratio(0.3, 20e3, 300.0))
