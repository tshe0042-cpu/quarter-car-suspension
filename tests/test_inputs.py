"""Tests for quarter_car.inputs and the builders in main.py. Run with: python -m pytest -q"""

import pytest

from main import WHEEL_MASS, build_car, build_road
from quarter_car.inputs import (CAR_INPUTS, ROAD_TYPES, ask_choice, ask_float, ask_road,
                                ask_values, ask_yes_no)
from quarter_car.model import damping_from_ratio
from quarter_car.road import Bump, Pothole


def fake_input(*answers):
    """Return an input function that replays the given answers in order."""
    replies = iter(answers)
    return lambda _prompt: next(replies)


def test_valid_number_is_accepted():
    assert ask_float("Speed", "km/h", 5.0, 80.0, fake_input("45")) == 45.0


def test_invalid_answers_are_rejected_until_a_valid_one(capsys):
    """An empty answer, text, NaN and out-of-range values are all rejected with a message."""
    value = ask_float("Speed", "km/h", 5.0, 80.0,
                      fake_input("", "fast", "nan", "200", "-5", "60"))
    assert value == 60.0
    messages = capsys.readouterr().out
    assert "Please enter a number." in messages
    assert "not a number" in messages
    assert messages.count("between 5 and 80") == 3


@pytest.mark.parametrize("answers, expected", [
    (("y",), True), (("No",), False), (("maybe", "yes"), True),
])
def test_yes_no(answers, expected):
    assert ask_yes_no("Again?", fake_input(*answers)) is expected


@pytest.mark.parametrize("answers, expected", [
    (("2",), "2"),
    (("", "3", "1"), "1"),        # empty and invalid answers are asked again
])
def test_menu_choice(answers, expected):
    options = {"1": "Bump", "2": "Pothole"}
    assert ask_choice("Road surface:", options, fake_input(*answers)) == expected


def test_ask_values_returns_the_answers_by_name():
    values = ask_values(CAR_INPUTS, fake_input("450", "25", "0.4", "250"))
    assert values == {"body_mass": 450.0, "spring_stiffness": 25.0,
                      "damping_ratio": 0.4, "tyre_stiffness": 250.0}


@pytest.mark.parametrize("answers, expected_type, expected_values", [
    (("1", "75", "3.7"), "bump", {"height": 75.0, "length": 3.7}),
    (("2", "40", "0.5"), "pothole", {"depth": 40.0, "length": 0.5}),
])
def test_ask_road(answers, expected_type, expected_values):
    road_type, values = ask_road(fake_input(*answers))
    assert road_type == expected_type
    assert values == expected_values


@pytest.mark.parametrize("limit", ["low", "high"])
def test_every_menu_road_can_be_built_at_its_limits(limit):
    """The allowed ranges in the menu must all give valid roads."""
    for road_type, _, inputs in ROAD_TYPES.values():
        values = {key: (low if limit == "low" else high) for key, _, _, low, high in inputs}
        assert build_road(road_type, values).length > 0


def test_build_car_converts_units():
    values = {"body_mass": 300.0, "spring_stiffness": 20.0, "damping_ratio": 0.3,
              "tyre_stiffness": 200.0}
    car = build_car(values)
    assert (car.m1, car.m2, car.Ks, car.Kw) == (WHEEL_MASS, 300.0, 20e3, 200e3)
    assert car.b == pytest.approx(damping_from_ratio(0.3, 20e3, 300.0))


def test_build_road_converts_millimetres_to_metres():
    assert build_road("bump", {"height": 75.0, "length": 3.7}) == Bump(0.075, 3.7)
    assert build_road("pothole", {"depth": 50.0, "length": 0.6}) == Pothole(0.05, 0.6)


def test_unknown_road_type_is_rejected():
    with pytest.raises(ValueError):
        build_road("gravel", {})
