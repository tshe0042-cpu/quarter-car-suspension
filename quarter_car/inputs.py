"""Ask the user for the simulation inputs in the terminal.

Every input has a unit, a default and a supported range. Pressing Enter accepts
the default. Text that is not a number, or a value outside the supported range,
is rejected with a message and the question is asked again, so the model only
ever receives valid inputs.
"""

import numpy as np

# Each input: (key, description, unit, default, lowest allowed, highest allowed).
# The wheel (unsprung) mass is not asked for; it is fixed in main.py.
INPUTS = [
    ("body_mass", "Body mass carried by one wheel", "kg", 300.0, 150.0, 700.0),    # small car to large SUV
    ("spring_stiffness", "Suspension spring stiffness", "kN/m", 20.0, 5.0, 100.0),
    ("damping_ratio", "Damping ratio", "", 0.3, 0.05, 1.0),             # dimensionless
    ("tyre_stiffness", "Tyre stiffness", "kN/m", 200.0, 100.0, 400.0),             # passenger car tyres
    ("bump_height", "Bump height", "mm", 75.0, 10.0, 150.0),
    ("bump_length", "Bump length", "m", 3.7, 0.3, 5.0),   # shorter bumps break the point-contact assumption
    ("speed", "Vehicle speed", "km/h", 30.0, 5.0, 80.0),
]


def ask_float(description, unit, default, low, high, input_func=input):
    """Ask for a number until the user enters a valid one.

    Parameters
    ----------
    description : str
        What is being asked for, e.g. "Bump height".
    unit : str
        Unit shown to the user, e.g. "mm". Use "" for dimensionless inputs.
    default : float
        Value returned if the user just presses Enter.
    low, high : float
        Supported range; values outside it are rejected.
    input_func : callable, optional
        Function used to read a line of input. Defaults to the built-in
        ``input``; tests pass a replacement so no typing is needed.

    Returns
    -------
    float
        The accepted value, in the unit shown.
    """
    unit_text = f" {unit}" if unit else ""
    prompt = f"{description} [{low:g} to {high:g}{unit_text}] (Enter = {default:g}): "
    while True:
        text = input_func(prompt).strip()
        if text == "":
            return float(default)
        try:
            value = float(text)
        except ValueError:
            print(f"  '{text}' is not a number. Please try again.")
            continue
        if not np.isfinite(value) or not low <= value <= high:
            print(f"  Please enter a value between {low:g} and {high:g}{unit_text}.")
            continue
        return value


def ask_yes_no(question, input_func=input):
    """Ask a yes/no question until the user answers y or n.

    Parameters
    ----------
    question : str
        The question to show.
    input_func : callable, optional
        Function used to read a line of input. Defaults to ``input``.

    Returns
    -------
    bool
        True for yes, False for no.
    """
    while True:
        answer = input_func(f"{question} (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


def ask_inputs(input_func=input):
    """Ask for every input in :data:`INPUTS`, in order.

    Parameters
    ----------
    input_func : callable, optional
        Function used to read a line of input. Defaults to ``input``.

    Returns
    -------
    dict
        Accepted values keyed by input name, in the units they were asked in
        (kg, kN/m, dimensionless, kN/m, mm, m, km/h).
    """
    return {key: ask_float(description, unit, default, low, high, input_func)
            for key, description, unit, default, low, high in INPUTS}
