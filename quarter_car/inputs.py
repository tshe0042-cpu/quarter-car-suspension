"""Ask the user for the simulation inputs in the terminal.

Every number has a unit and a supported range, both shown in the question.
An empty answer, text that is not a number, or a value outside the supported
range is rejected with a message and the question is asked again, so the
model only ever receives valid inputs. Menus work the same way.
"""

import numpy as np

# Each number: (key, description, unit, lowest allowed, highest allowed).
# The wheel (unsprung) mass is not asked for; it is fixed in main.py.
CAR_INPUTS = [
    ("body_mass", "Body mass carried by one wheel", "kg", 150.0, 700.0),    # small car to large SUV
    ("spring_stiffness", "Suspension spring stiffness", "kN/m", 5.0, 100.0),
    ("damping_ratio", "Damping ratio", "", 0.05, 1.0),                      # dimensionless
    ("tyre_stiffness", "Tyre stiffness", "kN/m", 100.0, 400.0),             # passenger car tyres
]

# Road surfaces. Menu key: (road type, menu text, numbers asked for that road).
ROAD_TYPES = {
    "1": ("bump", "Bump (speed hump or speed bump)", [
        ("height", "Bump height", "mm", 10.0, 150.0),
        ("length", "Bump length", "m", 0.3, 5.0),   # shorter bumps break the point-contact assumption
    ]),
    "2": ("pothole", "Pothole (sudden drop and rise)", [
        ("depth", "Pothole depth", "mm", 10.0, 150.0),
        ("length", "Pothole length", "m", 0.2, 3.0),
    ]),
}

SPEED_INPUT = ("speed", "Vehicle speed", "km/h", 5.0, 110.0)


def ask_float(description, unit, low, high, input_func=input):
    """Ask for a number until the user enters a valid one.

    Parameters
    ----------
    description : str
        What is being asked for, e.g. "Bump height".
    unit : str
        Unit shown to the user, e.g. "mm". Use "" for dimensionless inputs.
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
    prompt = f"{description} [{low:g} to {high:g}{unit_text}]: "
    while True:
        text = input_func(prompt).strip()
        if text == "":
            print("  Please enter a number.")
            continue
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


def ask_choice(question, options, input_func=input):
    """Show a menu and ask until the user picks one of its options.

    Parameters
    ----------
    question : str
        Heading shown above the menu.
    options : dict
        Menu keys mapped to their descriptions, e.g. ``{"1": "Bump"}``.
    input_func : callable, optional
        Function used to read a line of input. Defaults to ``input``.

    Returns
    -------
    str
        The chosen key.
    """
    print(question)
    for key, text in options.items():
        print(f"  {key}: {text}")
    keys = " or ".join(options)
    while True:
        answer = input_func(f"Choose {keys}: ").strip()
        if answer in options:
            return answer
        print(f"  Please choose {keys}.")


def ask_values(inputs, input_func=input):
    """Ask for each number in a list of inputs, in order.

    Parameters
    ----------
    inputs : list of tuple
        Entries of the form ``(key, description, unit, low, high)``, such as
        :data:`CAR_INPUTS`.
    input_func : callable, optional
        Function used to read a line of input. Defaults to ``input``.

    Returns
    -------
    dict
        The user's answers keyed by input name, in the units they were asked in.
    """
    return {key: ask_float(description, unit, low, high, input_func)
            for key, description, unit, low, high in inputs}


def ask_road(input_func=input):
    """Ask which road surface to drive over, then its size.

    Parameters
    ----------
    input_func : callable, optional
        Function used to read a line of input. Defaults to ``input``.

    Returns
    -------
    road_type : str
        "bump" or "pothole".
    values : dict
        The road's inputs in the units they were asked in (mm and m).
    """
    menu = {key: text for key, (_, text, _) in ROAD_TYPES.items()}
    road_type, _, inputs = ROAD_TYPES[ask_choice("Road surface:", menu, input_func)]
    return road_type, ask_values(inputs, input_func)
