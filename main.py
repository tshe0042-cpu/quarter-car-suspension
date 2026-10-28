"""Quarter-car suspension simulator.

Enter the car's suspension settings and a bump in the terminal, then watch the
car drive over the bump. Close the animation window to see the results and
choose whether to run again.

Run from the project folder::

    python main.py
"""

from quarter_car.animation import animate_bump, describe_modes
from quarter_car.inputs import ask_inputs, ask_yes_no
from quarter_car.model import QuarterCar, damping_from_ratio

WHEEL_MASS = 40.0   # unsprung mass (wheel, tyre, hub and brake) [kg], kept fixed


def build_car(values):
    """Create a QuarterCar in SI units from the user's inputs.

    Parameters
    ----------
    values : dict
        Inputs from :func:`quarter_car.inputs.ask_inputs`, in the units they
        were asked in (kg, kN/m, dimensionless, kN/m).

    Returns
    -------
    QuarterCar
        Model parameters in SI units, with the fixed wheel mass.
    """
    body_mass = values["body_mass"]
    spring = values["spring_stiffness"] * 1e3            # kN/m to N/m
    return QuarterCar(m1=WHEEL_MASS, m2=body_mass, Ks=spring,
                      b=damping_from_ratio(values["damping_ratio"], spring, body_mass),
                      Kw=values["tyre_stiffness"] * 1e3)


def main():
    """Run the simulator: ask for inputs, animate, report results, repeat."""
    print("Quarter-car suspension simulator")
    print("Press Enter to accept the default value shown.\n")
    while True:
        values = ask_inputs()
        car = build_car(values)
        print(f"\nDamper coefficient: {car.b:.0f} N s/m")
        print(describe_modes(car))
        print("Close the animation window to continue.\n")

        results = animate_bump(car, height=values["bump_height"] / 1000,   # mm to m
                               length=values["bump_length"],
                               speed=values["speed"] / 3.6)                # km/h to m/s

        print(f"Peak body displacement:  {1e3 * results['peak_body_displacement']:.1f} mm")
        print(f"Peak body acceleration:  {results['peak_body_acceleration']:.2f} m/s^2")
        print(f"Peak suspension travel:  {1e3 * results['peak_suspension_travel']:.1f} mm")
        print(f"Minimum tyre force:      {results['min_tyre_force']:.0f} N")
        if results["lift_off_time"] is not None:
            print(f"Warning: the wheel leaves the road at t = {results['lift_off_time']:.2f} s, "
                  "so results after that are not valid.")
        print()
        if not ask_yes_no("Run another simulation?"):
            break


if __name__ == "__main__":
    main()
