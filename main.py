"""Quarter-car suspension simulator.

Enter the car's suspension settings, the speed and the size of the road
feature, then choose a bump or a pothole and real-time or slow-motion
playback. The program first compares seven solution methods on those inputs
(accuracy, run time and memory), then opens two windows: the car driving
over the road, and the method comparison. Close both windows to see the
results and choose whether to run again.

Run from the project folder::

    python main.py
"""

from quarter_car.animation import animate_road, describe_modes
from quarter_car.compare import compare_methods, comparison_figure, comparison_table
from quarter_car.inputs import (CAR_INPUTS, SPEED_INPUT, ask_real_time, ask_road, ask_values,
                                ask_yes_no)
from quarter_car.model import G, QuarterCar, damping_from_ratio
from quarter_car.road import Bump, Pothole

WHEEL_MASS = 40.0   # unsprung mass (wheel, tyre, hub and brake) [kg], kept fixed
TIME_STEP = 1e-3    # RK4 step [s]; restarted at road edges this is within 1% on every road (time_step_study.py)
SLOW_MOTION = {"bump": 0.25, "pothole": 0.1}   # slow-motion playback, as a fraction of real time


def build_car(values):
    """Create a QuarterCar in SI units from the user's inputs.

    Parameters
    ----------
    values : dict
        Inputs from ``ask_values(CAR_INPUTS)``, in the units they were asked
        in (kg, kN/m, dimensionless, kN/m).

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


def build_road(road_type, values):
    """Create the road in SI units from the user's inputs.

    Parameters
    ----------
    road_type : str
        "bump" or "pothole", from :func:`quarter_car.inputs.ask_road`.
    values : dict
        ``height_or_depth`` [mm] and ``length`` [m], from
        :func:`quarter_car.inputs.ask_road`.

    Returns
    -------
    Bump or Pothole
        The road, with lengths in metres.

    Raises
    ------
    ValueError
        If ``road_type`` is not one of the two above.
    """
    if road_type not in ("bump", "pothole"):
        raise ValueError(f"Unknown road type {road_type!r}")
    size = values["height_or_depth"] / 1000                  # mm to m
    if road_type == "bump":
        return Bump(height=size, length=values["length"])
    return Pothole(depth=size, length=values["length"])


def playback_speed(road_type, real_time):
    """Animation speed as a fraction of real time.

    Parameters
    ----------
    road_type : str
        "bump" or "pothole".
    real_time : bool
        True to play in real time, False for slow motion.

    Returns
    -------
    float
        1 for real time. For slow motion, 0.25 for a bump and 0.1 for a
        pothole, which is slowed more because the wheel is in the air for
        only tens of milliseconds.
    """
    return 1.0 if real_time else SLOW_MOTION[road_type]


def print_results(car, results):
    """Print the key comfort and safety results of a run.

    Parameters
    ----------
    car : QuarterCar
        Model parameters (for the static tyre load).
    results : dict
        Output of ``scenarios.road_summary``.
    """
    print(f"Peak body displacement:  {1e3 * results['peak_body_displacement']:.1f} mm")
    print(f"Peak body acceleration:  {results['peak_body_acceleration']:.2f} m/s^2")
    print(f"Peak suspension travel:  {1e3 * results['peak_suspension_travel']:.1f} mm")
    print(f"Tyre force:              {results['min_tyre_force']:.0f} to {results['max_tyre_force']:.0f} N "
          f"(static load {(car.m1 + car.m2) * G:.0f} N)")
    if results["lift_off_time"] is not None:
        print(f"Warning: the wheel leaves the road at t = {results['lift_off_time']:.2f} s and is in "
              f"the air for {1e3 * results['airborne_time']:.0f} ms in total, so there is no grip then.")
    print()


def main():
    """Run the simulator: ask for inputs, animate, report results, repeat."""
    print("Quarter-car suspension simulator")
    print("Type a value for each question. The allowed range is shown in brackets.\n")
    while True:
        car = build_car(ask_values(CAR_INPUTS))
        speed_kmh = ask_values([SPEED_INPUT])["speed"]
        road_type, road_values = ask_road()
        road = build_road(road_type, road_values)
        playback = playback_speed(road_type, ask_real_time())

        print(f"\nDamper coefficient: {car.b:.0f} N s/m")
        print(describe_modes(car))

        print("\nComparing solution methods on these inputs (takes about 5 to 30 s)...")
        comparison = compare_methods(car, road, speed_kmh / 3.6)
        print(comparison_table(comparison))
        comparison_figure(comparison, f"{speed_kmh:.0f} km/h, {road.label}")   # opens with the animation

        print("\nPlaying in real time." if playback == 1.0 else f"\nPlaying at {playback:g} x real speed.")
        print("Close both windows to continue.\n")
        results = animate_road(car, road, speed_kmh / 3.6, h=TIME_STEP, playback=playback)
        print_results(car, results)
        if not ask_yes_no("Run another simulation?"):
            break


if __name__ == "__main__":
    main()
