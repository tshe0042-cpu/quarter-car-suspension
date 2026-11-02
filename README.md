# Quarter-car suspension simulator

MMA3001 individual project, Monash University. Author: Ted Sheldon.

A Python program that simulates a car's suspension driving over a bump or a
pothole. The car is modelled as a quarter car (one wheel and the share of the
body it carries), and its state space equations are solved with the
fourth-order Runge-Kutta method (RK4) at a 1 ms time step.

## What it does

- Asks for the car's suspension settings, the vehicle speed and the size of
  the bump or pothole, then whether it is a bump or a pothole and whether to
  play the animation in real time or slow motion.
- Solves the equations of motion with RK4, restarting the solver at the edges
  of the bump or pothole so that no step crosses a sudden change in the road.
  The tyre can push on the road but not pull on it, so the wheel can leave the
  road.
- Prints the damper coefficient, the natural frequencies of the body and wheel
  hop, the peak body displacement, peak body acceleration, peak suspension
  travel and the range of tyre force, with a warning if the wheel leaves the
  road.
- Animates the car over the road, with plots of the displacements and the tyre
  force over time.
- Compares the chosen solver with six alternative methods on the user's own
  inputs, showing the accuracy, run time and memory of each.

## Inputs

| Input | Unit | Allowed range |
|---|---|---|
| Body mass carried by one wheel | kg | 150 to 700 |
| Suspension spring stiffness | kN/m | 5 to 100 |
| Damping ratio | none | 0.05 to 1 |
| Tyre stiffness | kN/m | 100 to 400 |
| Vehicle speed | km/h | 5 to 110 |
| Bump height or pothole depth | mm | 10 to 150 |
| Bump or pothole length | m | 0.3 to 4 |

The program then asks for two choices: bump (1) or pothole (2), and real time
(1) or slow motion (2). An empty answer, text, `nan`, `inf` or a value outside
the allowed range is rejected and the question is asked again. The wheel mass
is fixed at 40 kg.

## Installation

Install Python 3 and the required packages:

```
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## Running the simulator

From the project folder:

```
python main.py
```

After the inputs, the program compares the solution methods (this takes about
5 to 30 seconds), then opens two windows: the animation and the method
comparison. Close both windows to see the results and choose whether to run
another simulation.

## Tests

The automated tests are in the `tests` folder. Run them with:

```
python -m pytest
```

An HTML test report is saved in `docs/test_report.html`. To make a new one:

```
pip install pytest-html
python -m pytest --html=docs/test_report.html --self-contained-html
```

## Documentation

HTML documentation of the source code, generated from the docstrings with
pdoc, is in the `docs` folder. Open `docs/index.html` in a web browser. To
generate it again:

```
pip install pdoc
pdoc --docformat numpy -o docs quarter_car main time_step_study alternatives_study performance_study
```

## Studies

Three scripts produce the tables and figures used in the project report. Each
saves its results in the `results` folder.

| Script | What it does | Results |
|---|---|---|
| `time_step_study.py` | Finds the time step needed to capture sudden road changes, compares RK4 with and without restarts at the edges, and checks the program's 1 ms step on every study road and at the extremes of every input range | `error_vs_time_step.png`, `required_time_step.md`, `restart_road_readings.png`, `restart_acceleration_error.png`, `program_time_step_check.md`, `input_range_check.md` |
| `alternatives_study.py` | Compares the chosen solver with six alternative methods on four test cases | `alternatives.md`, `alternatives_error_and_cost.png` |
| `performance_study.py` | Profiles and times the original and improved code, and measures FLOPs, memory, repeatability, scalability and stability | `performance.md`, `performance_tradeoff.png` |

Run them from the project folder, for example `python time_step_study.py`.
Run times and memory are measured on the computer that runs the script, so
they will differ between computers.

## Project structure

```
main.py                  runs the simulator
quarter_car/             the simulation package
    model.py             car parameters, state space model, equations of motion with lift-off
    road.py              bump and pothole road profiles
    solvers.py           Euler and RK4 steps, fixed-step loop, restarts at the road edges
    scenarios.py         road input over time, running a simulation, working out the outputs
    inputs.py            asking for and checking the user's inputs
    animation.py         the animation window
    compare.py           the comparison of solution methods
    timestep.py          the reference solution and error measures
tests/                   automated tests (pytest)
results/                 tables and figures made by the study scripts
docs/                    HTML documentation and test report
time_step_study.py       time step study
alternatives_study.py    comparison of alternative methods
performance_study.py     optimisation and performance study
```

## Model assumptions

- Quarter car: one wheel and the share of body mass it carries, moving
  vertically only.
- Linear suspension spring and damper, with no bump stops or travel limits.
- The tyre is a linear spring with no damping, touching the road at a single
  point, and can push on the road but not pull on it.
- Constant vehicle speed, with the car starting at rest on flat road 0.3 s
  before the bump or pothole.
- Displacements are measured from static equilibrium.

## Licence

This project is released under the MIT Licence. See `LICENSE`.

## Acknowledgements

The code was developed with assistance from Claude (Anthropic), as described in
the AI use and reflection section of the project report.
