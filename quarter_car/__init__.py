"""Quarter-car suspension simulation package.

The simulator (`main`) is built from these modules:

- `quarter_car.model`: car parameters, the state space model and the equations
  of motion, with the wheel able to leave the road.
- `quarter_car.road`: bump and pothole road profiles.
- `quarter_car.solvers`: Euler and RK4 steps, a fixed-step loop, and restarts
  at sudden changes in the road.
- `quarter_car.scenarios`: turns a road into an input over time, runs a
  simulation and works out the outputs (body acceleration, suspension travel
  and tyre force).
- `quarter_car.inputs`: asks for the user's inputs in the terminal and rejects
  invalid ones.
- `quarter_car.animation`: the animation window.
- `quarter_car.compare`: compares the chosen solver with alternative methods
  on the same inputs.
- `quarter_car.timestep`: the reference solution and the error measures used
  in the studies.
"""
