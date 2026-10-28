"""Interactive animation of the quarter car driving over a half-sine bump.

Run from the project folder (next to ``quarter_car``)::

    python animate_bump.py

Set the vehicle speed and the bump height and length with the sliders, then
press "Run" to simulate and animate.

The left panel draws the same two-mass model that is simulated, laid out like
the simplified system diagram: the body ``m2`` sits on the suspension spring
``Ks`` and damper ``b``, which sit on the wheel and axle ``m1``, which sits on
the tyre spring ``Kw`` touching the road. The car stays in the centre of the
view and the road moves underneath it. The right panel shows road, wheel and
body displacement against time. If the tyre force goes negative, the title warns
that the wheel would leave the road, because the linear model is not valid after
that point.

For smooth playback, the animation uses blitting: the axes, ticks and labels
are drawn once, and only the moving parts are redrawn each frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from quarter_car.model import QuarterCar
from quarter_car.road import half_sine_profile
from quarter_car.scenarios import simulate_bump, tyre_force

# Vehicle used in the animation (typical passenger car corner).
CAR = QuarterCar(m1=40.0, m2=300.0, Ks=20e3, b=1500.0, Kw=200e3)

# Simulation settings.
LEAD_IN_TIME = 0.3      # time before reaching the bump [s]
T_AFTER = 2.0           # time simulated after leaving the bump [s]
STEP = 1e-3             # RK4 step size [s]

# Drawing settings (only affect the picture, not the physics). Heights are at rest [m].
M1_BASE, M1_W, M1_H = 0.30, 0.50, 0.15      # bottom height, width and height of m1
M2_BASE, M2_W, M2_H = 0.95, 1.00, 0.25      # bottom height, width and height of m2
SIDE = 0.18             # sideways offset of Ks (left) and b (right) from the centre
CYL_LEN, CYL_HALF_W, ROD_LEN = 0.30, 0.05, 0.30   # damper cylinder and rod sizes
VIEW_HALF_WIDTH = 1.5   # half the width of the view around the car [m]
ROAD_X = np.linspace(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH, 400)  # road points, relative to the car [m]

# Playback settings.
PLAYBACK = 0.25         # animation speed as a fraction of real time
FPS = 50                # animation frames per second
HOLD_FRAMES = 40        # pause at the end before the animation repeats

state = {}              # current simulation results, replaced by each run


def spring_points(x_pos, bottom, top, coils=6, width=0.06):
    """Return x and y coordinates of a zig-zag spring between two heights."""
    ys = np.linspace(bottom, top, 2 * coils + 3)
    xs = np.full_like(ys, x_pos)
    xs[1:-1] += width * np.where(np.arange(1, len(ys) - 1) % 2, 1, -1)
    xs[1], xs[-2] = x_pos, x_pos            # straight ends
    return xs, ys


def damper_points(x_pos, bottom, top):
    """Return x and y coordinates of a damper (dashpot) between two heights.

    The cylinder is fixed to the lower mass and the piston rod to the upper
    mass, so the piston moves into the cylinder as the suspension compresses.
    NaN values split the line into separate pieces.
    """
    base = bottom + 0.05                    # bottom of the cylinder
    piston = top - ROD_LEN                  # piston height
    w, nan = CYL_HALF_W, np.nan
    xs = [x_pos, x_pos, nan,                                  # stem
          x_pos - w, x_pos - w, x_pos + w, x_pos + w, nan,    # cylinder
          x_pos, x_pos, nan,                                  # rod
          x_pos - 0.8 * w, x_pos + 0.8 * w]                   # piston
    ys = [bottom, base, nan,
          base + CYL_LEN, base, base, base + CYL_LEN, nan,
          top, piston, nan,
          piston, piston]
    return xs, ys


def run_simulation(speed_kmh, height_mm, length_m):
    """Simulate one bump crossing and prepare the data for every animation frame.

    The solver output is interpolated onto evenly spaced frame times, so the
    playback speed is exact and independent of the solver step size.
    """
    speed = speed_kmh / 3.6                  # km/h to m/s
    height = height_mm / 1000.0              # mm to m
    t, Z, r = simulate_bump(CAR, height, length_m, speed,
                            lead_in_time=LEAD_IN_TIME, t_after=T_AFTER, h=STEP)
    force = tyre_force(CAR, Z, r)
    bump_start = speed * LEAD_IN_TIME

    frame_t = np.append(np.arange(0.0, t[-1], PLAYBACK / FPS), t[-1])
    state.update(
        t=t, Z=Z, r=r, speed=speed, height=height, length=length_m,
        bump_start=bump_start, counter=0,
        frame_t=frame_t,
        frame_x=np.interp(frame_t, t, Z[:, 0]),
        frame_y=np.interp(frame_t, t, Z[:, 1]),
        frame_r=half_sine_profile(speed * frame_t, height, length_m, bump_start),
        lift_off_time=t[np.argmax(force < 0)] if np.any(force < 0) else None,
    )


def draw_results():
    """Update the time histories and redraw the still parts of the figure."""
    t, Z, r = state["t"], state["Z"], state["r"]
    road_hist.set_data(t, 1e3 * r)
    wheel_hist.set_data(t, 1e3 * Z[:, 0])
    body_hist.set_data(t, 1e3 * Z[:, 1])
    ax_hist.set_xlim(0, t[-1])
    ax_hist.relim()
    ax_hist.autoscale_view(scalex=False)

    ax_car.set_title(f"{3.6 * state['speed']:.0f} km/h over a {1e3 * state['height']:.0f} mm "
                     f"high, {state['length']:.1f} m long bump")
    if state["lift_off_time"] is None:
        ax_hist.set_title("Peak body displacement: "
                          f"{1e3 * Z[:, 1].max():.1f} mm", color="black")
    else:
        ax_hist.set_title(f"Warning: wheel leaves the road at t = {state['lift_off_time']:.2f} s, "
                          "results after this are not valid", color="red")
    # A full redraw now means the animation captures an up-to-date background.
    fig.canvas.draw()


def update(_frame):
    """Move both masses, both springs, the damper and the road to the next frame."""
    n = len(state["frame_t"])
    k = min(state["counter"] % (n + HOLD_FRAMES), n - 1)
    state["counter"] += 1

    t = state["frame_t"][k]
    x, y = state["frame_x"][k], state["frame_y"][k]   # wheel and body displacement [m]
    r = state["frame_r"][k]                            # road height under the tyre [m]
    distance = state["speed"] * t                      # distance travelled [m]

    m1_bottom, m2_bottom = M1_BASE + x, M2_BASE + y
    m1_top = m1_bottom + M1_H

    # The car stays at x = 0 and the road slides underneath it.
    road_line.set_data(ROAD_X, half_sine_profile(distance + ROAD_X, state["height"],
                                                 state["length"], state["bump_start"]))
    m1_box.set_xy((-M1_W / 2, m1_bottom))
    m2_box.set_xy((-M2_W / 2, m2_bottom))
    tyre_spring.set_data(*spring_points(0.0, r, m1_bottom))            # Kw: road to m1
    susp_spring.set_data(*spring_points(-SIDE, m1_top, m2_bottom))     # Ks: m1 to m2
    damper.set_data(*damper_points(SIDE, m1_top, m2_bottom))           # b: m1 to m2

    m1_label.set_position((0.0, m1_bottom + M1_H / 2))
    m2_label.set_position((0.0, m2_bottom + M2_H / 2))
    kw_label.set_position((0.10, (r + m1_bottom) / 2))
    ks_label.set_position((-SIDE - 0.10, (m1_top + m2_bottom) / 2))
    b_label.set_position((SIDE + 0.09, (m1_top + m2_bottom) / 2))

    cursor.set_xdata([t, t])
    time_text.set_text(f"t = {t:.2f} s")   # kept short: long text is slow to redraw
    return (road_line, m1_box, m2_box, tyre_spring, susp_spring, damper,
            m1_label, m2_label, kw_label, ks_label, b_label, time_text,
            road_hist, wheel_hist, body_hist, cursor)


def on_run(_event):
    """Read the sliders, re-run the simulation and restart the animation."""
    run_simulation(s_speed.val, s_height.val, s_length.val)
    draw_results()


# ---- Figure layout ------------------------------------------------------------
# Moving parts are created with animated=True so only they are redrawn each frame.
fig = plt.figure(figsize=(13, 6.5))
ax_car = fig.add_axes([0.04, 0.32, 0.44, 0.62])
ax_hist = fig.add_axes([0.56, 0.34, 0.41, 0.58])

ax_car.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
ax_car.set_ylim(-0.1, M2_BASE + M2_H + 0.2)
ax_car.set_aspect("equal", adjustable="box")
ax_car.set_xlabel("Distance from the car [m]")
ax_car.set_ylabel("Height [m]")
(road_line,) = ax_car.plot([], [], color="0.3", lw=2, animated=True)
m1_box = ax_car.add_patch(Rectangle((0, M1_BASE), M1_W, M1_H, color="tab:orange",
                                    alpha=0.8, animated=True))
m2_box = ax_car.add_patch(Rectangle((0, M2_BASE), M2_W, M2_H, color="tab:blue",
                                    alpha=0.8, animated=True))
(tyre_spring,) = ax_car.plot([], [], color="black", lw=1.5, animated=True)
(susp_spring,) = ax_car.plot([], [], color="tab:green", lw=1.5, animated=True)
(damper,) = ax_car.plot([], [], color="tab:purple", lw=1.5, animated=True)
m1_label = ax_car.text(0, 0, r"$m_1$", ha="center", va="center", color="white", animated=True)
m2_label = ax_car.text(0, 0, r"$m_2$", ha="center", va="center", color="white", animated=True)
kw_label = ax_car.text(0, 0, r"$K_w$", va="center", animated=True)
ks_label = ax_car.text(0, 0, r"$K_s$", ha="right", va="center", animated=True)
b_label = ax_car.text(0, 0, r"$b$", va="center", animated=True)
time_text = ax_car.text(0.02, 0.97, "", transform=ax_car.transAxes, va="top", animated=True)

(road_hist,) = ax_hist.plot([], [], color="0.3", label="Road r", animated=True)
(wheel_hist,) = ax_hist.plot([], [], color="tab:orange", label="Wheel x (m1)", animated=True)
(body_hist,) = ax_hist.plot([], [], color="tab:blue", label="Body y (m2)", animated=True)
cursor = ax_hist.axvline(0, color="red", lw=1, animated=True)
ax_hist.set_xlabel("Time [s]")
ax_hist.set_ylabel("Displacement [mm]")
ax_hist.legend(loc="upper right")
ax_hist.grid(alpha=0.3)

s_speed = Slider(fig.add_axes([0.15, 0.17, 0.55, 0.03]), "Speed [km/h]", 5, 80, valinit=30, valstep=1)
s_height = Slider(fig.add_axes([0.15, 0.11, 0.55, 0.03]), "Bump height [mm]", 10, 150, valinit=75, valstep=5)
s_length = Slider(fig.add_axes([0.15, 0.05, 0.55, 0.03]), "Bump length [m]", 0.3, 5.0, valinit=3.7, valstep=0.1)
run_button = Button(fig.add_axes([0.80, 0.05, 0.12, 0.15]), "Run")
run_button.on_clicked(on_run)

if __name__ == "__main__":
    run_simulation(s_speed.val, s_height.val, s_length.val)
    draw_results()
    # Keep a reference to the animation, otherwise Python deletes it.
    animation = FuncAnimation(fig, update, interval=1000 / FPS, blit=True, cache_frame_data=False)
    plt.show()
