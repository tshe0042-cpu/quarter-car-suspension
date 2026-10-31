"""Animation of the quarter car driving over a road.

The left panel draws the same two-mass model that is simulated, laid out like
the simplified system diagram: the body ``m2`` sits on the suspension spring
``Ks`` and damper ``b``, which sit on the wheel and axle ``m1``, which sits on
the tyre spring ``Kw`` touching the road. The car stays in the centre of the
view and the road moves underneath it. If the wheel leaves the road, the tyre
spring is drawn at its unloaded length with a gap below it.

The right panels show the road, wheel and body displacement and the tyre
force against time, and the text underneath lists the inputs and key results.

For smooth playback the animation uses blitting: the axes, labels and time
histories are drawn once, and only the moving parts are redrawn each frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from quarter_car.model import G, natural_modes
from quarter_car.scenarios import road_input, road_summary, simulate_road, tyre_force

# Drawing settings (only affect the picture, not the physics). Heights are at rest [m].
M1_BASE, M1_W, M1_H = 0.30, 0.50, 0.15      # bottom height, width and height of m1
M2_BASE, M2_W, M2_H = 0.95, 1.00, 0.25      # bottom height, width and height of m2
SIDE = 0.18             # sideways offset of Ks (left) and b (right) from the centre
CYL_LEN, CYL_HALF_W, ROD_LEN = 0.30, 0.05, 0.30   # damper cylinder and rod sizes
VIEW_HALF_WIDTH = 1.5   # half the width of the view around the car [m]
ROAD_X = np.linspace(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH, 400)  # road points, relative to the car [m]

# Colours: the wheel and body keep the same colour in the drawing and the plots.
WHEEL, BODY, ROAD, TYRE = "#eb6834", "#2a78d6", "#898781", "#1baf7a"
INK, SECONDARY, GRID, WARNING = "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b"


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


def describe_modes(car):
    """Return a one-line description of the natural frequencies and damping."""
    try:
        freqs, zetas = natural_modes(car)
    except ValueError:
        return "Natural frequencies: not shown, because a mode is overdamped"
    return (f"Natural frequencies: body {freqs[0]:.2f} Hz (damping {zetas[0]:.2f}), "
            f"wheel hop {freqs[1]:.1f} Hz (damping {zetas[1]:.2f})")


def style(ax):
    """Give an axes hairline gridlines, quiet axis lines and dark text."""
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors=SECONDARY)


def animate_road(car, road, speed, h=1e-3, lead_in_time=0.3, t_after=2.0,
                 playback=0.25, fps=50, hold_frames=40):
    """Simulate the car driving over a road, then play it as an animation.

    The run uses RK4 restarted at every sudden change in the road (see
    ``scenarios.simulate_road``). Opens a window and returns once the user
    closes it. The animation repeats until then.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    road : Bump or Pothole
        Road from ``quarter_car.road``.
    speed : float
        Vehicle speed [m/s].
    h : float, optional
        RK4 step size [s]. Default 1 ms.
    lead_in_time : float, optional
        Time before the tyre reaches the road feature [s]. Default 0.3.
    t_after : float, optional
        Time simulated after the tyre leaves the feature [s]. Default 2.0.
    playback : float, optional
        Animation speed as a fraction of real time. Default 0.25.
    fps : int, optional
        Animation frames per second. Default 50.
    hold_frames : int, optional
        Frames to pause at the end before repeating. Default 40.

    Returns
    -------
    dict
        Key results from :func:`quarter_car.scenarios.road_summary`.
    """
    t, Z, r = simulate_road(car, road, speed, h=h, lead_in_time=lead_in_time,
                            t_after=t_after, split_at_edges=True)
    results = road_summary(car, t, Z, r)
    force = tyre_force(car, Z, r)
    static_load = (car.m1 + car.m2) * G
    static_deflection = static_load / car.Kw   # how much the tyre is squashed at rest [m]

    # Interpolate onto evenly spaced frame times so playback speed is exact.
    frame_t = np.append(np.arange(0.0, t[-1], playback / fps), t[-1])
    frame_x = np.interp(frame_t, t, Z[:, 0])
    frame_y = np.interp(frame_t, t, Z[:, 1])
    frame_r = road_input(road, speed, lead_in_time)(frame_t)

    # ---- Figure layout: still parts are drawn once, moving parts are animated.
    fig = plt.figure(figsize=(13, 6.5))
    ax_car = fig.add_axes([0.06, 0.22, 0.42, 0.70])
    ax_disp = fig.add_axes([0.56, 0.60, 0.41, 0.32])
    ax_force = fig.add_axes([0.56, 0.22, 0.41, 0.30], sharex=ax_disp)

    ax_car.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
    ax_car.set_ylim(-0.25, M2_BASE + M2_H + 0.2)
    ax_car.set_aspect("equal", adjustable="box")
    ax_car.set_xlabel("Distance from the car [m]")
    ax_car.set_ylabel("Height [m]")
    ax_car.set_title(f"{3.6 * speed:.0f} km/h, {road.label}")

    (road_line,) = ax_car.plot([], [], color=ROAD, lw=2, animated=True)
    m1_box = ax_car.add_patch(Rectangle((0, M1_BASE), M1_W, M1_H, color=WHEEL, animated=True))
    m2_box = ax_car.add_patch(Rectangle((0, M2_BASE), M2_W, M2_H, color=BODY, animated=True))
    (tyre_spring,) = ax_car.plot([], [], color=INK, lw=1.5, animated=True)
    (susp_spring,) = ax_car.plot([], [], color=INK, lw=1.5, animated=True)
    (damper,) = ax_car.plot([], [], color=SECONDARY, lw=1.5, animated=True)
    m1_label = ax_car.text(0, 0, r"$m_1$", ha="center", va="center", color="white", animated=True)
    m2_label = ax_car.text(0, 0, r"$m_2$", ha="center", va="center", color="white", animated=True)
    kw_label = ax_car.text(0, 0, r"$K_w$", va="center", animated=True)
    ks_label = ax_car.text(0, 0, r"$K_s$", ha="right", va="center", animated=True)
    b_label = ax_car.text(0, 0, r"$b$", va="center", animated=True)
    time_text = ax_car.text(0.02, 0.97, "", transform=ax_car.transAxes, va="top", animated=True)

    ax_disp.plot(t, 1e3 * r, color=ROAD, lw=1.5, label="Road r")
    ax_disp.plot(t, 1e3 * Z[:, 0], color=WHEEL, lw=1.5, label="Wheel x (m1)")
    ax_disp.plot(t, 1e3 * Z[:, 1], color=BODY, lw=1.5, label="Body y (m2)")
    ax_disp.set_ylabel("Displacement [mm]")
    ax_disp.legend(loc="upper right", fontsize=8, frameon=False)
    ax_disp.set_title(f"Peak body displacement {1e3 * results['peak_body_displacement']:.1f} mm, "
                      f"peak body acceleration {results['peak_body_acceleration']:.1f} m/s$^2$", fontsize=10)
    plt.setp(ax_disp.get_xticklabels(), visible=False)

    ax_force.plot(t, force / 1e3, color=TYRE, lw=1.5)
    ax_force.axhline(static_load / 1e3, color=ROAD, lw=1)
    ax_force.text(t[-1], static_load / 1e3, "static load ", ha="right", va="bottom", fontsize=8, color=SECONDARY)
    ax_force.set_xlim(0, t[-1])
    ax_force.set_xlabel("Time [s]")
    ax_force.set_ylabel("Tyre force [kN]")
    if results["lift_off_time"] is None:
        ax_force.set_title(f"Tyre force stays above {results['min_tyre_force']:.0f} N, "
                           "so the wheel keeps its grip", fontsize=10)
    else:
        ax_force.set_title(f"Warning: the wheel leaves the road for {1e3 * results['airborne_time']:.0f} ms "
                           f"(first at t = {results['lift_off_time']:.2f} s)", fontsize=10, color=WARNING)
    cursors = [ax.axvline(0, color=INK, lw=1, animated=True) for ax in (ax_disp, ax_force)]
    for ax in (ax_disp, ax_force):
        style(ax)

    info = "\n".join([
        f"Inputs: body mass {car.m2:.0f} kg, wheel mass {car.m1:.0f} kg (fixed), "
        f"spring {car.Ks / 1e3:.1f} kN/m, damper {car.b:.0f} N s/m, tyre {car.Kw / 1e3:.0f} kN/m",
        describe_modes(car),
        f"Results: peak body acceleration {results['peak_body_acceleration']:.2f} m/s^2, "
        f"peak suspension travel {1e3 * results['peak_suspension_travel']:.1f} mm, "
        f"tyre force {results['min_tyre_force']:.0f} to {results['max_tyre_force']:.0f} N",
    ])
    fig.text(0.04, 0.04, info, fontsize=10, va="bottom")

    counter = {"frame": 0}

    def update(_frame):
        """Move both masses, both springs, the damper and the road to the next frame."""
        n = len(frame_t)
        k = min(counter["frame"] % (n + hold_frames), n - 1)
        counter["frame"] += 1

        x, y, road_height = frame_x[k], frame_y[k], frame_r[k]
        distance = speed * (frame_t[k] - lead_in_time)       # distance past the start of the feature
        m1_bottom, m2_bottom = M1_BASE + x, M2_BASE + y
        m1_top = m1_bottom + M1_H
        # In the air the tyre spring hangs at its unloaded length, so its bottom is above the road.
        tyre_bottom = max(road_height, x - static_deflection)

        # The car stays at x = 0 and the road slides underneath it.
        road_line.set_data(ROAD_X, road.profile(distance + ROAD_X))
        m1_box.set_xy((-M1_W / 2, m1_bottom))
        m2_box.set_xy((-M2_W / 2, m2_bottom))
        tyre_spring.set_data(*spring_points(0.0, tyre_bottom, m1_bottom))   # Kw: road to m1
        susp_spring.set_data(*spring_points(-SIDE, m1_top, m2_bottom))      # Ks: m1 to m2
        damper.set_data(*damper_points(SIDE, m1_top, m2_bottom))            # b: m1 to m2

        m1_label.set_position((0.0, m1_bottom + M1_H / 2))
        m2_label.set_position((0.0, m2_bottom + M2_H / 2))
        kw_label.set_position((0.10, (tyre_bottom + m1_bottom) / 2))
        ks_label.set_position((-SIDE - 0.10, (m1_top + m2_bottom) / 2))
        b_label.set_position((SIDE + 0.09, (m1_top + m2_bottom) / 2))

        for cursor in cursors:
            cursor.set_xdata([frame_t[k], frame_t[k]])
        time_text.set_text(f"t = {frame_t[k]:.2f} s")   # kept short: long text is slow to redraw
        return (road_line, m1_box, m2_box, tyre_spring, susp_spring, damper,
                m1_label, m2_label, kw_label, ks_label, b_label, time_text, *cursors)

    # Keep a reference to the animation while the window is open.
    animation = FuncAnimation(fig, update, interval=1000 / fps, blit=True, cache_frame_data=False)
    plt.show()
    plt.close(fig)
    return results
