"""Animation of the quarter car driving over a half-sine bump.

The left panel draws the same two-mass model that is simulated, laid out like
the simplified system diagram: the body ``m2`` sits on the suspension spring
``Ks`` and damper ``b``, which sit on the wheel and axle ``m1``, which sits on
the tyre spring ``Kw`` touching the road. The car stays in the centre of the
view and the road moves underneath it. The right panel shows road, wheel and
body displacement against time, and the text underneath lists the inputs and
key results.

For smooth playback the animation uses blitting: the axes, labels and time
histories are drawn once, and only the moving parts are redrawn each frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from quarter_car.model import natural_modes
from quarter_car.road import half_sine_profile
from quarter_car.scenarios import bump_summary, simulate_bump

# Drawing settings (only affect the picture, not the physics). Heights are at rest [m].
M1_BASE, M1_W, M1_H = 0.30, 0.50, 0.15      # bottom height, width and height of m1
M2_BASE, M2_W, M2_H = 0.95, 1.00, 0.25      # bottom height, width and height of m2
SIDE = 0.18             # sideways offset of Ks (left) and b (right) from the centre
CYL_LEN, CYL_HALF_W, ROD_LEN = 0.30, 0.05, 0.30   # damper cylinder and rod sizes
VIEW_HALF_WIDTH = 1.5   # half the width of the view around the car [m]
ROAD_X = np.linspace(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH, 400)  # road points, relative to the car [m]


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


def animate_bump(car, height, length, speed, lead_in_time=0.3, t_after=2.0, h=1e-3,
                 playback=0.25, fps=50, hold_frames=40):
    """Simulate the car crossing a half-sine bump, then play it as an animation.

    Opens a window and returns once the user closes it. The animation repeats
    until then.

    Parameters
    ----------
    car : QuarterCar
        Model parameters.
    height : float
        Peak bump height [m].
    length : float
        Bump length along the road [m].
    speed : float
        Vehicle speed [m/s].
    lead_in_time : float, optional
        Time before the tyre reaches the bump [s]. Default 0.3.
    t_after : float, optional
        Time simulated after the tyre leaves the bump [s]. Default 2.0.
    h : float, optional
        RK4 step size [s]. Default 1 ms.
    playback : float, optional
        Animation speed as a fraction of real time. Default 0.25.
    fps : int, optional
        Animation frames per second. Default 50.
    hold_frames : int, optional
        Frames to pause at the end before repeating. Default 40.

    Returns
    -------
    dict
        Key results from :func:`quarter_car.scenarios.bump_summary`.
    """
    t, Z, r = simulate_bump(car, height, length, speed, lead_in_time, t_after, h)
    results = bump_summary(car, t, Z, r)
    bump_start = speed * lead_in_time

    # Interpolate onto evenly spaced frame times so playback speed is exact.
    frame_t = np.append(np.arange(0.0, t[-1], playback / fps), t[-1])
    frame_x = np.interp(frame_t, t, Z[:, 0])
    frame_y = np.interp(frame_t, t, Z[:, 1])
    frame_r = half_sine_profile(speed * frame_t, height, length, bump_start)

    # ---- Figure layout: still parts are drawn once, moving parts are animated.
    fig = plt.figure(figsize=(13, 6))
    ax_car = fig.add_axes([0.04, 0.22, 0.44, 0.70])
    ax_hist = fig.add_axes([0.56, 0.25, 0.41, 0.66])

    ax_car.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
    ax_car.set_ylim(-0.1, M2_BASE + M2_H + 0.2)
    ax_car.set_aspect("equal", adjustable="box")
    ax_car.set_xlabel("Distance from the car [m]")
    ax_car.set_ylabel("Height [m]")
    ax_car.set_title(f"{3.6 * speed:.0f} km/h over a {1e3 * height:.0f} mm high, "
                     f"{length:.1f} m long bump")

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

    ax_hist.plot(t, 1e3 * r, color="0.3", label="Road r")
    ax_hist.plot(t, 1e3 * Z[:, 0], color="tab:orange", label="Wheel x (m1)")
    ax_hist.plot(t, 1e3 * Z[:, 1], color="tab:blue", label="Body y (m2)")
    cursor = ax_hist.axvline(0, color="red", lw=1, animated=True)
    ax_hist.set_xlim(0, t[-1])
    ax_hist.set_xlabel("Time [s]")
    ax_hist.set_ylabel("Displacement [mm]")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(alpha=0.3)
    if results["lift_off_time"] is None:
        ax_hist.set_title(f"Peak body displacement: {1e3 * results['peak_body_displacement']:.1f} mm")
    else:
        ax_hist.set_title(f"Warning: wheel leaves the road at t = {results['lift_off_time']:.2f} s, "
                          "results after this are not valid", color="red")

    info = "\n".join([
        f"Inputs: body mass {car.m2:.0f} kg, wheel mass {car.m1:.0f} kg (fixed), "
        f"spring {car.Ks / 1e3:.1f} kN/m, damper {car.b:.0f} N s/m, tyre {car.Kw / 1e3:.0f} kN/m",
        describe_modes(car),
        f"Results: peak body acceleration {results['peak_body_acceleration']:.2f} m/s^2, "
        f"peak suspension travel {1e3 * results['peak_suspension_travel']:.1f} mm, "
        f"minimum tyre force {results['min_tyre_force']:.0f} N",
    ])
    fig.text(0.04, 0.04, info, fontsize=10, va="bottom")

    counter = {"frame": 0}

    def update(_frame):
        """Move both masses, both springs, the damper and the road to the next frame."""
        n = len(frame_t)
        k = min(counter["frame"] % (n + hold_frames), n - 1)
        counter["frame"] += 1

        x, y, road_height = frame_x[k], frame_y[k], frame_r[k]
        distance = speed * frame_t[k]
        m1_bottom, m2_bottom = M1_BASE + x, M2_BASE + y
        m1_top = m1_bottom + M1_H

        # The car stays at x = 0 and the road slides underneath it.
        road_line.set_data(ROAD_X, half_sine_profile(distance + ROAD_X, height, length, bump_start))
        m1_box.set_xy((-M1_W / 2, m1_bottom))
        m2_box.set_xy((-M2_W / 2, m2_bottom))
        tyre_spring.set_data(*spring_points(0.0, road_height, m1_bottom))   # Kw: road to m1
        susp_spring.set_data(*spring_points(-SIDE, m1_top, m2_bottom))      # Ks: m1 to m2
        damper.set_data(*damper_points(SIDE, m1_top, m2_bottom))            # b: m1 to m2

        m1_label.set_position((0.0, m1_bottom + M1_H / 2))
        m2_label.set_position((0.0, m2_bottom + M2_H / 2))
        kw_label.set_position((0.10, (road_height + m1_bottom) / 2))
        ks_label.set_position((-SIDE - 0.10, (m1_top + m2_bottom) / 2))
        b_label.set_position((SIDE + 0.09, (m1_top + m2_bottom) / 2))

        cursor.set_xdata([frame_t[k], frame_t[k]])
        time_text.set_text(f"t = {frame_t[k]:.2f} s")   # kept short: long text is slow to redraw
        return (road_line, m1_box, m2_box, tyre_spring, susp_spring, damper,
                m1_label, m2_label, kw_label, ks_label, b_label, time_text, cursor)

    # Keep a reference to the animation while the window is open.
    animation = FuncAnimation(fig, update, interval=1000 / fps, blit=True, cache_frame_data=False)
    plt.show()
    plt.close(fig)
    return results
