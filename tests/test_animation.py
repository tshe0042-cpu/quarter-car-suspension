"""Tests for quarter_car.animation. Run with: python -m pytest -q"""

from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from quarter_car.animation import _ClosableAnimation


def test_animation_stops_once_its_window_is_closed():
    """Regression test for a bug seen when closing the window: one last timer tick
    raised AttributeError ('FigureCanvasBase' object has no attribute 'restore_region').

    Follows the real sequence: the window opens, frames are drawn, the window is
    closed, then one more timer tick arrives.
    """
    fig = Figure()
    manager = FigureManagerBase(FigureCanvasAgg(fig), 1)    # stands in for the window
    (line,) = fig.add_subplot().plot([0, 1], [0, 1], animated=True)
    animation = _ClosableAnimation(fig, lambda frame: (line,), interval=20, blit=True,
                                   cache_frame_data=False)
    fig.canvas.draw()                    # first draw starts the animation, as when the window opens
    assert animation._step() is True     # a normal tick while the window is open
    manager.destroy()                    # what clicking the close button does to the figure
    assert animation._step() is False    # stops cleanly instead of raising an error
