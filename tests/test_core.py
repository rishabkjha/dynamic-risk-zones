"""Quick checks of the maths: python -m pytest tests/  (or python tests/test_core.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from dynnav import zones  # noqa: E402
from dynnav.planner import Planner  # noqa: E402
from dynnav.tracker import Tracker  # noqa: E402


def walker(tracker, speed=(0.0, 1.0), steps=15, dt=0.1):
    for k in range(steps):
        pos = np.array([3.0, -1.5]) + np.array(speed) * k * dt
        tracker.step(k * dt, [dict(pos=pos, label="person", pos_std=0.03)])
    return tracker.confirmed()


def test_tracker_learns_velocity():
    tr = walker(Tracker())[0]
    assert np.allclose(tr.vel, [0.0, 1.0], atol=0.15), tr.vel


def test_reach_zone_grows_with_class_speed():
    tracks = walker(Tracker())
    z = zones.build("reach", tracks, np.array([0.0, 1.0, 2.0]))
    typ, vmax, body = config.CLASS_MOTION["person"]
    assert np.isclose(z.axes[0, 2, 2, 0], body + config.ROBOT_RADIUS + 2.0 * vmax)


def test_planner_goes_around_a_standing_person():
    tracker = Tracker()
    for k in range(5):
        tracker.step(0.1 * k, [dict(pos=np.array([1.5, 0.0]), label="person", pos_std=0.03)])
    planner = Planner()
    z = zones.build("static", tracker.confirmed(), planner.times)
    cmd, paths, best, cost = planner.plan(np.zeros(3), 0.5, (4.0, 0.0), z)
    assert cost.max() < config.ZONE_COSTS[0]              # the chosen path never enters the no-go zone
    assert abs(cmd[1]) > 0.1                               # so it has to turn


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
