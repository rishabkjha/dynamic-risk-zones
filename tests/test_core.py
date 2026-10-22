"""Quick checks of the maths: python -m unittest  (or python tests/test_core.py)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from dynnav import sim, zones  # noqa: E402
from dynnav.planner import Planner  # noqa: E402
from dynnav.tracker import Tracker  # noqa: E402


def walker(tracker, speed=(0.0, 1.0), steps=15, dt=0.1):
    for k in range(steps):
        pos = np.array([3.0, -1.5]) + np.array(speed) * k * dt
        tracker.step(k * dt, [dict(pos=pos, label="person", pos_std=0.03)])
    return tracker.confirmed()


class TestCore(unittest.TestCase):
    def test_tracker_learns_velocity(self):
        tr = walker(Tracker())[0]
        self.assertTrue(np.allclose(tr.vel, [0.0, 1.0], atol=0.15), tr.vel)

    def test_reach_zone_grows_with_class_speed(self):
        tracks = walker(Tracker())
        z = zones.build("reach", tracks, np.array([0.0, 1.0, 2.0]))
        typ, vmax, body = config.CLASS_MOTION["person"]
        self.assertTrue(np.isclose(z.axes[0, 2, 2, 0], body + config.ROBOT_RADIUS + 2.0 * vmax))

    def test_planner_goes_around_a_standing_person(self):
        tracker = Tracker()
        for k in range(5):
            tracker.step(0.1 * k, [dict(pos=np.array([1.5, 0.0]), label="person", pos_std=0.03)])
        planner = Planner()
        z = zones.build("static", tracker.confirmed(), planner.times)
        cmd, paths, best, cost = planner.plan(np.zeros(3), 0.5, (4.0, 0.0), z)
        self.assertLess(cost.max(), config.ZONE_COSTS[0])     # the chosen path never enters the no-go zone
        self.assertGreater(abs(cmd[1]), 0.1)                  # so it has to turn

    def test_collision_attribution(self):
        pose = np.array([0.0, 0.0, 0.0])                      # robot at the origin facing +x
        self.assertTrue(sim.robot_at_fault(pose, 0.5, [0.4, 0.1]))       # drove into a person ahead
        self.assertFalse(sim.robot_at_fault(pose, 0.05, [0.4, 0.1]))     # standing still: person's fault
        self.assertFalse(sim.robot_at_fault(pose, 0.5, [-0.4, 0.0]))     # hit from behind
        self.assertFalse(sim.robot_at_fault(pose, 0.5, [0.1, 0.4]))      # from the side (outside the 90° cone)
        turned = np.array([0.0, 0.0, np.pi])                  # facing -x: wrap-around of the bearing
        self.assertTrue(sim.robot_at_fault(turned, 0.5, [-0.4, -0.05]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
