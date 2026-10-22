"""Risk zones around moving objects, as a cost the planner can look up at any place and time.

Every object gets three nested zones for every future time step: inner (no-go), middle and outer,
each an ellipse (a circle is the special case of equal axes). Outside all of them is free.
How the zones are placed and how fast they grow is what the four methods differ in:

  static   only where the object is now (what a plain costmap does)
  reach    the object's reachable set: circles growing at 0.5x typical, typical and maximum speed
           of its class (a person can only be so far away after t seconds)
  cv       constant-velocity prediction; ellipses at 1, 2, 3 sigma of its measured error
  learned  the GRU predictor; ellipses at 1, 2, 3 sigma of its own predicted uncertainty

All of them are inflated by the object's body radius plus the robot's radius, so the robot
can be treated as a point.
"""
import numpy as np

import config
from dynnav import data

METHODS = ("static", "reach", "cv", "learned")


def class_motion(label):
    return config.CLASS_MOTION.get(label, config.DEFAULT_MOTION)


class Zones:
    """centers (N, K, 2), theta (N,), axes (N, K, 3, 2) [level, (along, across)] over times (K,)."""

    def __init__(self, times, centers, theta, axes):
        self.times = times
        self.centers = centers
        self.theta = theta
        self.axes = axes

    @property
    def empty(self):
        return len(self.centers) == 0

    def cost(self, points):
        """points (M, K, 2) at self.times -> cost (M, K): the highest zone cost each point is in."""
        M, K = points.shape[:2]
        if self.empty:
            return np.zeros((M, K))
        rel = points[None, :, :, :] - self.centers[:, None, :, :]           # (N, M, K, 2)
        c, s = np.cos(self.theta)[:, None, None], np.sin(self.theta)[:, None, None]
        u = c * rel[..., 0] + s * rel[..., 1]                              # along the heading
        v = -s * rel[..., 0] + c * rel[..., 1]                             # across it
        cost = np.zeros((len(self.centers), M, K))
        for level in range(2, -1, -1):                                     # outer first
            a = self.axes[:, None, :, level, 0]
            b = self.axes[:, None, :, level, 1]
            inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
            cost[inside] = config.ZONE_COSTS[level]
        return cost.max(axis=0)


def _static_axes(R, K):
    radii = np.array([R, R + 0.3, R + 0.6])
    return np.broadcast_to(np.stack([radii, radii], -1), (K, 3, 2))


def build(method, tracks, times, predictor=None):
    """Zones for the confirmed tracks at the planning times (0, dt, 2dt, ...)."""
    tracks = [t for t in tracks if t.label not in config.IGNORE_CLASSES]
    K = len(times)
    N = len(tracks)
    centers = np.zeros((N, K, 2))
    theta = np.zeros(N)
    axes = np.zeros((N, K, 3, 2))
    if N == 0:
        return Zones(times, centers, theta, axes)

    moving = []
    for i, tr in enumerate(tracks):
        typ, vmax, body = class_motion(tr.label)
        R = body + config.ROBOT_RADIUS
        centers[i] = tr.pos
        axes[i] = _static_axes(R, K)
        if typ == 0.0 or method == "static":
            continue
        if method == "reach":
            for level, speed in enumerate((0.5 * typ, typ, vmax)):
                axes[i, :, level, :] = (R + speed * times)[:, None]
        else:
            moving.append(i)

    if moving and method in ("cv", "learned"):
        past = np.stack([tracks[i].past(predictor.dt, config.PRED_OBS) for i in moving])
        pos = np.array([t.pos for t in tracks])
        vel = np.array([t.vel for t in tracks])
        others = [np.arange(N) != i for i in moving]
        neighbours = np.stack([data.neighbour_features(pos[i], pos[o], vel[o])
                               for i, o in zip(moving, others)])
        pred = predictor(past, neighbours)
        t_pred = predictor.dt * np.arange(0, pred.mean.shape[1] + 1)
        for j, i in enumerate(moving):
            R = class_motion(tracks[i].label)[2] + config.ROBOT_RADIUS
            mean = np.vstack([tracks[i].pos, pred.mean[j]])
            sig = np.vstack([[0.05, 0.05], pred.sigma[j]])
            for d in range(2):
                centers[i, :, d] = np.interp(times, t_pred, mean[:, d])
            sig_t = np.stack([np.interp(times, t_pred, sig[:, d]) for d in range(2)], -1)
            for level, k in enumerate(config.ZONE_SIGMAS):
                axes[i, :, level, :] = R + k * sig_t
            theta[i] = pred.theta[j]
    return Zones(times, centers, theta, axes)
