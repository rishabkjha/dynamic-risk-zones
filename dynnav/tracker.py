"""Multi-object tracking on the ground plane with a constant-velocity Kalman filter per object.

Tracking happens in metres (x forward, y left) instead of image boxes, so each track directly
carries a position, a velocity and a history: the inputs the zone methods and the predictor need.
"""
import itertools

import numpy as np


class Track:
    _ids = itertools.count(1)

    def __init__(self, pos, label, t, pos_std):
        self.id = next(Track._ids)
        self.label = label
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])          # x, y, vx, vy
        self.P = np.diag([pos_std ** 2, pos_std ** 2, 1.0, 1.0])
        self.t = t
        self.last_seen = t
        self.hits = 1
        self.history = [(t, self.x[0], self.x[1])]

    @property
    def pos(self):
        return self.x[:2]

    @property
    def vel(self):
        return self.x[2:]

    def predict(self, t, accel_std):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * accel_std ** 2
        self.t = t

    def _update(self, z, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def update(self, pos, pos_std, lateral=None):
        """pos: measured (x, y). lateral: optional (unit direction, speed) from optical flow."""
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self._update(np.asarray(pos, float), H, np.eye(2) * pos_std ** 2)
        if lateral is not None:
            direction, speed, speed_std = lateral
            H = np.array([[0, 0, direction[0], direction[1]]])
            self._update(np.array([speed]), H, np.array([[speed_std ** 2]]))
        self.hits += 1
        self.last_seen = self.t
        self.history.append((self.t, self.x[0], self.x[1]))
        if len(self.history) > 200:
            del self.history[:50]

    def past(self, dt, n):
        """n positions spaced dt apart, ending now (the predictor's input), shape (n, 2).

        A track younger than (n-1)*dt is extended backwards with its current velocity.
        """
        h = np.array(self.history)
        if self.t > h[-1, 0]:                      # coasting: add where the filter thinks it is now
            h = np.vstack([h, [self.t, self.x[0], self.x[1]]])
        times = self.t - dt * np.arange(n - 1, -1, -1)
        if len(h) == 1:
            return self.pos[None, :] + (times - self.t)[:, None] * self.vel[None, :]
        xs = np.interp(times, h[:, 0], h[:, 1])
        ys = np.interp(times, h[:, 0], h[:, 2])
        early = times < h[0, 0]
        xs[early] = h[0, 1] + (times[early] - h[0, 0]) * self.vel[0]
        ys[early] = h[0, 2] + (times[early] - h[0, 0]) * self.vel[1]
        return np.stack([xs, ys], axis=1)


class Tracker:
    def __init__(self, gate=1.0, max_age=1.0, min_hits=3, accel_std=1.5):
        self.gate = gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.accel_std = accel_std
        self.tracks = []

    def step(self, t, measurements):
        """measurements: list of dicts with pos, label, pos_std and optionally lateral.

        Returns the confirmed tracks. Matching is greedy nearest-first, which is as good as the
        Hungarian algorithm for the handful of objects a camera sees at once.
        """
        for tr in self.tracks:
            tr.predict(t, self.accel_std)
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, m in enumerate(measurements):
                if m["label"] != tr.label:
                    continue
                d = float(np.linalg.norm(tr.pos - m["pos"]))
                if d < self.gate + 0.5 * tr.P[0, 0] ** 0.5:
                    pairs.append((d, i, j))
        used_t, used_m = set(), set()
        for d, i, j in sorted(pairs):
            if i in used_t or j in used_m:
                continue
            used_t.add(i)
            used_m.add(j)
            m = measurements[j]
            self.tracks[i].update(m["pos"], m["pos_std"], m.get("lateral"))
        for j, m in enumerate(measurements):
            if j not in used_m:
                self.tracks.append(Track(m["pos"], m["label"], t, m["pos_std"]))
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_age]
        return self.confirmed()

    def confirmed(self):
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]
