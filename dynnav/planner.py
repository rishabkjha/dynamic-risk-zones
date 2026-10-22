"""Sampling planner: try a few hundred short drives, keep the fastest one that stays out of the zones.

Each candidate is a constant (speed, turn rate) command held for PLAN_HORIZON seconds, with the
speed ramping at the robot's acceleration limit, like the Dynamic Window Approach. Every
candidate is checked against the zones AT THE TIME the robot would be there, which is what lets
it pass behind a walking person instead of freezing in front of their whole possible path.
Re-planned every PLAN_DT, so only the first step of the best candidate is ever driven.
"""
import numpy as np

import config


class Planner:
    def __init__(self, n_v=7, n_w=13):
        self.dt = config.PLAN_DT
        self.K = int(round(config.PLAN_HORIZON / self.dt))
        self.times = self.dt * np.arange(self.K + 1)
        v, w = np.meshgrid(np.linspace(0, config.ROBOT_MAX_V, n_v),
                           np.linspace(-config.ROBOT_MAX_W, config.ROBOT_MAX_W, n_w))
        self.cmds = np.stack([v.ravel(), w.ravel()], 1)

    def rollouts(self, pose, v0):
        """pose (x, y, heading), current speed -> positions (M, K+1, 2) of every candidate."""
        M = len(self.cmds)
        xy = np.zeros((M, self.K + 1, 2))
        xy[:, 0] = pose[:2]
        th = np.full(M, pose[2])
        v = np.full(M, v0)
        dv = config.ROBOT_MAX_ACC * self.dt
        for k in range(1, self.K + 1):
            v = v + np.clip(self.cmds[:, 0] - v, -dv, dv)
            th = th + self.cmds[:, 1] * self.dt
            xy[:, k, 0] = xy[:, k - 1, 0] + v * np.cos(th) * self.dt
            xy[:, k, 1] = xy[:, k - 1, 1] + v * np.sin(th) * self.dt
        return xy

    def plan(self, pose, v0, goal, zones, risk_weight=2.0):
        """-> (v, w) command, all candidate paths, index of the chosen one, its costs."""
        xy = self.rollouts(pose, v0)
        cost = zones.cost(xy)                                            # (M, K+1)
        dist = np.linalg.norm(xy - np.asarray(goal)[None, None, :], axis=-1)
        # Once a candidate reaches the goal it counts as staying there.
        reached = np.cumsum(dist < 0.2, axis=1) > 0
        cost = np.where(reached, 0.0, cost)
        score = dist.min(axis=1) + risk_weight * cost[:, 1:].sum(axis=1) * self.dt
        score += 0.05 * np.abs(self.cmds[:, 1])
        lethal = cost[:, 1:] >= config.ZONE_COSTS[0]
        hit = lethal.any(axis=1)
        first_hit = np.where(hit, lethal.argmax(axis=1), self.K)
        # A candidate that enters a no-go zone is only chosen when all do; then the one that
        # stays out the longest wins (usually braking or turning away).
        score += np.where(hit, 100.0 + 100.0 * (self.K - first_hit) / self.K, 0.0)
        best = int(np.argmin(score))
        return self.cmds[best], xy, best, cost[best]
