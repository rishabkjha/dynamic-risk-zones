"""Crowd simulator: a robot crosses real recorded pedestrian traffic (ETH/UCY) to a goal.

The pedestrians replay their recorded paths and do not react to the robot, the usual setup
for comparing navigation methods on these datasets: it tests whether the ROBOT avoids them.
The robot only knows what a camera would tell it: people inside the field of view and range,
with stereo-like position noise that grows with distance, missed detections, and (optionally)
an optical-flow measurement of each person's sideways speed. Everything after that is the real
pipeline: tracker -> zones (one of the four methods) -> planner.
"""
from dataclasses import dataclass

import numpy as np

import config
from dynnav import data, zones
from dynnav.planner import Planner
from dynnav.tracker import Tracker

PED_RADIUS = 0.2        # physical body radius used to count a collision


class Scene:
    def __init__(self, path):
        people = data.load_file(path)
        step = data.frame_step(people)
        self.tracks = {pid: np.column_stack([p[:, 0] / step * config.PRED_DT, p[:, 1:]])
                       for pid, p in people.items() if len(p) >= 2}
        allp = np.concatenate(list(self.tracks.values()))
        self.t_min, self.t_max = allp[:, 0].min(), allp[:, 0].max()
        self.lo, self.hi = allp[:, 1:].min(0), allp[:, 1:].max(0)

    def at(self, t):
        """Positions and velocities of everyone present at time t: (ids, pos (n,2), vel (n,2))."""
        ids, pos, vel = [], [], []
        for pid, tr in self.tracks.items():
            if tr[0, 0] <= t <= tr[-1, 0]:
                x = np.interp(t, tr[:, 0], tr[:, 1])
                y = np.interp(t, tr[:, 0], tr[:, 2])
                k = min(max(np.searchsorted(tr[:, 0], t), 1), len(tr) - 1)
                v = (tr[k, 1:] - tr[k - 1, 1:]) / max(tr[k, 0] - tr[k - 1, 0], 1e-6)
                ids.append(pid)
                pos.append((x, y))
                vel.append(v)
        return ids, np.array(pos).reshape(-1, 2), np.array(vel).reshape(-1, 2)


@dataclass
class Episode:
    t0: float
    start: np.ndarray
    goal: np.ndarray


def make_episodes(scene, n, length=10.0, seed=0):
    """Random crossings through the busiest parts of the scene."""
    rng = np.random.default_rng(seed)
    eps = []
    tries = 0
    while len(eps) < n and tries < n * 200:
        tries += 1
        t0 = rng.uniform(scene.t_min, scene.t_max - 30.0)
        _, pos, _ = scene.at(t0 + 5.0)
        if len(pos) < 2:
            continue
        centre = pos[rng.integers(len(pos))]
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)]) * length / 2
        start, goal = centre - d, centre + d
        _, p0, _ = scene.at(t0)
        if len(p0) and np.min(np.linalg.norm(p0 - start, axis=1)) < 1.0:
            continue
        eps.append(Episode(t0, start, goal))
    return eps


@dataclass
class Camera:
    fov_deg: float = 70.0       # the Lenovo webcams see about 60-70 degrees across
    range_m: float = 8.0
    miss_prob: float = 0.1
    use_flow: bool = True


def sense(robot_pose, pos, vel, cam, rng):
    """Detections a stereo camera on the robot would produce, in world coordinates."""
    out = []
    heading = np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])])
    left = np.array([-heading[1], heading[0]])
    for p, v in zip(pos, vel):
        rel = p - robot_pose[:2]
        dist = float(np.linalg.norm(rel))
        if dist > cam.range_m or dist < 0.3:
            continue
        bearing = np.degrees(np.arctan2(rel @ left, rel @ heading))
        if abs(bearing) > cam.fov_deg / 2 or rng.random() < cam.miss_prob:
            continue
        # Stereo error grows with distance squared (about 1 cm at 1 m, 25 cm at 5 m on this rig).
        std_along = 0.02 + 0.01 * dist ** 2
        ray = rel / dist
        noisy = p + ray * rng.normal(0, std_along) + np.array([-ray[1], ray[0]]) * rng.normal(0, 0.03)
        m = dict(pos=noisy, label="person", pos_std=max(std_along, 0.05))
        if cam.use_flow:
            side = np.array([-ray[1], ray[0]])                    # perpendicular to the view ray
            m["lateral"] = (side, float(v @ side) + rng.normal(0, 0.08), 0.12)
        out.append(m)
    return out


def run_episode(scene, ep, method, predictor, cam=Camera(), seed=0, record=False):
    rng = np.random.default_rng(seed)
    planner = Planner()
    tracker = Tracker(gate=config.TRACK_GATE_M, max_age=config.TRACK_MAX_AGE_S,
                      min_hits=config.TRACK_MIN_HITS)
    sim_dt = 0.1
    steps_per_plan = int(round(config.PLAN_DT / sim_dt))
    to_goal = ep.goal - ep.start
    pose = np.array([*ep.start, np.arctan2(to_goal[1], to_goal[0])])
    v = 0.0
    cmd = np.zeros(2)
    timeout = np.linalg.norm(to_goal) / config.ROBOT_MAX_V * 3.0 + 10.0
    t = 0.0
    min_dist = np.inf
    collided = False
    path_len = 0.0
    frozen = 0.0
    frames = []
    reached = False
    k = 0
    while t < timeout:
        ids, pos, vel = scene.at(ep.t0 + t)
        if len(pos):
            d = np.linalg.norm(pos - pose[:2], axis=1) - PED_RADIUS - config.ROBOT_RADIUS
            min_dist = min(min_dist, float(d.min()))
            if d.min() < 0:
                collided = True
        if np.linalg.norm(ep.goal - pose[:2]) < 0.3:
            reached = True
            break
        if k % steps_per_plan == 0:
            tracks = tracker.step(t, sense(pose, pos, vel, cam, rng))
            z = zones.build(method, tracks, planner.times, predictor)
            cmd, xy, best, _ = planner.plan(pose, v, ep.goal, z)
            if record:
                frames.append(dict(t=t, pose=pose.copy(), peds=pos.copy(), tracks=[
                    (tr.id, tr.pos.copy()) for tr in tracks], zones=z, paths=xy, best=best))
        dv = config.ROBOT_MAX_ACC * sim_dt
        v = v + np.clip(cmd[0] - v, -dv, dv)
        pose[2] += cmd[1] * sim_dt
        step = v * sim_dt * np.array([np.cos(pose[2]), np.sin(pose[2])])
        pose[:2] += step
        path_len += float(np.linalg.norm(step))
        if v < 0.05:
            frozen += sim_dt
        t += sim_dt
        k += 1
    result = dict(success=reached and not collided, reached=reached, collision=collided,
                  time=t if reached else np.nan, min_dist=min_dist, path_len=path_len,
                  frozen=frozen)
    return (result, frames) if record else result
