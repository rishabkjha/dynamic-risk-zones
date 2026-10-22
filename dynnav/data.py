"""ETH/UCY pedestrian trajectories: loading, and cutting them into (past, future) training samples.

The five scenes (eth, hotel, univ, zara1, zara2) are the standard benchmark for pedestrian
motion prediction. Positions are in metres, one row every 0.4 s: frame, person id, x, y.
The usual protocol is leave-one-scene-out: train on four scenes, test on the fifth.
"""
import glob
import os

import numpy as np

NEIGHBOURS = 6
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
URL = "https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/master/experiments/pedestrians/raw"
FILES = ["biwi_eth", "biwi_hotel", "crowds_zara01", "crowds_zara02", "crowds_zara03",
         "students001", "students003", "uni_examples"]


def load_file(path):
    """-> {person id: array of (frame, x, y)} sorted by frame."""
    raw = np.loadtxt(path, dtype=np.float64)
    people = {}
    for pid in np.unique(raw[:, 1]):
        rows = raw[raw[:, 1] == pid]
        rows = rows[np.argsort(rows[:, 0])]
        people[int(pid)] = rows[:, [0, 2, 3]]
    return people


def frame_step(people):
    """Frame number difference between consecutive rows (10 in most files)."""
    diffs = np.concatenate([np.diff(p[:, 0]) for p in people.values() if len(p) > 1])
    return float(np.median(diffs))


def scene_files(root, scene, split):
    return sorted(glob.glob(os.path.join(root, scene, split, "*.txt")))


def windows(people, length):
    """All runs of `length` consecutive rows without a gap -> array (N, length, 2)."""
    step = frame_step(people)
    out = []
    for p in people.values():
        gaps = np.where(np.abs(np.diff(p[:, 0]) - step) > 0.5)[0]
        for seg in np.split(p, gaps + 1):
            for s in range(0, len(seg) - length + 1):
                out.append(seg[s:s + length, 1:3])
    return np.array(out, np.float32).reshape(-1, length, 2)


def neighbour_features(own_pos, others_pos, others_vel, k=NEIGHBOURS, radius=5.0):
    """The k nearest other people around one person -> (k, 5): dx, dy, vx, vy, present (1/0).

    World frame, relative to own_pos; empty rows (fewer than k people near) are all zero.
    """
    out = np.zeros((k, 5), np.float32)
    if len(others_pos) == 0:
        return out
    rel = np.asarray(others_pos) - own_pos
    dist = np.linalg.norm(rel, axis=1)
    order = [i for i in np.argsort(dist) if dist[i] < radius][:k]
    for row, i in enumerate(order):
        out[row, :2] = rel[i]
        out[row, 2:4] = others_vel[i]
        out[row, 4] = 1.0
    return out


def windows_social(people, length, obs, k=NEIGHBOURS):
    """Like windows(), plus the neighbours of each person at the last observed step.

    -> samples (N, length, 2), neighbours (N, k, 5).
    """
    step = frame_step(people)
    dt = 0.4
    at = {}                                         # frame -> {pid: (x, y, vx, vy)}
    for pid, p in people.items():
        for j in range(1, len(p)):
            if abs(p[j, 0] - p[j - 1, 0] - step) < 0.5:
                v = (p[j, 1:] - p[j - 1, 1:]) / dt
                at.setdefault(p[j, 0], {})[pid] = (p[j, 1], p[j, 2], v[0], v[1])
    samples, nbs = [], []
    for pid, p in people.items():
        gaps = np.where(np.abs(np.diff(p[:, 0]) - step) > 0.5)[0]
        for seg in np.split(p, gaps + 1):
            for s in range(0, len(seg) - length + 1):
                frame = seg[s + obs - 1, 0]
                others = [v for q, v in at.get(frame, {}).items() if q != pid]
                o = np.array(others, np.float64).reshape(-1, 4)
                samples.append(seg[s:s + length, 1:3])
                nbs.append(neighbour_features(seg[s + obs - 1, 1:3], o[:, :2], o[:, 2:], k))
    return (np.array(samples, np.float32).reshape(-1, length, 2),
            np.array(nbs, np.float32).reshape(-1, k, 5))


def load_split(root, scene, split, length, obs=None):
    """Samples for one leave-one-out split, e.g. scene 'eth', split 'train' = the other 4 scenes.

    With obs given, returns (samples, neighbours) instead of just samples.
    """
    files = scene_files(root, scene, split)
    if obs is None:
        parts = [windows(load_file(f), length) for f in files]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.zeros((0, length, 2), np.float32)
    parts = [windows_social(load_file(f), length, obs) for f in files]
    parts = [p for p in parts if len(p[0])]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def download(root):
    import urllib.request
    n = 0
    for scene in SCENES:
        for split in ["train", "val", "test"]:
            for name in FILES:
                for suffix in ["", "_train", "_val"]:
                    rel = f"{scene}/{split}/{name}{suffix}.txt"
                    dest = os.path.join(root, rel)
                    if os.path.exists(dest):
                        n += 1
                        continue
                    try:
                        data = urllib.request.urlopen(f"{URL}/{rel}", timeout=30).read()
                    except Exception:
                        continue
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(data)
                    n += 1
    return n
