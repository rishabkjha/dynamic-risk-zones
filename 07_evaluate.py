"""Step 7: compare the four zone methods by driving a simulated robot through real crowds.

    python 07_evaluate.py                         # all 5 scenes, 40 crossings each, ~10 min
    python 07_evaluate.py --episodes 10           # quick check
    python 07_evaluate.py --video                 # also saves one video per method to results/
    python 07_evaluate.py --fov 360               # a 360-degree sensor instead of the webcams
    python 07_evaluate.py --no-flow               # tracker without the optical-flow speed

Every method sees exactly the same crossings, sensor noise and missed detections (same seeds).
In each scene the learned predictor is the one trained WITHOUT that scene (leave-one-out), so
the crowd is new to it. Writes results/nav_metrics.csv and results/nav_metrics.md.
"""
import argparse
import os
import time

import cv2
import numpy as np

import config
from dynnav import data, sim, zones
from dynnav.predictor import ConstantVelocity, GRUPredictor
from dynnav.viz import TopDown


def predictor_for(method, scene):
    path = f"models/predictor_{scene}.npz"
    if not os.path.exists(path):
        path = config.PREDICTOR_FILE
    if method == "learned":
        return GRUPredictor.load(path)
    if method == "cv":
        return ConstantVelocity.load(path)
    return None


def save_video(path, scene, ep, method, predictor, cam):
    result, frames = sim.run_episode(scene, ep, method, predictor, cam, seed=0, record=True)
    view = TopDown(size=480, metres=10.0, follow=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (480, 480))
    for f in frames:
        tracks = [type("T", (), dict(pos=p, vel=np.zeros(2), id=i, label="person"))() for i, p in f["tracks"]]
        img = view.draw(f["pose"], ep.goal, tracks, f["zones"], f["paths"], f["best"], f["peds"],
                        title=f"{method}  t={f['t']:.1f}s")
        out.write(img)
    out.release()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--scenes", nargs="+", default=data.SCENES)
    ap.add_argument("--methods", nargs="+", default=list(zones.METHODS))
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cam = sim.Camera(fov_deg=args.fov, use_flow=not args.no_flow)
    os.makedirs("results", exist_ok=True)

    rows = []
    for scene_name in args.scenes:
        files = data.scene_files(args.data, scene_name, "test")
        scene = sim.Scene(files[0])
        episodes = sim.make_episodes(scene, args.episodes, seed=1)
        for method in args.methods:
            pred = predictor_for(method, scene_name)
            t0 = time.time()
            res = [sim.run_episode(scene, ep, method, pred, cam, seed=i) for i, ep in enumerate(episodes)]
            r = summarise(res)
            rows.append(dict(scene=scene_name, method=method, **r))
            print(f"[{scene_name:5s}] {method:8s} success {100 * r['success']:5.1f}%  "
                  f"collisions {100 * r['collision']:5.1f}%  time {r['time']:5.1f}s  "
                  f"min dist {r['min_dist']:.2f}m  frozen {r['frozen']:.1f}s  ({time.time() - t0:.0f}s)",
                  flush=True)
            if args.video:
                save_video(f"results/{scene_name}_{method}.mp4", scene, episodes[0], method, pred, cam)
    write(rows, args.tag)


def summarise(res):
    return dict(success=np.mean([r["success"] for r in res]),
                collision=np.mean([r["collision"] for r in res]),
                reached=np.mean([r["reached"] for r in res]),
                time=np.nanmean([r["time"] for r in res]) if any(r["reached"] for r in res) else np.nan,
                min_dist=np.mean([r["min_dist"] for r in res]),
                path_len=np.mean([r["path_len"] for r in res]),
                frozen=np.mean([r["frozen"] for r in res]))


def write(rows, tag):
    name = "results/nav_metrics" + (f"_{tag}" if tag else "")
    keys = ["success", "collision", "reached", "time", "min_dist", "path_len", "frozen"]
    with open(name + ".csv", "w") as f:
        f.write("scene,method," + ",".join(keys) + "\n")
        for r in rows:
            f.write(f"{r['scene']},{r['method']}," + ",".join(f"{r[k]:.4f}" for k in keys) + "\n")
    with open(name + ".md", "w") as f:
        f.write("| scene | method | success | collisions | time to goal (s) | closest gap (m) | "
                "standing still (s) |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['scene']} | {r['method']} | {100 * r['success']:.0f}% | "
                    f"{100 * r['collision']:.0f}% | {r['time']:.1f} | {r['min_dist']:.2f} | "
                    f"{r['frozen']:.1f} |\n")
        for m in dict.fromkeys(r["method"] for r in rows):
            sub = [r for r in rows if r["method"] == m]
            f.write(f"| **all** | **{m}** | **{100 * np.mean([r['success'] for r in sub]):.0f}%** | "
                    f"**{100 * np.mean([r['collision'] for r in sub]):.0f}%** | "
                    f"**{np.nanmean([r['time'] for r in sub]):.1f}** | "
                    f"**{np.mean([r['min_dist'] for r in sub]):.2f}** | "
                    f"**{np.mean([r['frozen'] for r in sub]):.1f}** |\n")
    print("wrote", name + ".md")


if __name__ == "__main__":
    main()
