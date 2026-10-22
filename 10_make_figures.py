"""Step 10: the figures for the report, from the results/ files (PNG, into docs/figures/).

    python 10_make_figures.py

Needs results/ from 06, 07 (Day 1 runs and ablations), 09 and the live replays in results/live/;
a figure whose input is missing is skipped with a message. The simulator snapshots are recomputed (about a minute).
Each method keeps one colour in every figure.
"""
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import config  # noqa: E402
from dynnav import data, sim  # noqa: E402
from dynnav.predictor import ConstantVelocity, GRUPredictor  # noqa: E402

OUT = "docs/figures"
COLOURS = {"static": "#2a78d6", "reach": "#eb6834", "cv": "#1baf7a", "learned": "#eda100",
           "learned_nosocial": "#e87ba4"}
PRED_COLOURS = {"constant velocity": COLOURS["cv"], "GRU": COLOURS["learned_nosocial"],
                "social GRU (ours)": COLOURS["learned"]}
NAMES = {"static": "static", "reach": "reach", "cv": "cv", "learned": "learned\n(social GRU)",
         "learned_nosocial": "learned, no social"}
MAIN = ["static", "reach", "cv", "learned"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ZONE_FILL = ["#d03b3b", "#ec835a", "#fab219"]              # inner, middle, outer

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def overall(rows, method, key):
    vals = [float(r[key]) for r in rows if r["method"] == method]
    return float(np.nanmean(vals)) if vals else np.nan


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def label_bars(ax, bars, fmt="{:.0f}%"):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK2)


def nav_outcomes():
    sets = [("360° sensor", read_csv("results/nav_metrics_fov360.csv")),
            ("webcam view (70°)", read_csv("results/nav_metrics_webcam.csv"))]
    sets = [(n, r) for n, r in sets if r]
    if not sets:
        return print("skip nav_outcomes: no nav_metrics csv")
    fig, axes = plt.subplots(len(sets), 2, figsize=(10, 3.6 * len(sets)), squeeze=False)
    x = np.arange(len(MAIN))
    for row, (name, rows) in enumerate(sets):
        ax = axes[row, 0]
        bars = ax.bar(x, [100 * overall(rows, m, "success") for m in MAIN], 0.6,
                      color=[COLOURS[m] for m in MAIN], edgecolor="white", linewidth=2)
        label_bars(ax, bars)
        ax.set_title(f"Reached the goal without contact, {name}")
        ax.set_ylabel("success (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x, [NAMES[m] for m in MAIN])
        ax = axes[row, 1]
        has_fault = "collision_robot_fault" in rows[0]
        if has_fault:
            robot = [100 * overall(rows, m, "collision_robot_fault") for m in MAIN]
            ped = [100 * overall(rows, m, "collision_ped_fault") for m in MAIN]
            ax.bar(x, robot, 0.6, color=[COLOURS[m] for m in MAIN], edgecolor="white", linewidth=2,
                   label="robot drove into person")
            bars = ax.bar(x, ped, 0.6, bottom=robot, color="white", edgecolor=[COLOURS[m] for m in MAIN],
                          hatch="///", linewidth=1.5, label="person walked into robot")
            for b, r_, p_ in zip(bars, robot, ped):
                ax.annotate(f"{r_ + p_:.0f}%", (b.get_x() + b.get_width() / 2, r_ + p_), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=9, color=INK2)
            ax.legend(loc="upper right", frameon=False)
        else:
            bars = ax.bar(x, [100 * overall(rows, m, "collision") for m in MAIN], 0.6,
                          color=[COLOURS[m] for m in MAIN], edgecolor="white", linewidth=2)
            label_bars(ax, bars)
        ax.set_title(f"Collisions, {name}")
        ax.set_ylabel("episodes with contact (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x, [NAMES[m] for m in MAIN])
    fig.suptitle("Crowd benchmark: 40 crossings × 5 ETH/UCY scenes per method", y=1.0)
    fig.tight_layout()
    save(fig, "nav_outcomes.png")


def predictor_per_scene():
    rows = read_csv("results/predictor_metrics.csv")
    if not rows:
        return print("skip predictor_per_scene: no predictor_metrics.csv")
    methods = list(PRED_COLOURS)
    scenes = data.SCENES
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.26
    for ax, key, title in ((axes[0], "ADE", "Average displacement error (ADE)"),
                           (axes[1], "FDE", "Final displacement error at 4.8 s (FDE)")):
        for i, m in enumerate(methods):
            vals = [next(float(r[key]) for r in rows if r["scene"] == s and r["method"] == m) for s in scenes]
            ax.bar(np.arange(len(scenes)) + (i - 1) * w, vals, w, color=PRED_COLOURS[m], edgecolor="white",
                   linewidth=1.5, label=m)
        ax.set_xticks(np.arange(len(scenes)), scenes)
        ax.set_ylabel("metres (lower is better)")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.suptitle("Motion prediction per test scene (leave-one-scene-out, 3.2 s observed → 4.8 s predicted)")
    fig.tight_layout()
    save(fig, "predictor_per_scene.png")


def speed_sweep():
    runs = [(0.5, "results/nav_metrics_vmax0.5_fov360.csv"), (0.8, "results/nav_metrics_fov360.csv"),
            (1.2, "results/nav_metrics_vmax1.2_fov360.csv")]
    runs = [(v, read_csv(p)) for v, p in runs]
    runs = [(v, r) for v, r in runs if r]
    if len(runs) < 2:
        return print("skip speed_sweep: missing --v-max runs")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    v = [r[0] for r in runs]
    for ax, key, scale, ylabel, title in (
            (axes[0], "success", 100, "success (%)", "Success"),
            (axes[1], "collision", 100, "episodes with contact (%)", "Collisions"),
            (axes[2], "time", 1, "seconds", "Time to goal (successful runs)")):
        for m in MAIN:
            y = [scale * overall(r, m, key) for _, r in runs]
            ax.plot(v, y, "-o", color=COLOURS[m], lw=2, ms=8, label=NAMES[m].replace("\n", " "))
        ax.set_xticks(v)
        ax.set_xlabel("robot top speed (m/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(min(v) - 0.1, max(v) + 0.1)
        if scale == 100:
            ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Ablation: robot top speed (360° sensor, 40 crossings × 5 scenes)")
    fig.tight_layout()
    save(fig, "ablation_speed.png")


def sigma_sweep():
    runs = [(0.5, "results/nav_metrics_sigma0.5_fov360.csv"), (1.0, "results/nav_metrics_fov360.csv"),
            (1.5, "results/nav_metrics_sigma1.5_fov360.csv")]
    runs = [(s, read_csv(p)) for s, p in runs]
    runs = [(s, r) for s, r in runs if r]
    if len(runs) < 2:
        return print("skip sigma_sweep: missing --sigma-scale runs")
    s = [r[0] for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, scale, ylabel, title in (
            (axes[0], "success", 100, "success (%)", "Success"),
            (axes[1], "collision", 100, "episodes with contact (%)", "Collisions"),
            (axes[2], "min_dist", 1, "metres", "Closest gap to a person")):
        y = [scale * overall(r, "learned", key) for _, r in runs]
        ax.plot(s, y, "-o", color=COLOURS["learned"], lw=2, ms=8)
        for xi, yi in zip(s, y):
            ax.annotate(f"{yi:.0f}" if scale == 100 else f"{yi:.2f}", (xi, yi), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9, color=INK2)
        ax.set_xticks(s, [f"{k:g}×\n({', '.join(f'{k * z:g}' for z in (1, 2, 3))} σ)" for k in s])
        ax.set_xlabel("zone size (inner, middle, outer edge)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100 if scale == 100 else 0.5)
        ax.set_title(title)
    fig.suptitle("Ablation: size of the learned zones (360° sensor, 40 crossings × 5 scenes)")
    fig.tight_layout()
    save(fig, "ablation_sigma.png")


def inputs_ablation():
    """Social input and optical flow on/off, webcam view."""
    web = read_csv("results/nav_metrics_webcam.csv")
    ns = read_csv("results/nav_metrics_nosocial_webcam.csv")
    nf = read_csv("results/nav_metrics_noflow_webcam.csv")
    if not (web and (ns or nf)):
        return print("skip inputs_ablation: missing runs")
    bars = [("cv", "flow on", web, "cv"), ("cv", "flow off", nf, "cv"),
            ("learned", "flow on", web, "learned"), ("learned", "flow off", nf, "learned"),
            ("learned_nosocial", "no social", ns, "learned_nosocial")]
    bars = [b for b in bars if b[2]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(bars))
    for ax, key, title in ((axes[0], "success", "Success"), (axes[1], "collision", "Collisions")):
        vals = [100 * overall(r, m, key) for m, _, r, _ in bars]
        b = ax.bar(x, vals, 0.6, color=[COLOURS[c] for *_, c in bars], edgecolor="white", linewidth=2,
                   hatch=None)
        for bar, (_, what, _, _) in zip(b, bars):
            if what == "flow off":
                bar.set_hatch("///")
                bar.set_edgecolor("white")
        label_bars(ax, b)
        ax.set_xticks(x, [f"{m.split('_')[0]}\n{what}" for m, what, _, _ in bars])
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_title(title)
    fig.suptitle("Ablation: optical-flow speed and social input (webcam view, 40 crossings × 5 scenes)")
    fig.tight_layout()
    save(fig, "ablation_inputs.png")


def latency():
    files = sorted(glob.glob("results/speed_*.md"))
    pt = [f for f in files if f.endswith("_pytorch.md")]
    ov = [f for f in files if not f.endswith("_pytorch.md")]
    if not (pt and ov):
        return print("skip latency: need results/speed_<host>.md and speed_<host>_pytorch.md")

    def parse(path):
        rows = []
        for line in open(path):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) == 2 and not parts[0].startswith(("stage", "---", "**")):
                rows.append((parts[0], float(parts[1])))
        return rows

    a, b = parse(pt[0]), parse(ov[0])
    stages = ["detection (YOLO11n)" if s.startswith("detection") else s for s, _ in a]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(stages))
    ax.barh(y - 0.2, [t for _, t in a], 0.38, color="#2a78d6", label=f"YOLO11n in PyTorch ({sum(t for _, t in a):.0f} ms total)")
    ax.barh(y + 0.2, [t for _, t in b], 0.38, color="#86b6ef", label=f"YOLO11n in OpenVINO ({sum(t for _, t in b):.0f} ms total)")
    for yi, (_, t1), (_, t2) in zip(y, a, b):
        ax.annotate(f"{t1:.1f}", (t1, yi - 0.2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.annotate(f"{t2:.1f}", (t2, yi + 0.2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds per frame (median of 50, 320×240)")
    ax.set_title("Time per pipeline stage on the laptop (Intel i5-10210U, CPU only)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "latency.png")


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fill="#eef4fc"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fill, ec="#2a78d6", lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=INK)
        return (x, y, w, h)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.4))

    cams = box(0.1, 1.6, 1.5, 1.0, "2 USB webcams\n640×480 MJPG")
    rect = box(2.0, 1.6, 1.4, 1.0, "rectify\n(calibration)")
    sgbm = box(3.9, 3.0, 1.7, 0.9, "SGBM\nstereo depth")
    yolo = box(3.9, 1.65, 1.7, 0.9, "YOLO11n\ndetection")
    dis = box(3.9, 0.3, 1.7, 0.9, "DIS\noptical flow")
    box(6.1, 1.65, 1.9, 0.9, "ground position\n+ sideways speed")
    box(8.4, 1.65, 1.3, 0.9, "Kalman\ntracker")
    box(10.2, 3.0, 2.2, 0.9, "social GRU\nfuture path + σ", fill="#fdf3dc")
    box(10.2, 1.65, 2.2, 0.9, "nested risk zones\ninner / middle / outer")
    box(10.2, 0.3, 2.2, 0.9, "sampling planner\n91 candidate paths")
    arrow((1.6, 2.1), (2.0, 2.1))
    for b, y_in in zip((sgbm, yolo, dis), (2.35, 2.1, 1.85)):
        arrow((3.4, 2.1), (3.9, b[1] + b[3] / 2))
        arrow((5.6, b[1] + b[3] / 2), (6.1, y_in))
    arrow((8.0, 2.1), (8.4, 2.1))
    arrow((9.7, 2.3), (10.2, 3.45))
    arrow((11.3, 3.0), (11.3, 2.55))
    arrow((9.7, 2.1), (10.2, 2.1))
    arrow((11.3, 1.65), (11.3, 1.2))
    ax.text(11.3, 0.05, "→ speed and turn rate", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_title("Pipeline: from two webcams to a path that stays out of the predicted zones", fontsize=12)
    save(fig, "pipeline.png")


def snapshots(scene_name="zara1", episode=None):
    files = data.scene_files("data/raw", scene_name, "test")
    if not files:
        return print("skip snapshots: no data/raw")
    scene = sim.Scene(files[0])
    eps = sim.make_episodes(scene, 40, seed=1)
    cam = sim.Camera(fov_deg=360.0)
    preds = {"cv": ConstantVelocity.load(f"models/predictor_{scene_name}.npz"),
             "learned": GRUPredictor.load(f"models/predictor_{scene_name}.npz")}
    # Pick a crossing where the learned zones get the robot through and at least one other method
    # does not, preferring the one where the robot passes closest to several people.
    if episode is None:
        best = (-1, 0)
        for i, ep in enumerate(eps[:20]):
            res, frames = sim.run_episode(scene, ep, "learned", preds["learned"], cam, seed=i, record=True)
            if not res["success"]:
                continue
            others = [sim.run_episode(scene, ep, m, preds.get(m), cam, seed=i)["success"] for m in MAIN[:3]]
            if all(others):
                continue
            near = max(sum(np.linalg.norm(np.array(p) - f["pose"][:2]) < 3.0 for _, p, _ in f["tracks"])
                       for f in frames)
            if near > best[0]:
                best = (near, i)
        episode = best[1]
    ep = eps[episode]
    runs = {m: sim.run_episode(scene, ep, m, preds.get(m), cam, seed=episode, record=True) for m in MAIN}
    frames_l = runs["learned"][1]
    k_snap = max(range(len(frames_l)), key=lambda k: sum(
        np.linalg.norm(np.array(p) - frames_l[k]["pose"][:2]) < 4.0 for _, p, _ in frames_l[k]["tracks"]))
    t_snap = frames_l[k_snap]["t"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, m in zip(axes, MAIN):
        result, frames = runs[m]
        k = min(range(len(frames)), key=lambda i: abs(frames[i]["t"] - t_snap))
        f = frames[k]
        z = f["zones"]
        centre = f["pose"][:2]
        ki = 0                                             # zones now: all three levels, filled
        for level in (2, 1, 0):
            for n in range(len(z.centers)):
                a, b = z.axes[n, ki, level]
                ax.add_patch(Ellipse(z.centers[n, ki], 2 * a, 2 * b, angle=np.degrees(z.theta[n]),
                                     fc=ZONE_FILL[level], ec="none", alpha=0.35))
        for t_show, ls in ((1.0, "-"), (2.0, ":")):        # no-go zone in 1 s and 2 s: outlines
            ki = int(np.argmin(np.abs(z.times - t_show)))
            for n in range(len(z.centers)):
                a, b = z.axes[n, ki, 0]
                ax.add_patch(Ellipse(z.centers[n, ki], 2 * a, 2 * b, angle=np.degrees(z.theta[n]),
                                     fc="none", ec=ZONE_FILL[0], lw=1.3, ls=ls))
        for p in f["peds"]:
            ax.add_patch(plt.Circle(p, sim.PED_RADIUS, fc="white", ec=INK, lw=1.2))
        trail = np.array([g["pose"][:2] for g in frames[:k + 1]])
        ax.plot(trail[:, 0], trail[:, 1], color=INK2, lw=1.5, ls="--", label="driven so far")
        ax.plot(f["paths"][f["best"]][:, 0], f["paths"][f["best"]][:, 1], color=COLOURS[m], lw=3,
                label="chosen plan (3 s)")
        ax.add_patch(plt.Circle(f["pose"][:2], config.ROBOT_RADIUS, fc=COLOURS[m], ec=INK, lw=1.2))
        ax.plot(*ep.goal, marker="*", ms=16, color=INK)
        ax.set_xlim(centre[0] - 4.5, centre[0] + 4.5)
        ax.set_ylim(centre[1] - 4.5, centre[1] + 4.5)
        ax.set_aspect("equal")
        outcome = "reached goal" if result["success"] else ("contact" if result["collision"] else "timed out")
        ax.set_title(f"{NAMES[m].replace(chr(10), ' ')}  (t = {f['t']:.1f} s, {outcome})", fontsize=11)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)
    fig.suptitle(f"Same crossing in the {scene_name} crowd, four zone methods. Filled: zones now (red = no-go, "
                 "orange / yellow = costly). Red outlines: no-go zone in 1 s (solid) and 2 s (dotted). "
                 "White circles: people.", fontsize=11)
    fig.tight_layout()
    save(fig, "snapshots.png")


def live_methods(recording=None):
    """One moment of a real recording, replayed with each zone method (needs results/live/ from
    04_live_zones.py --video <recording> --method <m> --out results/live/<name>_<m>.mp4 --log ...csv)."""
    import cv2
    logs = sorted(glob.glob("results/live/*_learned.csv"))
    if not logs:
        return print("skip live_methods: no results/live/*_learned.csv")
    best = None                                 # (speed, recording, frame): fastest person 1.5-5 m away
    for path in logs:
        name = os.path.basename(path)[:-len("_learned.csv")]
        if recording and name != recording:
            continue
        for r in read_csv(path):
            if r["label"] == "person" and 1.5 < float(r["x"]) < 5.0:
                sp = float(np.hypot(float(r["vx"]), float(r["vy"])))
                if best is None or sp > best[0]:
                    best = (sp, name, int(r["frame"]) - 1)
    if best is None:
        return print("skip live_methods: no moving person in the logs")
    _, name, frame = best
    panels = {}
    for m in MAIN:
        cap = cv2.VideoCapture(f"results/live/{name}_{m}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            return print(f"skip live_methods: cannot read results/live/{name}_{m}.mp4")
        panels[m] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = panels["learned"].shape[:2]
    cam = panels["learned"][:, :w - h]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.9), gridspec_kw=dict(width_ratios=[w - h, h, h, h, h]))
    axes[0].imshow(cam)
    axes[0].set_title("left camera (detections, distance)", fontsize=11)
    for ax, m in zip(axes[1:], MAIN):
        ax.imshow(panels[m][:, w - h:])
        ax.set_title(f"zones: {NAMES[m].replace(chr(10), ' ')}", fontsize=11, color=INK)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_edgecolor(COLOURS[m])
            side.set_linewidth(3)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Real recording ({name}), the same frame replayed with each zone method. Top views: "
                 "rig at the bottom, 1 m grid, goal (star) 4 m ahead, green = chosen path.", fontsize=11)
    fig.tight_layout()
    save(fig, "live_methods.png")


def live_video(out="results/live_comparison.mp4"):
    """All recordings one after another: camera + the four methods' top views in one frame."""
    import cv2
    names = sorted({os.path.basename(p)[:-len("_learned.mp4")] for p in glob.glob("results/live/*_learned.mp4")})
    if not names:
        return print("skip live_video: no results/live/*.mp4")
    writer = None
    for name in names:
        caps = {m: cv2.VideoCapture(f"results/live/{name}_{m}.mp4") for m in MAIN}
        while True:
            frames = {m: c.read()[1] for m, c in caps.items()}
            if any(f is None for f in frames.values()):
                break
            h, w = frames["learned"].shape[:2]
            cam = frames["learned"][:, :w - h]
            tops = {m: f[:, w - h:].copy() for m, f in frames.items()}
            for m, img in tops.items():
                hexc = COLOURS[m].lstrip("#")
                bgr = tuple(int(hexc[i:i + 2], 16) for i in (4, 2, 0))
                cv2.rectangle(img, (0, 0), (h - 1, h - 1), bgr, 6)
            info = np.full_like(cam, 255)
            cv2.putText(info, f"recording {name}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
            cv2.putText(info, "top views: static | reach (top row)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)
            cv2.putText(info, "           cv | learned (bottom row)", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)
            cv2.putText(info, "red = no-go zone, green = chosen path", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)
            grid = np.vstack([np.hstack([cam, tops["static"], tops["reach"]]),
                              np.hstack([info, tops["cv"], tops["learned"]])])
            if writer is None:
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (grid.shape[1], grid.shape[0]))
            writer.write(grid)
    if writer is not None:
        writer.release()
        print("wrote", out)


def live_screenshots():
    """The screenshots taken with P in 04_live_zones.py, as small JPEGs for the report."""
    import cv2
    shots = sorted(glob.glob("screenshots/*.png"))
    for path in shots:
        img = cv2.imread(path)
        dest = os.path.join(OUT, "live_" + os.path.basename(path).replace(".png", ".jpg"))
        cv2.imwrite(dest, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", dest)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline_diagram()
    predictor_per_scene()
    nav_outcomes()
    speed_sweep()
    sigma_sweep()
    inputs_ablation()
    latency()
    snapshots()
    live_methods()
    live_screenshots()
    live_video()
