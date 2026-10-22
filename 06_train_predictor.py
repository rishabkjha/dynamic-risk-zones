"""Step 6: train the pedestrian motion predictor (a small GRU) on ETH/UCY and compare it with
constant velocity.

    python 05_get_data.py                 # once: downloads ETH/UCY to data/raw (14 MB)
    python 06_train_predictor.py          # about 5-10 minutes on a laptop CPU

What it does:
1. Leave-one-scene-out (the standard protocol): for each of the 5 scenes, train on the other 4
   and test on it. Prints ADE/FDE (average/final distance error, metres), NLL, and how often the
   true position falls inside the 2-sigma zone, for constant velocity and for the GRU.
   Saves models/predictor_<scene>.npz (used by 07_evaluate.py, so every test scene is unseen)
   and results/predictor_metrics.csv / .md.
2. Trains the final model on all 5 scenes -> models/predictor.npz (used by the live demo).

The network: past 8 positions (3.2 s) -> GRU(64), plus the 6 nearest people -> small MLP ->
max-pool ("social" part) -> MLP -> for each of the next 12 steps (4.8 s) a correction to
constant velocity (dx, dy) and an uncertainty growth (along / across the heading). Trained
with the Gaussian negative log-likelihood, so it learns WHERE people go and HOW SURE it can be.
The same network without the social part is trained too, to show what the neighbours add.
"""
import argparse
import os
import time

import numpy as np
import torch
from torch import nn

import config
from dynnav import data
from dynnav.predictor import SCALE, ConstantVelocity, GRUPredictor, canonical, social_input

OBS, PRED = config.PRED_OBS, config.PRED_STEPS


class Net(nn.Module):
    def __init__(self, hidden=64, social=True):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.use_social = social
        if social:
            self.social = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + (32 if social else 0), 128), nn.ReLU(),
                                  nn.Linear(128, 4 * PRED))

    def forward(self, x, nb):
        _, h = self.gru(x)
        h = h[-1]
        if self.use_social:
            f = (self.social(nb) * nb[..., 4:5]).max(1).values               # pool the neighbours
            h = torch.cat([h, f], 1)
        out = self.head(h).view(-1, PRED, 4)
        # The network corrects constant velocity: it starts from "keep walking" and learns turns,
        # slowing down and moving aside.
        mean = torch.cumsum(x[:, -1:, :] + out[..., :2], 1)                   # scaled offsets
        sigma = torch.cumsum(nn.functional.softplus(out[..., 2:]), 1) + 0.01  # scaled std
        return mean, sigma


def prepare(samples, neighbours):
    """World samples -> network input, social input, target offsets (turned, scaled frame)."""
    past, fut = samples[:, :OBS].astype(np.float64), samples[:, OBS:].astype(np.float64)
    d, theta = canonical(past)
    c, s = np.cos(-theta), np.sin(-theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    target = np.einsum("bij,btj->bti", rot, fut - past[:, -1:, :])
    nb = social_input(neighbours, theta, config.PRED_DT)
    return (torch.tensor(d / SCALE, dtype=torch.float32), torch.tensor(nb, dtype=torch.float32),
            torch.tensor(target / SCALE, dtype=torch.float32))


def nll(mean, sigma, target):
    return (torch.log(sigma) + 0.5 * ((target - mean) / sigma) ** 2).sum(-1).mean()


def train(train_s, val_s, epochs, social=True, seed=0, log=""):
    """train_s, val_s: (samples, neighbours) pairs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, nb, y = prepare(*train_s)
    vx, vnb, vy = prepare(*val_s)
    net = Net(social=social)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    for ep in range(epochs):
        net.train()
        perm = torch.tensor(rng.permutation(len(x)))
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            xb, nbb, yb = x[idx].clone(), nb[idx].clone(), y[idx].clone()
            flip = torch.tensor(rng.random(len(idx)) < 0.5)             # mirror left/right
            xb[flip, :, 1] *= -1
            yb[flip, :, 1] *= -1
            nbb[flip, :, 1] *= -1
            nbb[flip, :, 3] *= -1
            xb += torch.randn_like(xb) * (0.03 / SCALE)                 # tracker noise
            mean, sigma = net(xb, nbb)
            loss = nll(mean, sigma, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float(nll(*net(vx, vnb), vy))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        print(f"  {log} epoch {ep + 1:2d}/{epochs}  val NLL {v:.3f}", flush=True)
    net.load_state_dict(best_state)
    return net


def cv_sigma(samples):
    """Measured spread of constant velocity's error, per future step, along / across heading."""
    cv = ConstantVelocity(np.ones(PRED), np.ones(PRED), config.PRED_DT)
    pred = cv(samples[:, :OBS])
    err = samples[:, OBS:] - pred.mean
    c, s = np.cos(-pred.theta), np.sin(-pred.theta)
    along = c[:, None] * err[..., 0] - s[:, None] * err[..., 1]
    across = s[:, None] * err[..., 0] + c[:, None] * err[..., 1]
    return np.stack([np.sqrt(np.mean(along ** 2, 0)), np.sqrt(np.mean(across ** 2, 0))], -1)


def save(net, path, cv_sig):
    weights = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    np.savez(path, dt=config.PRED_DT, cv_sigma=cv_sig, obs=OBS, pred=PRED,
             neighbours=data.NEIGHBOURS, **weights)


def metrics(pred, fut):
    err = fut - pred.mean
    dist = np.linalg.norm(err, axis=-1)
    c, s = np.cos(-pred.theta), np.sin(-pred.theta)
    along = (c[:, None] * err[..., 0] - s[:, None] * err[..., 1]) / pred.sigma[..., 0]
    across = (s[:, None] * err[..., 0] + c[:, None] * err[..., 1]) / pred.sigma[..., 1]
    m2 = along ** 2 + across ** 2
    nll_v = np.mean(np.log(pred.sigma).sum(-1) + 0.5 * m2 + np.log(2 * np.pi))
    return dict(ADE=dist.mean(), FDE=dist[:, -1].mean(), NLL=nll_v,
                inside_2sigma=float(np.mean(m2[:, -1] < 4.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip-loo", action="store_true", help="only train the final model")
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    L = OBS + PRED
    rows = []
    if not args.skip_loo:
        for scene in data.SCENES:
            t0 = time.time()
            tr = data.load_split(args.data, scene, "train", L, OBS)
            va = data.load_split(args.data, scene, "val", L, OBS)
            te, te_nb = data.load_split(args.data, scene, "test", L, OBS)
            print(f"[{scene}] train {len(tr[0])}  val {len(va[0])}  test {len(te)}")
            sig = cv_sigma(tr[0])
            plain = f"models/predictor_nosocial_{scene}.npz"
            save(train(tr, va, args.epochs, social=False, log=scene + " GRU"), plain, sig)
            path = f"models/predictor_{scene}.npz"
            save(train(tr, va, args.epochs, social=True, log=scene + " social GRU"), path, sig)
            for name, p in (("constant velocity", ConstantVelocity.load(path)),
                            ("GRU", GRUPredictor.load(plain)),
                            ("social GRU (ours)", GRUPredictor.load(path))):
                m = metrics(p(te[:, :OBS], te_nb), te[:, OBS:])
                rows.append(dict(scene=scene, method=name, **m))
                print(f"  {name:18s} ADE {m['ADE']:.3f}  FDE {m['FDE']:.3f}  NLL {m['NLL']:.2f}  "
                      f"inside 2-sigma {100 * m['inside_2sigma']:.0f}%")
            print(f"  ({time.time() - t0:.0f} s)")
        write_tables(rows)

    print("[final] training on all 5 scenes")
    parts = [data.load_split(args.data, s, "test", L, OBS) for s in data.SCENES]
    full = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(full[0]))
    n_val = len(idx) // 10
    net = train((full[0][idx[n_val:]], full[1][idx[n_val:]]), (full[0][idx[:n_val]], full[1][idx[:n_val]]),
                args.epochs, log="final")
    save(net, config.PREDICTOR_FILE, cv_sigma(full[0]))
    print("saved", config.PREDICTOR_FILE)


def write_tables(rows):
    keys = ["ADE", "FDE", "NLL", "inside_2sigma"]
    with open("results/predictor_metrics.csv", "w") as f:
        f.write("scene,method," + ",".join(keys) + "\n")
        for r in rows:
            f.write(f"{r['scene']},{r['method']}," + ",".join(f"{r[k]:.4f}" for k in keys) + "\n")
    methods = list(dict.fromkeys(r["method"] for r in rows))
    with open("results/predictor_metrics.md", "w") as f:
        f.write("| scene | method | ADE (m) | FDE (m) | NLL | true position inside 2σ zone |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['scene']} | {r['method']} | {r['ADE']:.2f} | {r['FDE']:.2f} | "
                    f"{r['NLL']:.2f} | {100 * r['inside_2sigma']:.0f}% |\n")
        for m in methods:
            sub = [r for r in rows if r["method"] == m]
            f.write(f"| **average** | **{m}** | **{np.mean([r['ADE'] for r in sub]):.2f}** | "
                    f"**{np.mean([r['FDE'] for r in sub]):.2f}** | "
                    f"**{np.mean([r['NLL'] for r in sub]):.2f}** | "
                    f"**{100 * np.mean([r['inside_2sigma'] for r in sub]):.0f}%** |\n")
    print("wrote results/predictor_metrics.md")


if __name__ == "__main__":
    main()
