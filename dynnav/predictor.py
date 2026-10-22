"""Motion prediction: where will each tracked object be over the next few seconds, and how sure are we?

Every predictor returns, for each object and each future step, a mean position and two standard
deviations: along the object's heading and across it. A zone is then an ellipse around the mean,
stretched along the heading, that grows as the prediction gets less certain.

- ConstantVelocity: keeps walking straight at the current speed. Its uncertainty per step is
  measured on the training data, so it is a fair, calibrated baseline.
- GRUPredictor: a small recurrent network trained on real pedestrian trajectories (ETH/UCY,
  06_train_predictor.py), optionally also looking at the people around each person. It runs
  here in plain numpy, so the robot needs no PyTorch and it runs on anything, a Pi 4 included
  (about 0.1 ms per object).
"""
import numpy as np

SCALE = 0.5   # metres per step that inputs/outputs are divided by (a walk is ~0.5 m per 0.4 s)


def canonical(past):
    """Moves each past track to the origin and turns it to face +x.

    past: (B, n, 2) -> (displacements (B, n-1, 2) in the turned frame, heading angle (B,)).
    Facing +x makes the network's job independent of the direction people walk in.
    """
    d = np.diff(past, axis=1)
    last = d[:, -1]
    speed = np.linalg.norm(last, axis=1)
    theta = np.where(speed > 1e-3, np.arctan2(last[:, 1], last[:, 0]), 0.0)
    c, s = np.cos(-theta), np.sin(-theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)       # (B, 2, 2)
    return np.einsum("bij,btj->bti", rot, d), theta


def to_world(offsets, theta, origin):
    """Offsets (B, P, 2) in the turned frame -> world positions."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return origin[:, None, :] + np.einsum("bij,btj->bti", rot, offsets)


class Prediction:
    def __init__(self, mean, sigma, theta, dt):
        self.mean = mean        # (B, P, 2) world positions at dt, 2dt, ...
        self.sigma = sigma      # (B, P, 2) std along / across the heading
        self.theta = theta      # (B,) heading
        self.dt = dt


class ConstantVelocity:
    def __init__(self, sigma_along, sigma_across, dt):
        self.sigma = np.stack([sigma_along, sigma_across], -1)   # (P, 2)
        self.dt = dt

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["cv_sigma"][:, 0], z["cv_sigma"][:, 1], float(z["dt"]))

    def __call__(self, past, neighbours=None):
        past = np.asarray(past, np.float64)
        d, theta = canonical(past)
        P = len(self.sigma)
        v = d[:, -1]                                             # last step, turned frame
        offsets = v[:, None, :] * np.arange(1, P + 1)[None, :, None]
        mean = to_world(offsets, theta, past[:, -1])
        sigma = np.broadcast_to(self.sigma, (len(past), P, 2)).copy()
        return Prediction(mean, sigma, theta, self.dt)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def turn(vectors, theta):
    """Rotates (B, k, 2) vectors by -theta: world frame -> each object's turned frame."""
    c, s = np.cos(-theta)[:, None], np.sin(-theta)[:, None]
    return np.stack([c * vectors[..., 0] - s * vectors[..., 1],
                     s * vectors[..., 0] + c * vectors[..., 1]], -1)


def social_input(neighbours, theta, dt):
    """Neighbours (B, k, 5) world [dx, dy, vx, vy, present] -> network input in the turned frame."""
    nb = np.asarray(neighbours, np.float64)
    rel = turn(nb[..., :2], theta) / 2.0                     # metres / 2
    vel = turn(nb[..., 2:4], theta) * dt / SCALE             # steps, like the track input
    return np.concatenate([rel, vel, nb[..., 4:5]], -1) * nb[..., 4:5]


class GRUPredictor:
    """Numpy copy of the network in 06_train_predictor.py (weights in models/predictor.npz).

    The GRU reads the object's own past steps. The social part (when the file has it) reads the
    nearest other people and max-pools them, so the network can learn that people walking
    towards each other move aside. The output is a correction to constant velocity.
    """

    def __init__(self, weights):
        self.w = {k: weights[k].astype(np.float64) for k in weights.files
                  if "." in k}
        self.dt = float(weights["dt"])
        self.H = self.w["gru.weight_hh_l0"].shape[1]
        self.P = self.w["head.2.weight"].shape[0] // 4
        self.social = "social.0.weight" in self.w
        self.k = int(weights["neighbours"]) if "neighbours" in weights.files else 0

    @classmethod
    def load(cls, path):
        return cls(np.load(path))

    def forward(self, x, nb=None):
        """x: (B, T, 2) scaled displacements, nb: (B, k, 5) social input -> head output (B, 4P)."""
        w, H = self.w, self.H
        Wi, Wh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"]
        bi, bh = w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
        h = np.zeros((x.shape[0], H))
        for t in range(x.shape[1]):
            gi = x[:, t] @ Wi.T + bi
            gh = h @ Wh.T + bh
            r = _sigmoid(gi[:, :H] + gh[:, :H])
            z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1 - z) * n + z * h
        if self.social:
            f = np.maximum(nb @ w["social.0.weight"].T + w["social.0.bias"], 0.0)
            f = np.maximum(f @ w["social.2.weight"].T + w["social.2.bias"], 0.0)
            f = (f * nb[..., 4:5]).max(axis=1)
            h = np.concatenate([h, f], 1)
        y = np.maximum(h @ w["head.0.weight"].T + w["head.0.bias"], 0.0)
        return y @ w["head.2.weight"].T + w["head.2.bias"]

    def __call__(self, past, neighbours=None):
        past = np.asarray(past, np.float64)
        d, theta = canonical(past)
        x = d / SCALE
        nb = None
        if self.social:
            if neighbours is None:
                neighbours = np.zeros((len(past), self.k, 5))
            nb = social_input(neighbours, theta, self.dt)
        out = self.forward(x, nb).reshape(len(past), self.P, 4)
        offsets = np.cumsum(x[:, -1:, :] + out[..., :2], axis=1) * SCALE
        sigma = (np.cumsum(_softplus(out[..., 2:]), axis=1) + 0.01) * SCALE
        return Prediction(to_world(offsets, theta, past[:, -1]), sigma, theta, self.dt)


def load_predictor(kind, path):
    if kind == "learned":
        return GRUPredictor.load(path)
    if kind == "cv":
        return ConstantVelocity.load(path)
    raise ValueError(kind)
