"""Crop rules for training-free zoom-in (paper, Sec. 3.5).

Windows are token-aligned: (i, j, h, w) = top row, left column, height and width in visual tokens.

* ``best_window``  - primary rule: the fixed-size window with the largest sum of positive map scores.
* ``bits_window``  - secondary rule: the window that captures the most where-to-look bits at the coarse level
  "inside W or not". For a head, with mu = P^xt(W) and nu = P^xg(W), the score is KL_bin(mu || nu) if mu > nu and
  0 otherwise; scores are averaged with the head weights and the layer prior of the map.
"""
from __future__ import annotations

import random

import numpy as np

BW_EPS = 1e-12                     # numerical floor of the bits-window scores


def window_size(grid, frac: float):
    """Window side in tokens: about ``frac`` of each grid side (at least one token)."""
    gh, gw = grid
    return max(1, round(frac * gh)), max(1, round(frac * gw))


def _window_sums(x: np.ndarray, wh: int, ww: int) -> np.ndarray:
    """Sums of every wh x ww window over the last two axes (summed-area table)."""
    gh, gw = x.shape[-2:]
    cs = np.zeros(x.shape[:-2] + (gh + 1, gw + 1), dtype=np.float64)
    cs[..., 1:, 1:] = x.cumsum(-2).cumsum(-1)
    return cs[..., wh:, ww:] - cs[..., :-wh, ww:] - cs[..., wh:, :-ww] + cs[..., :-wh, :-ww]


def center_window(grid, frac: float = 0.5):
    gh, gw = grid
    wh, ww = window_size(grid, frac)
    return (gh - wh) // 2, (gw - ww) // 2, wh, ww


def random_window(grid, frac: float = 0.5, seed: int = 0):
    gh, gw = grid
    wh, ww = window_size(grid, frac)
    rng = random.Random(seed)
    return rng.randint(0, gh - wh), rng.randint(0, gw - ww), wh, ww


def best_window(map2d, frac: float = 0.5):
    """Primary crop rule: the window of side ``frac`` x grid with the largest positive map mass.

    ``map2d`` is a [rows, cols] array (numpy or torch). An empty map (no positive score) gives the center window.
    """
    m = np.asarray(map2d.detach().cpu().numpy() if hasattr(map2d, "detach") else map2d, dtype=np.float64)
    grid = m.shape
    wh, ww = window_size(grid, frac)
    m = np.clip(np.nan_to_num(m, nan=0.0), 0.0, None)
    if float(m.sum()) <= 0:
        return center_window(grid, frac)
    sums = _window_sums(m, wh, ww)
    i, j = divmod(int(np.argmax(sums)), sums.shape[1])
    return int(i), int(j), wh, ww


def bits_window_scores(log2a: dict, mass: dict, prior, grid, frac: float):
    """Score of every window under the bits-window rule: [n_rows, n_cols] and the window size (wh, ww).

    Routing distributions are renormalized per head over the stored (floored) attention values, and the head
    weights are the where-to-look bits of the map, w = [B^x - B^0]_+.
    """
    lp = {}
    for c in ("xt", "0t", "xg", "0g"):
        la = np.asarray(log2a[c], np.float64)
        lp[c] = la - np.log2(np.maximum(np.exp2(la).sum(-1, keepdims=True), BW_EPS))
    P = {c: np.exp2(v) for c, v in lp.items()}
    klq = (P["xt"] * (lp["xt"] - lp["xg"])).sum(-1)
    klq0 = (P["0t"] * (lp["0t"] - lp["0g"])).sum(-1)
    w = np.maximum(np.asarray(mass["xt"], np.float64) * klq - np.asarray(mass["0t"], np.float64) * klq0, 0.0)
    gh, gw = grid
    wh, ww = window_size(grid, frac)
    L, H = w.shape
    mq = np.clip(_window_sums(P["xt"].reshape(L, H, gh, gw), wh, ww), BW_EPS, 1 - 1e-9)
    mg = np.clip(_window_sums(P["xg"].reshape(L, H, gh, gw), wh, ww), BW_EPS, 1 - 1e-9)
    b = mq * np.log2(mq / mg) + (1 - mq) * np.log2((1 - mq) / (1 - mg))
    b = np.where(mq > mg, b, 0.0)
    ws = np.maximum(w.sum(1), BW_EPS)
    per_layer = (w[..., None, None] * b).sum(1) / ws[:, None, None]
    return (np.asarray(prior, np.float64)[:, None, None] * per_layer).sum(0), (wh, ww)


def bits_window(log2a: dict, mass: dict, prior, grid, frac: float = 0.5):
    """Secondary crop rule (bits window): returns (i, j, h, w) in tokens."""
    scores, (wh, ww) = bits_window_scores(log2a, mass, prior, grid, frac)
    i, j = divmod(int(np.argmax(scores)), scores.shape[1])
    return int(i), int(j), wh, ww


def window_indicator(grid, window) -> np.ndarray:
    """0/1 map of a window; ``best_window`` of this map returns the same window (for maps bundles)."""
    i, j, wh, ww = window
    m = np.zeros(tuple(grid))
    m[i:i + wh, j:j + ww] = 1.0
    return m


def crop_original(original, grid, window):
    """Map a token window of the model frame (a resize of ``original``) back to original pixels and crop."""
    gh, gw = grid
    i, j, wh, ww = window
    W, H = original.size
    box = (int(j * W / gw), int(i * H / gh), int(round((j + ww) * W / gw)), int(round((i + wh) * H / gh)))
    return original.crop(box), box
