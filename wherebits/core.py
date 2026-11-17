"""The WhereBits estimator (paper, Sec. 3) on numpy arrays.

Inputs are the attention rows of the answer-start query (the last prompt token) to the N visual tokens,
for every layer l and head h, under up to four input conditions:

    "xt"  image x,     question t            "xg"  image x,     generic prompt g
    "0t"  gray image,  question t            "0g"  gray image,  generic prompt g

For condition c, a^c_lh(n) is the attention to visual token n, m^c_lh = sum_n a^c_lh(n) the visual mass and
P^c_lh = a^c_lh / m^c_lh the visual routing distribution. All logarithms are base 2.

    d_lh(n) = log2 P^xt_lh(n) - log2 P^xg_lh(n)                        per-head token score       (Eq. 1)
    B^z_lh  = m^zt_lh * KL(P^zt_lh || P^zg_lh),   z in {x, 0}           where-to-look bits         (Prop. 2)
    w_lh    = [B^x_lh - B^0_lh]_+                                       head weight                (Eq. 3)
    F_l(n)  = sum_h w_lh d_lh(n) / sum_h w_lh   (0 if every w_lh = 0)   layer map
    W_l     = sum_h w_lh                                                layer bits of one question
    pi_l    = Bbar_l^gamma / sum_j Bbar_j^gamma,  Bbar_l = mean of W_l over unlabeled questions
    M(n)    = sum_l pi_l F_l(n)                                         final map                  (Eq. 4)

Attention is passed as log2 values (``log2a``, floored at 2^-31.875) together with the exact visual mass
(``mass``). This is the format of the stored attention dumps (``wherebits.dump``), so the online and the
offline code paths compute the same quantities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONDITIONS = ("xt", "0t", "xg", "0g")     # order of the stored dumps
GAMMA = 4.0                              # prior exponent, fixed once for all models
LOG2_FLOOR = -255.0 / 8.0                # smallest stored log2 attention (2^-31.875)
EPS = 2.0 ** -40


def log2_attention(attn) -> np.ndarray:
    """Attention probabilities -> log2 values, floored at 2^-31.875 as in the stored dumps."""
    a = np.asarray(attn, dtype=np.float64)
    return np.log2(np.clip(a, 2.0 ** LOG2_FLOOR, None))


def log_routing(log2a: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """log2 P = log2 a - log2 m for every head: [..., N] from log2 attention [..., N] and visual mass [...]."""
    return np.asarray(log2a, np.float64) - np.log2(np.maximum(np.asarray(mass, np.float64), EPS))[..., None]


def token_scores(log2a_t, mass_t, log2a_g, mass_g) -> np.ndarray:
    """Per-head token scores d = log2 P^t - log2 P^g (Eq. 1): [L, H, N]."""
    return log_routing(log2a_t, mass_t) - log_routing(log2a_g, mass_g)


def where_bits(log2a_t, mass_t, log2a_g, mass_g) -> np.ndarray:
    """Where-to-look bits B = m^t KL(P^t || P^g) of every head (Prop. 2, second term): [L, H]."""
    lp_t = log_routing(log2a_t, mass_t)
    d = lp_t - log_routing(log2a_g, mass_g)
    return np.asarray(mass_t, np.float64) * (np.exp2(lp_t) * d).sum(-1)


def whether_bits(mass_t, mass_g) -> np.ndarray:
    """Whether-to-look bits KL_bin(m^t || m^g) of every head (Prop. 2, first term): [L, H]."""
    a = np.clip(np.asarray(mass_t, np.float64), EPS, 1 - 1e-9)
    b = np.clip(np.asarray(mass_g, np.float64), EPS, 1 - 1e-9)
    return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))


def head_weights(log2a: dict, mass: dict, *, weighting: str = "bits") -> np.ndarray:
    """Head weights w [L, H].

    ``weighting="bits"``: w = [B^x - B^0]_+ (the paper's method; needs "xt", "xg", "0t", "0g").
    ``weighting="mass"``: w = m^xt (two-pass variant; needs only "xt" and "xg").
    """
    if weighting == "mass":
        return np.asarray(mass["xt"], np.float64).copy()
    if weighting != "bits":
        raise ValueError(f"unknown head weighting {weighting!r}")
    b_x = where_bits(log2a["xt"], mass["xt"], log2a["xg"], mass["xg"])
    b_0 = where_bits(log2a["0t"], mass["0t"], log2a["0g"], mass["0g"])
    return np.maximum(b_x - b_0, 0.0)


def layer_maps(d: np.ndarray, w: np.ndarray) -> np.ndarray:
    """F_l = sum_h w_lh d_lh / sum_h w_lh, and 0 for a layer whose heads all have zero weight: [L, N]."""
    ws = np.maximum(w.sum(1, keepdims=True), EPS)
    return (w[..., None] * d).sum(1) / ws


def layer_prior(layer_bits_mean, gamma: float = GAMMA) -> np.ndarray:
    """Layer prior pi = Bbar^gamma / sum Bbar^gamma from the mean layer bits of unlabeled questions (Eq. 4)."""
    v = np.maximum(np.asarray(layer_bits_mean, np.float64), 0.0) ** gamma
    s = v.sum()
    return v / s if s > 0 else np.full(v.shape, 1.0 / v.size)


def per_question_layers(layer_bits, top_k: int = 3) -> np.ndarray:
    """Layer weights without a prior: the question's own top-k layers by layer bits, weighted by their bits."""
    v = np.maximum(np.asarray(layer_bits, np.float64), 0.0)
    thr = np.sort(v)[-top_k]
    v = np.where(v >= thr, v, 0.0)
    s = v.sum()
    return v / s if s > 0 else np.full(v.shape, 1.0 / v.size)


@dataclass
class WhereBitsResult:
    """Output for one question. Token axes are in visual-token order (row-major over the token grid)."""

    map: np.ndarray            # [N]    final map M
    layer_maps: np.ndarray     # [L, N] per-layer maps F_l
    head_weights: np.ndarray   # [L, H] head weights w_lh
    layer_bits: np.ndarray     # [L]    W_l = sum_h w_lh of this question (input of the prior)
    layer_weights: np.ndarray  # [L]    weights pi_l used for the map

    def as_grid(self, grid) -> np.ndarray:
        """The final map reshaped to the visual-token grid (rows, cols)."""
        return self.map.reshape(tuple(grid))


def wherebits(log2a: dict, mass: dict, prior=None, *, weighting: str = "bits") -> WhereBitsResult:
    """WhereBits map of one question.

    log2a: condition -> log2 attention [L, H, N]; mass: condition -> visual mass [L, H] (see CONDITIONS).
    prior: layer prior pi [L] (normalized weights, e.g. from ``priors/<model>.json``). ``None`` uses the
           question's own top-3 layers instead (the "per-question layers" variant).
    weighting: "bits" (default, four conditions) or "mass" (two-pass variant, "xt" and "xg" only; use a prior).
    """
    d = token_scores(log2a["xt"], mass["xt"], log2a["xg"], mass["xg"])
    w = head_weights(log2a, mass, weighting=weighting)
    F = layer_maps(d, w)
    bits = w.sum(1)
    if prior is None:
        pi = per_question_layers(bits)
    else:
        pi = np.asarray(prior, np.float64)
        if pi.shape != (F.shape[0],):
            raise ValueError(f"prior has shape {pi.shape}, the model has {F.shape[0]} layers")
    return WhereBitsResult(map=(pi[:, None] * F).sum(0), layer_maps=F, head_weights=w, layer_bits=bits,
                           layer_weights=pi)


def from_attention(attn: dict) -> tuple[dict, dict]:
    """Attention rows (condition -> [L, H, N] probabilities over the visual tokens) -> (log2a, mass)."""
    log2a, mass = {}, {}
    for c, a in attn.items():
        a = np.clip(np.asarray(a, np.float64), 0.0, None)
        log2a[c] = log2_attention(a)
        mass[c] = a.sum(-1)
    return log2a, mass
