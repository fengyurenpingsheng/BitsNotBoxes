"""Compact per-question attention dumps.

For every question we keep the answer-start attention row of every layer and head over the N visual tokens under
the four input conditions (order: xt, 0t, xg, 0g). WhereBits, its ablations and the attention baselines can then be
computed offline without running the model again (``wherebits.offline``).

Arrays of one dump (``<qid>.npz``):
    la    uint8   [4, L, H, N]  q = round(-8 * log2 a), attention probabilities a (floored at 2^-31.875)
    vm    float32 [4, L, H]     exact visual attention mass sum_n a(n)
    gd    float32 [3, L, H]     gradient . attention per head (optional): sum g (a_xt - a_0t), sum g (a_xt - a_xg), sum g a_xt
    gxa   float16 [L, N]        sum_h g * a_xt per layer (optional), scaled by gxa_scale
    hn    float16 [L', N]       hidden-state L2 norm of every visual token per residual site (optional)
    base  float16 [M, N]        baseline maps computed with vlmprobe (names in meta["base_names"])
    gt    bool    [gh, gw]      token-level ground truth (localization tasks)
    meta  json                  question id, question, answer, grid, frame, model, ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

STEPS = 8.0                     # quantization steps per bit
MAX_BITS = 255.0 / STEPS        # 31.875 bits: smallest stored attention 2^-31.875 ~ 4e-10
CONDITIONS = ("xt", "0t", "xg", "0g")


def quantize_log2(a) -> np.ndarray:
    """Attention probabilities (>= 0) -> uint8 codes of -log2(a) in 1/8-bit steps."""
    a = np.asarray(a, dtype=np.float64)
    bits = -np.log2(np.clip(a, 2.0 ** -MAX_BITS, 1.0))
    return np.clip(np.rint(bits * STEPS), 0, 255).astype(np.uint8)


def dequantize_log2(q: np.ndarray) -> np.ndarray:
    """uint8 codes -> log2(a) as float32 (<= 0)."""
    return -(q.astype(np.float32) / STEPS)


def pack_attention(vis: dict) -> dict:
    """Condition -> visual attention rows [L, H, N] (numpy, >= 0) -> {"la": [4,L,H,N] uint8, "vm": [4,L,H]}."""
    rows = [np.clip(np.asarray(vis[c], np.float64), 0.0, None) for c in CONDITIONS]
    n = {r.shape for r in rows}
    if len(n) != 1:
        raise ValueError(f"conditions have different shapes: {sorted(n)}")
    return {"la": np.stack([quantize_log2(r) for r in rows]),
            "vm": np.stack([r.sum(-1) for r in rows]).astype(np.float32)}


def pack_rows(r_xt, r_0t, r_xg, r_0g) -> dict:
    """Captured rows (``wherebits.capture.Rows``) of the four conditions -> dump arrays (needs torch)."""
    import torch
    vis = {c: r.vis.clamp_min(0).double() for c, r in zip(CONDITIONS, (r_xt, r_0t, r_xg, r_0g))}
    out = pack_attention({c: v.numpy() for c, v in vis.items()})
    if r_xt.grad is not None:
        g = r_xt.grad[..., r_xt.visual].double()                                 # [L,H,N]
        a_xt, a_0t, a_xg = vis["xt"], vis["0t"], vis["xg"]
        gd = torch.stack([(g * (a_xt - a_0t)).sum(-1), (g * (a_xt - a_xg)).sum(-1), (g * a_xt).sum(-1)])
        out["gd"] = gd.float().numpy()
        gxa = (g * a_xt).sum(1)                                                   # [L,N]
        scale = float(gxa.abs().max()) or 1.0
        out["gxa"] = (gxa / scale).half().numpy()
        out["gxa_scale"] = np.float64(scale)
    return out


def save_dump(path, arrays: dict, meta: dict):
    """Atomic write: a job killed mid-write never leaves a truncated .npz behind."""
    path = Path(path)
    arrays = dict(arrays)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_dump(path) -> dict:
    """npz -> dict with 'log2a' [4,L,H,N] float32, 'vm' [4,L,H], 'meta' (dict) and the optional arrays."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(bytes(d["meta"]).decode())
    d["log2a"] = dequantize_log2(d.pop("la"))
    if "gxa" in d:
        d["gxa"] = d["gxa"].astype(np.float32) * float(d.pop("gxa_scale", 1.0))
    if "base" in d:
        d["base"] = {name: d["base"][i].astype(np.float32) for i, name in enumerate(d["meta"].get("base_names", []))}
    return d


def conditions(d: dict) -> tuple[dict, dict]:
    """A loaded dump -> (log2a, mass) dicts keyed by condition, in float64 (input of ``wherebits.core``)."""
    la = d["log2a"].astype(np.float64)
    vm = d["vm"].astype(np.float64)
    return {c: la[i] for i, c in enumerate(CONDITIONS)}, {c: vm[i] for i, c in enumerate(CONDITIONS)}


def list_dumps(folder) -> list[str]:
    """Finished dumps of one task/model folder, sorted by file name."""
    return sorted(str(p) for p in Path(folder).glob("*.npz") if not p.name.endswith(".partial.npz"))
