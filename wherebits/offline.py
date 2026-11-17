"""Offline computation from attention dumps: WhereBits layer maps and the attention baselines.

``reduce_dump`` turns one dump into a compact record (float32 arrays, as used for every table of the paper):

    F       [L, N]  WhereBits layer maps F_l (head weights w = [B^x - B^0]_+)
    W       [L]     layer bits sum_h w_lh of this question (averaged over unlabeled questions -> prior)
    F_mass  [L, N]  layer maps with head weights w = m^xt (two-pass variant)
    ratio   [L, N]  ViCrop's relative attention per layer, A^xt / A^xg (head-mean attention)
    attn    [L, N]  head-mean attention under the question; attn_g under the generic prompt
    klq     [L, N]  KL-weighted per-layer map sum_h [P^xt d]_+ (routing renormalized per head)

With ``ablations=True`` the record also holds the layer maps and statistics of the ablations (paper, Table 5):

    F_where   [L, N]  head weights without the gray-image correction, w = B^x
    W_where   [L]     layer bits without the gray-image correction (prior variant)
    F_kl      [L, N]  head weights KL(P^xt || P^xg) without the visual mass (routing renormalized per head)
    F_uniform [L, N]  uniform head weights
    mix_dq    [L, N]  log-ratio of the head-mean attention (ViCrop-style head mixture)

``final_map`` combines the layer maps with a prior; ``baseline_maps`` gives the label-free baselines.
"""
from __future__ import annotations

import zlib

import numpy as np

from .core import EPS, head_weights, layer_maps, per_question_layers, token_scores, where_bits
from .dump import conditions, load_dump


def reduce_dump(path, ablations: bool = False) -> dict:
    d = load_dump(path)
    meta = d["meta"]
    log2a, mass = conditions(d)
    gh, gw = meta["grid"]
    L, H, N = log2a["xt"].shape
    dq = token_scores(log2a["xt"], mass["xt"], log2a["xg"], mass["xg"])
    w = head_weights(log2a, mass)
    F = layer_maps(dq, w)
    F_mass = layer_maps(dq, mass["xt"])
    A = np.exp2(log2a["xt"]).mean(1)                 # [L,N] head-mean attention (floored values)
    Ag = np.exp2(log2a["xg"]).mean(1)
    lp = {c: log2a[c] - np.log2(np.maximum(np.exp2(log2a[c]).sum(-1, keepdims=True), EPS)) for c in ("xt", "xg")}
    klq = np.maximum(np.exp2(lp["xt"]) * (lp["xt"] - lp["xg"]), 0).sum(1)
    rec = {"meta": meta, "grid": (gh, gw), "L": L, "H": H, "N": N,
           "F": F.astype(np.float32), "W": w.sum(1).astype(np.float32), "F_mass": F_mass.astype(np.float32),
           "ratio": (A / np.maximum(Ag, EPS)).astype(np.float32), "attn": A.astype(np.float32),
           "attn_g": Ag.astype(np.float32), "klq": klq.astype(np.float32),
           "base": {k: np.asarray(v, np.float32) for k, v in d.get("base", {}).items()}}
    if ablations:
        w_where = where_bits(log2a["xt"], mass["xt"], log2a["xg"], mass["xg"])
        lp_x0 = {c: log2a[c] - np.log2(np.maximum(np.exp2(log2a[c]).sum(-1, keepdims=True), EPS)) for c in ("xt", "xg")}
        dq_n = lp_x0["xt"] - lp_x0["xg"]
        kl_n = (np.exp2(lp_x0["xt"]) * dq_n).sum(-1)
        Pb = {c: a / np.maximum(a.sum(-1, keepdims=True), EPS) for c, a in (("xt", A), ("xg", Ag))}
        rec.update({"F_where": layer_maps(dq, w_where).astype(np.float32), "W_where": w_where.sum(1).astype(np.float32),
                    "F_kl": layer_maps(dq_n, kl_n).astype(np.float32), "F_uniform": dq.mean(1).astype(np.float32),
                    "mix_dq": (np.log2(np.maximum(Pb["xt"], EPS)) - np.log2(np.maximum(Pb["xg"], EPS))).astype(np.float32)})
    if "gt" in d:
        rec["gt"] = d["gt"].astype(bool).ravel()
    return rec


def mean_layer_bits(records) -> np.ndarray:
    """Mean layer bits Bbar over a set of (unlabeled) questions."""
    return np.mean([r["W"] for r in records], 0)


def final_map(rec: dict, prior=None, *, layers: str = "F") -> np.ndarray:
    """M = sum_l pi_l F_l with a prior pi (or the question's own top-3 layers when ``prior`` is None).

    ``layers="F"``: WhereBits; ``layers="F_mass"``: two-pass variant (head weights from the visual mass).
    """
    pi = per_question_layers(rec["W"]) if prior is None else np.asarray(prior, np.float64)
    return (pi[:, None] * rec[layers].astype(np.float64)).sum(0)


def band(L: int, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of the layers between relative depths lo and hi (inclusive, rounded)."""
    a, b = int(round(lo * (L - 1))), int(round(hi * (L - 1)))
    m = np.zeros(L, bool)
    m[a:max(a + 1, b + 1)] = True
    return m


def baseline_maps(rec: dict) -> dict:
    """Label-free baseline maps [N] from a reduced record.

    attn_mean   mean attention over layers and heads (question prompt)
    vicrop      ViCrop's relative attention A^xt / A^xg averaged over all layers
    vicrop_band the same over layers at 50-90% depth
    carve       CARVE re-implementation: max-normalized attention ratio A / (A_g + 0.05), layers at 70-90% depth
    kl_weighted KL-weighted contrast sum_l sum_h [P^xt d]_+
    random      uniform random scores (seeded by the question id)
    plus the vlmprobe baselines stored in the dump (relevance = Chefer relevance, grad_cam, rollout, tokenact)
    """
    L = rec["L"]
    A, Ag = rec["attn"].astype(np.float64), rec["attn_g"].astype(np.float64)
    ratio = rec["ratio"].astype(np.float64)
    out = {"attn_mean": A.mean(0), "vicrop": ratio.mean(0), "vicrop_band": ratio[band(L, 0.5, 0.9)].mean(0)}
    An = A / np.maximum(A.max(-1, keepdims=True), EPS)
    Agn = Ag / np.maximum(Ag.max(-1, keepdims=True), EPS)
    out["carve"] = (An / (Agn + 0.05))[band(L, 0.70, 0.90)].mean(0)
    out["kl_weighted"] = rec["klq"].astype(np.float64).sum(0)
    for name, v in rec.get("base", {}).items():
        out[name] = v.astype(np.float64)
    seed = zlib.crc32(str(rec["meta"]["qid"]).encode()) % (2 ** 31)
    out["random"] = np.random.default_rng(seed).random(rec["N"])
    return out


def tuned_layer(records, mask=None) -> int:
    """ViCrop's layer tuned with localization labels: the layer whose relative-attention map has the most pointing
    hits on the labeled questions selected by ``mask`` (default: all)."""
    L = records[0]["L"]
    hits = np.zeros(L)
    for i, r in enumerate(records):
        if (mask is None or mask[i]) and "gt" in r:
            for l in range(L):
                hits[l] += r["gt"][int(np.argmax(r["ratio"][l]))]
    return int(np.argmax(hits))



def reduce_folder(folder, workers: int = 1, require_gt: bool = False, ablations: bool = False) -> list[dict]:
    """``reduce_dump`` for every finished dump of a task/model folder (sorted by file name)."""
    from functools import partial

    from .dump import list_dumps
    files = list_dumps(folder)
    fn = partial(reduce_dump, ablations=ablations)
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            recs = pool.map(fn, files, chunksize=8)
    else:
        recs = [fn(f) for f in files]
    return [r for r in recs if "gt" in r] if require_gt else recs
