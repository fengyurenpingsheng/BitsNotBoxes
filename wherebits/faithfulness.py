"""Perturbation faithfulness: deletion and insertion with two budget axes. Needs torch.

Image cells (one per visual token) are replaced by gray or blurred pixels in the order of a map, and we track the
probability of the model's original first answer token. Each curve is summarized by its area (trapezoid rule):

    x-axis "token": fraction of cells removed (deletion) or restored (insertion)
    x-axis "bit":   fraction of the image's coding cost carried by those cells (learned codec, see wherebits.codec)

Lower deletion AUC and higher insertion AUC are better; the paper reports insertion AUC minus deletion AUC.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def trap(xs, ys) -> float:
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    area = sum((x1 - x0) * (y0 + y1) / 2 for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]))
    span = xs[-1] - xs[0]
    return area / span if span > 0 else float("nan")


def rank_tokens(grid_map, *, descending: bool = True):
    """Token order of a map (most relevant first); ties are broken deterministically by position."""
    import torch
    flat = grid_map.flatten()
    noise = torch.arange(flat.numel(), dtype=flat.dtype) * 1e-12
    return torch.argsort(flat + noise, descending=descending)


def keep_mask_from_order(order, n: int, grid: tuple, *, insertion: bool = False):
    """Deletion: drop the first n tokens of ``order``. Insertion: keep only the first n tokens."""
    import torch
    keep = torch.ones(grid[0] * grid[1], dtype=torch.bool)
    if insertion:
        keep[:] = False
        keep[order[:n]] = True
    else:
        keep[order[:n]] = False
    return keep.reshape(grid)


def paint_tokens(image: Image.Image, keep_tokens, *, operator: str = "gray") -> Image.Image:
    """Replace the cells of the tokens that are not kept by 50% gray or by a strongly blurred copy of the image."""
    w, h = image.size
    gh, gw = keep_tokens.shape
    out = image.copy()
    if operator == "gray":
        fill = Image.new("RGB", (w, h), (128, 128, 128))
    elif operator == "blur":
        fill = image.filter(ImageFilter.GaussianBlur(radius=max(1, min(w, h) // 16)))
    else:
        raise ValueError(operator)
    m = np.kron((~keep_tokens).numpy().astype("uint8"), np.ones((h // gh, w // gw), dtype="uint8")) * 255
    out.paste(fill, (0, 0), Image.fromarray(m[:h, :w], mode="L"))
    return out


class FaithEvaluator:
    def __init__(self, score_fn, *, fractions=DEFAULT_FRACTIONS, operators=("gray", "blur"), insertion=True):
        """``score_fn(pil_image) -> (log p(target), argmax token id, next-token log-distribution [V])``."""
        self.score_fn = score_fn
        self.fractions = sorted(set(float(f) for f in fractions) | {0.0, 1.0})
        self.operators = tuple(operators)
        self.insertion = insertion

    def evaluate(self, image, maps: dict, *, token_bits=None, methods=None) -> dict:
        names = [n for n in (list(methods) if methods is not None else list(maps)) if n in maps]
        grid = tuple(next(iter(maps.values())).shape)
        n = grid[0] * grid[1]
        base_logp, base_argmax, base_logdist = self.score_fn(image)
        tb = None
        if token_bits is not None:
            tb = token_bits.flatten().double().clamp_min(0)
            tb_total = float(tb.sum()) or 1.0
        cache = {}

        def point(op, keep):
            key = (op, keep.numpy().tobytes())
            if key not in cache:
                logp, argmax, logdist = self.score_fn(paint_tokens(image, keep, operator=op))
                kl = float((base_logdist.exp() * (base_logdist - logdist)).sum())
                cache[key] = (math.exp(logp), int(argmax == base_argmax), kl)
            return cache[key]

        out = {"p0": math.exp(base_logp), "argmax": int(base_argmax), "n_tokens": n, "fractions": self.fractions,
               "methods": {}}
        modes = ("deletion", "insertion") if self.insertion else ("deletion",)
        for name in names:
            order = rank_tokens(maps[name])
            entry = {}
            for op in self.operators:
                for mode in modes:
                    ps, same, kls, bx = [], [], [], []
                    for f in self.fractions:
                        k = int(round(f * n))
                        keep = keep_mask_from_order(order, k, grid, insertion=(mode == "insertion"))
                        p, s, kl = point(op, keep)
                        ps.append(p)
                        same.append(s)
                        kls.append(kl)
                        if tb is not None:
                            bx.append(float(tb[order[:k]].sum()) / tb_total if k > 0 else 0.0)
                    tag = f"{'del' if mode == 'deletion' else 'ins'}_{op}"
                    entry[f"{tag}_auc"] = trap(self.fractions, ps)
                    entry[f"{tag}_same_auc"] = trap(self.fractions, same)
                    entry[f"{tag}_kl_auc"] = trap(self.fractions, kls)
                    entry[f"{tag}_p"] = [round(x, 6) for x in ps]
                    if bx:
                        entry[f"{tag}_auc_bits"] = trap(bx, ps)
                        entry[f"{tag}_bits"] = [round(x, 6) for x in bx]
            out["methods"][name] = entry
        out["n_forwards"] = len(cache) + 1
        return out
