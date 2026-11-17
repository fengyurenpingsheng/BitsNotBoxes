#!/usr/bin/env python
"""Maps bundles for the zoom-in and faithfulness evaluations, computed from attention dumps (no model needed).

    python scripts/build_maps.py --task vstar   --dumps results/dumps --out results/maps --models qwen3-vl-2b ...
    python scripts/build_maps.py --task textvqa --dumps results/dumps --out results/maps --models qwen3-vl-2b ...
    python scripts/build_maps.py --task coco    --dumps results/dumps --out results/maps --models qwen3-vl-2b ...

One bundle per task and model: <out>/<task>/<model>.npz with keys "<qid>::<method>" -> float16 [rows, cols].
Methods (zoom-in): wherebits, bits_window_0.5 and bits_window_0.33 (the bits-window crops as 0/1 window maps),
vicrop_tuned (ViCrop's layer tuned with COCO labels), vicrop (all layers), carve, relevance, attn_mean and center.
Methods (COCO faithfulness, questions on even image ids): wherebits, vicrop_tuned, vicrop, carve, relevance, random.
The WhereBits prior is the shipped prior of each model unless --priors is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.crop import best_window, bits_window, window_indicator  # noqa: E402
from wherebits.data import split_parity  # noqa: E402
from wherebits.dump import conditions, list_dumps, load_dump  # noqa: E402
from wherebits.offline import baseline_maps, final_map, reduce_folder, tuned_layer  # noqa: E402
from wherebits.prior import load_prior  # noqa: E402

WINDOWS = (0.5, 0.33)


def center_map(gh, gw):
    yy, xx = np.mgrid[0:gh, 0:gw]
    return np.exp(-(((yy - (gh - 1) / 2) / (gh / 4)) ** 2 + ((xx - (gw - 1) / 2) / (gw / 4)) ** 2))


def to_f16(m, grid):
    """float16 map; maps with very large values are rescaled (ranking kept) to stay within the float16 range."""
    m = np.nan_to_num(np.asarray(m, np.float64).reshape(grid), nan=0.0, posinf=0.0, neginf=0.0)
    mx = np.abs(m).max()
    if mx > 6e4:
        m = m / mx * 6e4
    return m.astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["coco", "textvqa", "vstar"])
    ap.add_argument("--dumps", required=True, help="dump root with <task>/<model>/ folders (and coco/<model>/)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--priors", default=None, help="folder with <model>.json priors (default: the shipped priors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-questions", type=int, default=None,
                    help="COCO: random subset of the evaluation questions (seed 0; the paper uses 200 for LLaVA-1.5)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    info = {}
    for model in args.models:
        prior = load_prior(Path(args.priors) / f"{model}.json" if args.priors else model)
        coco = reduce_folder(Path(args.dumps) / "coco" / model, workers=args.workers, require_gt=True)
        odd, even = split_parity([r["meta"] for r in coco])
        l_star = tuned_layer(coco, odd)                     # ViCrop's layer tuned with COCO labels
        if args.task == "coco":
            recs = coco
            keep = even.copy()
            if args.max_questions and keep.sum() > args.max_questions:
                idx = np.flatnonzero(even)
                sel = np.random.default_rng(0).choice(idx, size=args.max_questions, replace=False)
                keep = np.zeros(len(recs), bool)
                keep[sel] = True
        else:
            del coco
            recs = reduce_folder(Path(args.dumps) / args.task / model, workers=args.workers)
            keep = np.ones(len(recs), bool)
        dump_files = {Path(f).stem: f for f in list_dumps(Path(args.dumps) / args.task / model)}
        arrays = {}
        for r, k in zip(recs, keep):
            if not k:
                continue
            q, grid = r["meta"]["qid"], tuple(r["grid"])
            b = baseline_maps(r)
            maps = {"wherebits": final_map(r, prior), "vicrop_tuned": r["ratio"][l_star], "vicrop": b["vicrop"],
                    "carve": b["carve"], "relevance": b.get("relevance")}
            if args.task == "coco":
                maps["random"] = b["random"]
            else:
                maps["attn_mean"] = b["attn_mean"]
                maps["center"] = center_map(*grid).ravel()
                log2a, mass = conditions(load_dump(dump_files[q]))
                for frac in WINDOWS:
                    win = bits_window(log2a, mass, prior, grid, frac)
                    ind = window_indicator(grid, win)
                    assert best_window(ind, frac) == win
                    maps[f"bits_window_{frac}"] = ind
            for name, m in maps.items():
                if m is not None:
                    arrays[f"{q}::{name}"] = to_f16(m, grid)
        path = Path(args.out) / args.task / f"{model}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        info[model] = {"questions": int(keep.sum()), "maps": len(arrays), "vicrop_tuned_layer_from_coco": l_star}
        print(f"[maps] {args.task}/{model}: {int(keep.sum())} questions, {len(arrays)} maps, "
              f"ViCrop tuned layer {l_star}", flush=True)
    info_path = Path(args.out) / args.task / "info.json"
    old = json.loads(info_path.read_text()) if info_path.exists() else {}
    info_path.write_text(json.dumps({**old, **info}, indent=1))


if __name__ == "__main__":
    main()
