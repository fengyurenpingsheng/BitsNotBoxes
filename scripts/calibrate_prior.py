#!/usr/bin/env python
"""Estimate the layer prior of a model from unlabeled COCO calibration questions (no localization labels).

The prior uses only the attention dumps of the questions on images with odd ids (the calibration split):
Bbar_l = mean over these questions of the layer bits W_l = sum_h w_lh, and pi_l = Bbar_l^4 / sum_j Bbar_j^4.

    python scripts/calibrate_prior.py --dumps results/dumps/coco --models qwen3-vl-2b llava-1.5-7b --out priors
    python scripts/calibrate_prior.py --dumps results/dumps/coco --models qwen3-vl-2b --all-questions --out priors

``--all-questions`` uses every dumped question, for a new model or input size calibrated on your own unlabeled data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.core import GAMMA  # noqa: E402
from wherebits.data import split_parity  # noqa: E402
from wherebits.offline import mean_layer_bits, reduce_folder  # noqa: E402
from wherebits.prior import save_prior  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="folder with one sub-folder of dumps per model (e.g. results/dumps/coco)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output folder for <model>.json")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--all-questions", action="store_true", help="use every dumped question, not only odd image ids")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    for model in args.models:
        recs = reduce_folder(Path(args.dumps) / model, workers=args.workers)
        if not recs:
            print(f"[prior] {model}: no dumps in {Path(args.dumps) / model}", flush=True)
            continue
        if args.all_questions:
            use = recs
            source = f"{len(use)} unlabeled questions in {Path(args.dumps).name}"
        else:
            odd, _ = split_parity([r["meta"] for r in recs])
            use = [r for r, o in zip(recs, odd) if o]
            source = "COCO val2017 calibration split (questions on images with odd ids), no labels used"
        bits = mean_layer_bits(use)
        data = save_prior(Path(args.out) / f"{model}.json", model, bits, n_questions=len(use), source=source,
                          gamma=args.gamma)
        top = np.argsort(data["prior"])[::-1][:3]
        print(f"[prior] {model}: {len(use)} questions, {len(bits)} layers, top layers {top.tolist()} "
              f"(weights {[round(data['prior'][i], 3) for i in top]})", flush=True)


if __name__ == "__main__":
    main()
