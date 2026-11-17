#!/usr/bin/env python
"""Zoom-in accuracy per model and method, and pooled comparisons (scripts/zoom_in.py results).

    python scripts/summarize_zoom.py --results results/zoom/vstar --models smolvlm-500m internvl3-1b qwen3-vl-2b \
        qwen3-vl-8b llava-1.5-7b --out results/zoom/vstar_summary.json

Accuracy is in percent and averaged over the two window sizes. Pooled differences average the paired differences of
each question over models and windows; the 95% interval is a bootstrap over questions (10,000 resamples) and p is a
two-sided sign-flip test (20,000 flips). ``--cluster model-question`` treats each question of each model as its own
cluster instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.stats import pooled  # noqa: E402

WINDOWS = ("w0.5", "w0.33")


def correct(r):
    return float(r["correct"]) if "correct" in r else float(r["acc"])


def load(folder):
    out = {}
    for f in sorted(Path(folder).glob("*.json")):
        r = json.loads(f.read_text())
        out[r["qid"]] = r
    return out


def table(recs):
    """(qid, window) -> {method: 0/1 or VQA accuracy}, with "none" (no zoom) in every row."""
    T = {}
    for q, r in recs.items():
        for w in WINDOWS:
            if w in r:
                T[(q, w)] = {"none": correct(r["none"]), **{m: correct(v) for m, v in r[w].items()}}
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder with one sub-folder per model")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference", default="wherebits", help="method compared with every other one")
    ap.add_argument("--cluster", default="question", choices=["question", "model-question"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    T = {m: table(load(Path(args.results) / m)) for m in args.models}
    methods = sorted({k for t in T.values() for d in t.values() for k in d})
    acc = {m: {k: 100 * float(np.mean([d[k] for d in T[m].values() if k in d])) for k in methods} for m in args.models}
    mean = {k: float(np.mean([acc[m][k] for m in args.models if k in acc[m]])) for k in methods}
    comp = {}
    for other in methods:
        if other == args.reference:
            continue
        clus = defaultdict(list)
        for m, t in T.items():
            for (q, _w), d in t.items():
                if args.reference in d and other in d:
                    key = q if args.cluster == "question" else (m, q)
                    clus[key].append(100 * (d[args.reference] - d[other]))
        if clus:
            comp[other] = pooled([np.mean(v) for v in clus.values()])
    print(f"accuracy (%), mean of the windows {', '.join(WINDOWS)}")
    print(f"{'method':22s}" + "".join(f"{m[:14]:>15s}" for m in args.models) + f"{'mean':>9s}")
    for k in sorted(methods, key=lambda k: -mean[k]):
        print(f"{k:22s}" + "".join(f"{acc[m].get(k, float('nan')):15.1f}" for m in args.models) + f"{mean[k]:9.1f}")
    print(f"\npooled {args.reference} - other (points), 95% CI, sign-flip p  [clusters: {args.cluster}]")
    for k, c in comp.items():
        print(f"  {k:20s} {c['diff']:+6.2f}  [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]  p={c['p']:.3g}  (n={c['n_clusters']})")
    if args.out:
        Path(args.out).write_text(json.dumps({"accuracy": acc, "mean": mean, "pooled": comp}, indent=1))


if __name__ == "__main__":
    main()
