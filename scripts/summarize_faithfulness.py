#!/usr/bin/env python
"""Faithfulness summaries (scripts/faithfulness.py results): insertion AUC minus deletion AUC per removal operator
(gray, blur) and budget axis (token, bit), mean ranks of the maps and paired Wilcoxon tests against WhereBits with
Holm correction over the settings (model x operator x axis).

    python scripts/summarize_faithfulness.py --results results/faithfulness --models smolvlm-256m qwen3-vl-2b ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.stats import holm, paired  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference", default="wherebits")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    settings, ranks, tests = {}, {}, []
    for model in args.models:
        recs = [json.loads(f.read_text()) for f in sorted((Path(args.results) / model).glob("*.json"))]
        if not recs:
            continue
        first = recs[0]["faith"]["methods"][args.reference]
        ops = [op for op in ("gray", "blur") if f"del_{op}_auc" in first]
        for op in ops:
            for axis, suffix in (("token", ""), ("bit", "_bits")):
                comb = {}
                for m in recs[0]["faith"]["methods"]:
                    d = np.array([r["faith"]["methods"][m][f"del_{op}_auc{suffix}"] for r in recs])
                    i = np.array([r["faith"]["methods"][m][f"ins_{op}_auc{suffix}"] for r in recs])
                    comb[m] = i - d
                key = f"{model}|{op}|{axis}"
                means = {m: float(np.nanmean(v)) for m, v in comb.items()}
                settings[key] = {"n": len(recs), "ins_minus_del": means,
                                 "tests": {m: paired(comb[args.reference], comb[m]) for m in comb if m != args.reference}}
                r = rankdata([-means[m] for m in means])
                for m, rk in zip(means, r):
                    ranks.setdefault(m, []).append(float(rk))
                for m, t in settings[key]["tests"].items():
                    tests.append((key, m, t["diff"], t["p"]))
    for m in sorted({t[1] for t in tests}):
        sel = [t for t in tests if t[1] == m]
        adj = holm([t[3] for t in sel])
        better = sum(1 for t, a in zip(sel, adj) if a < 0.05 and t[2] > 0)
        worse = sum(1 for t, a in zip(sel, adj) if a < 0.05 and t[2] < 0)
        print(f"{args.reference} vs {m:14s}: better in {better}, worse in {worse}, n.s. in {len(sel) - better - worse} "
              f"of {len(sel)} settings (Holm)")
    print("\nmean rank by insertion - deletion AUC (1 = best):")
    for m, v in sorted(ranks.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {m:16s} {np.mean(v):.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"settings": settings, "mean_rank": {m: float(np.mean(v)) for m, v in ranks.items()}},
                                             indent=1))


if __name__ == "__main__":
    main()
