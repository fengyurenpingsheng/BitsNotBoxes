#!/usr/bin/env python
"""Localization accuracy (pointing and token AUROC) of WhereBits and the baselines from attention dumps.

COCO and RefCOCO report questions on images with even ids; label-based baselines (ViCrop's tuned layer) are tuned on
the odd ids of the same benchmark. V* reports all questions, with ViCrop's layer tuned on COCO labels. The WhereBits
prior always comes from the unlabeled COCO calibration questions (a shipped prior or one from calibrate_prior.py).

    python scripts/evaluate_localization.py --task coco --dumps results/dumps --out results/localization \
        --models smolvlm-256m smolvlm-500m internvl3-1b qwen3-vl-2b qwen3-vl-8b llava-1.5-7b
    python scripts/evaluate_localization.py --task vstar --dumps results/dumps --out results/localization \
        --models smolvlm-500m internvl3-1b qwen3-vl-2b qwen3-vl-8b llava-1.5-7b

Writes <out>/<task>.json (per model: summaries with bootstrap intervals, paired Wilcoxon tests against WhereBits,
per-question values) and prints a table. --ablations adds the single-change variants of the ablation table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.data import split_parity  # noqa: E402
from wherebits.metrics import localization  # noqa: E402
from wherebits.offline import baseline_maps, final_map, reduce_folder, tuned_layer  # noqa: E402
from wherebits.core import layer_prior  # noqa: E402
from wherebits.prior import load_prior  # noqa: E402
from wherebits.stats import boot, paired  # noqa: E402

ORDER = ["wherebits", "wherebits_two_pass", "wherebits_per_question", "vicrop_tuned", "vicrop", "vicrop_band",
         "carve", "relevance", "grad_cam", "rollout", "tokenact", "attn_mean", "kl_weighted", "random"]
NAMES = {"wherebits": "WhereBits (ours)", "wherebits_two_pass": "WhereBits, two passes",
         "wherebits_per_question": "WhereBits, per-question layers", "vicrop_tuned": "ViCrop, tuned layer (labels)",
         "vicrop": "ViCrop, all layers", "vicrop_band": "ViCrop, layer band", "carve": "CARVE (re-impl.)",
         "relevance": "Chefer relevance", "grad_cam": "Attention Grad-CAM", "rollout": "Attention rollout",
         "tokenact": "Token activation map", "attn_mean": "Mean attention", "kl_weighted": "KL-weighted map",
         "random": "Random", "abl_uniform_layers": "  no prior (uniform layers)",
         "abl_gamma1": "  prior exponent 1 instead of 4", "abl_prior_no_gray": "  prior without gray correction",
         "abl_heads_no_gray": "  head weights without gray correction", "abl_heads_kl": "  head weights KL only (no mass)",
         "abl_heads_uniform": "  uniform head weights", "abl_head_mean": "  log-ratio of head mean (ViCrop-style)"}
ABLATIONS = ["abl_uniform_layers", "abl_gamma1", "abl_prior_no_gray", "abl_heads_no_gray", "abl_heads_kl",
             "abl_heads_uniform", "abl_head_mean"]


def evaluate_model(task, model, dumps, prior, workers, ablations=False, prior_bits=None):
    recs = reduce_folder(Path(dumps) / task / model, workers=workers, require_gt=True, ablations=ablations)
    if not recs:
        raise SystemExit(f"no dumps with ground truth in {Path(dumps) / task / model}")
    coco = recs if task == "coco" else None
    if task == "vstar" or (ablations and task != "coco"):
        coco = reduce_folder(Path(dumps) / "coco" / model, workers=workers, require_gt=True, ablations=ablations)
    if task == "vstar":
        report = np.ones(len(recs), bool)
        l_star = tuned_layer(coco, split_parity([r["meta"] for r in coco])[0])
    else:
        odd, report = split_parity([r["meta"] for r in recs])
        l_star = tuned_layer(recs, odd)
    variants = {}
    if ablations:
        cal = split_parity([r["meta"] for r in coco])[0]
        L = recs[0]["L"]
        variants = {"abl_uniform_layers": ("F", np.full(L, 1.0 / L)), "abl_gamma1": ("F", layer_prior(prior_bits, 1.0)),
                    "abl_prior_no_gray": ("F", layer_prior(np.mean([r["W_where"] for r, c in zip(coco, cal) if c], 0))),
                    "abl_heads_no_gray": ("F_where", prior), "abl_heads_kl": ("F_kl", prior),
                    "abl_heads_uniform": ("F_uniform", prior), "abl_head_mean": ("mix_dq", prior)}
    del coco
    per = {}
    for r, keep in zip(recs, report):
        if not keep:
            continue
        maps = {"wherebits": final_map(r, prior), "wherebits_two_pass": final_map(r, prior, layers="F_mass"),
                "wherebits_per_question": final_map(r, None),
                "vicrop_tuned": r["ratio"][l_star].astype(np.float64), **baseline_maps(r)}
        for name, (layers, pi) in variants.items():
            maps[name] = final_map(r, pi, layers=layers)
        for name, m in maps.items():
            for key, val in localization(m, r["gt"]).items():
                per.setdefault(name, {}).setdefault(key, []).append(val)
    summary = {n: {k: boot(v) for k, v in d.items()} for n, d in per.items()}
    tests = {n: {k: paired(per["wherebits"][k], per[n][k]) for k in ("pointing", "auroc")}
             for n in per if n != "wherebits"}
    return {"n": int(report.sum()), "tuned_layer": l_star, "summary": summary, "tests": tests,
            "per_question": {n: {k: [float(x) for x in v] for k, v in d.items()} for n, d in per.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["coco", "refcoco", "vstar"])
    ap.add_argument("--dumps", required=True, help="dump root with <task>/<model>/ folders")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--priors", default=None, help="folder with <model>.json priors (default: the shipped priors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ablations", action="store_true", help="also evaluate the ablation variants")
    args = ap.parse_args()
    res = {}
    for model in args.models:
        src = Path(args.priors) / f"{model}.json" if args.priors else model
        prior = load_prior(src)
        bits = None
        if args.ablations:
            from wherebits.prior import prior_data
            bits = np.asarray(prior_data(src)["layer_bits"], np.float64)
        res[model] = evaluate_model(args.task, model, args.dumps, prior, args.workers, args.ablations, bits)
        s = res[model]["summary"]
        print(f"\n[{args.task}] {model}: n={res[model]['n']}, ViCrop tuned layer {res[model]['tuned_layer']}", flush=True)
        for n in ORDER + ABLATIONS:
            if n in s:
                t = res[model]["tests"].get(n)
                star = "" if t is None else ("*" if t["pointing"]["p"] < 0.05 else " ") + ("*" if t["auroc"]["p"] < 0.05 else " ")
                print(f"   {NAMES[n]:34s} pointing {s[n]['pointing'][0]:.3f}   AUROC {s[n]['auroc'][0]:.3f} {star}", flush=True)
    models = list(res)
    print(f"\n[{args.task}] mean over {len(models)} models (* = Wilcoxon p<0.05 against WhereBits, pointing/AUROC)")
    for n in ORDER + ABLATIONS:
        if all(n in res[m]["summary"] for m in models):
            pt = np.mean([res[m]["summary"][n]["pointing"][0] for m in models])
            au = np.mean([res[m]["summary"][n]["auroc"][0] for m in models])
            print(f"   {NAMES[n]:34s} pointing {pt:.3f}   AUROC {au:.3f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.task}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
