#!/usr/bin/env python
"""Deletion and insertion faithfulness of the maps in a bundle (scripts/build_maps.py --task coco).

Cells are replaced by gray or blurred pixels in map order and the probability of the model's first answer token is
tracked. Every curve is scored against the fraction of cells (token axis) and against the fraction of the image's
coding cost (bit axis, CompressAI mbt2018-mean at quality 4). Two controls are added to every bundle: texture-first
(cells by descending coding cost, "codec_bits") and flat-first (ascending, "flat_first").

    python scripts/faithfulness.py --model qwen3-vl-2b --questions data/coco/questions.jsonl \
        --maps results/maps/coco/qwen3-vl-2b.npz --out results/faithfulness

Output: <out>/<model>/<qid>.json. Resumable; shard with --shard/--nshards. Summaries: scripts/summarize_faithfulness.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.data import TEXTVQA_SUFFIX, load_questions  # noqa: E402
from wherebits.faithfulness import DEFAULT_FRACTIONS, FaithEvaluator  # noqa: E402
from wherebits.models import (REGISTRY, answer_distribution, build_prompt, check_grid, frame_and_grid,  # noqa: E402
                              greedy_answer, load_model, prepare)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--fractions", default=",".join(str(f) for f in DEFAULT_FRACTIONS))
    ap.add_argument("--operators", default="gray,blur")
    ap.add_argument("--codec", default="mbt2018-mean")
    ap.add_argument("--quality", type=int, default=4)
    ap.add_argument("--textvqa-suffix", action="store_true", help="append the TextVQA answer-format instruction")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from vlm_probing import Prober

    from wherebits.codec import load_codec, token_bits
    spec = REGISTRY[args.model]
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    bundle = {}
    with np.load(args.maps, allow_pickle=False) as z:
        for key in z.files:
            qid, method = key.split("::", 1)
            if not args.methods or method in args.methods:
                bundle.setdefault(qid, {})[method] = torch.from_numpy(z[key].astype(np.float32))
    rows = [r for r in load_questions(args.questions) if r["qid"] in bundle][args.shard::args.nshards]
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if not (out / f"{r['qid']}.json").exists()]
    print(f"[faith] {args.model}: {len(rows)} questions with maps, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, processor = load_model(spec, device=args.device, dtype=args.dtype)
    probe = Prober(model, processor)
    codec = load_codec(args.codec, args.quality, device=args.device)
    fractions = [float(x) for x in args.fractions.split(",")]
    operators = tuple(x for x in args.operators.split(",") if x)
    for i, row in enumerate(todo):
        t_start = time.time()
        try:
            image, grid = frame_and_grid(spec, Image.open(row["image_path"]))
            maps = {m: v.reshape(grid) for m, v in bundle[row["qid"]].items()}
            prompt = build_prompt(processor, row["question"] + (TEXTVQA_SUFFIX if args.textvqa_suffix else ""))
            inputs = prepare(probe, spec, prompt, image, device=args.device)
            check_grid(probe, inputs, spec, grid)
            answer_ids, answer_text = greedy_answer(model, processor, inputs, max_new_tokens=spec.answer_tokens)
            target = answer_ids[0]

            def score_fn(img, prompt=prompt, target=target):
                logdist = answer_distribution(probe, prepare(probe, spec, prompt, img, device=args.device))
                return float(logdist[target]), int(logdist.argmax()), logdist

            bits_grid, bpp = token_bits(codec, image, grid)
            maps.setdefault("codec_bits", bits_grid.float())           # texture-first control
            maps.setdefault("flat_first", -bits_grid.float())          # flat-first control
            faith = FaithEvaluator(score_fn, fractions=fractions, operators=operators).evaluate(
                image, maps, token_bits=bits_grid)
            rec = {"qid": row["qid"], "model": args.model, "target": target, "answer_text": answer_text, "bpp": bpp,
                   "codec": f"{args.codec}:{args.quality}", "grid": list(grid), "faith": faith,
                   "seconds": time.time() - t_start}
            (out / f"{row['qid']}.json").write_text(json.dumps(rec))
            print(f"[faith] {i + 1}/{len(todo)} {row['qid']} {rec['seconds']:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            (out / f"{row['qid']}.error").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            print(f"[faith] FAILED {row['qid']}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
