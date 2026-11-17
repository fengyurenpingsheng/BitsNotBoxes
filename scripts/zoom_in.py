#!/usr/bin/env python
"""Training-free zoom-in with the maps of a bundle (scripts/build_maps.py).

For each map and window size (about 1/2 and 1/3 of the image side), the best window is cropped from the original
image and the model answers from the image and the crop (from the crop alone for LLaVA-1.5). Without zoom, the
model sees only the image.

    python scripts/zoom_in.py --task textvqa --model qwen3-vl-2b --questions data/textvqa/questions.jsonl \
        --maps results/maps/textvqa/qwen3-vl-2b.npz --out results/zoom
    python scripts/zoom_in.py --task vstar --model qwen3-vl-2b --questions data/vstar/questions.jsonl \
        --maps results/maps/vstar/qwen3-vl-2b.npz --out results/zoom

TextVQA: greedy answers with at most 10 new tokens, VQA accuracy. V*: forced choice (the option letter with the
highest next-token probability in one forward pass); the generated no-zoom answer is stored for reference.
Output: <out>/<task>/<model>/<qid>.json. Resumable; failures leave <qid>.error. Shard with --shard/--nshards.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.crop import best_window, crop_original  # noqa: E402
from wherebits.data import TEXTVQA_SUFFIX, load_questions  # noqa: E402
from wherebits.metrics import mc_letter, vqa_accuracy  # noqa: E402
from wherebits.models import REGISTRY, frame_and_grid, load_model  # noqa: E402
from wherebits.zoom import answer, forced_choice, letter_token_ids  # noqa: E402

WINDOWS = (0.5, 0.33)


def load_bundle(path):
    by_q = defaultdict(dict)
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            qid, method = key.split("::", 1)
            by_q[qid][method] = z[key].astype(np.float32)
    return by_q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["textvqa", "vstar"])
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of the bundle's methods")
    ap.add_argument("--mode", default="", choices=["", "both", "crop"],
                    help="model input with a crop: image + crop (default) or the crop alone (default for LLaVA-1.5)")
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()

    from PIL import Image
    from vlm_probing import Prober
    spec = REGISTRY[args.model]
    mode = args.mode or ("crop" if spec.family == "llava" else "both")
    out = Path(args.out) / args.task / args.model
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.maps)
    rows = [r for r in load_questions(args.questions) if r["qid"] in bundle][args.shard::args.nshards]
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if not (out / f"{r['qid']}.json").exists()]
    print(f"[zoom] {args.task}/{args.model}: {len(rows)} questions with maps, {len(todo)} to do, mode={mode}", flush=True)
    if not todo:
        return
    model, processor = load_model(spec, device=args.device, dtype=args.dtype)
    Prober(model, processor)          # attached as in the dump runs, so the model runs exactly as there
    letters_all = letter_token_ids(processor.tokenizer) if args.task == "vstar" else None

    def respond(question, images, ids):
        if args.task == "vstar":
            r = forced_choice(model, processor, spec, question, images, ids, args.device)
            r["n_images"] = len(images)
            return r
        return {"pred": answer(model, processor, spec, question, images, max_new_tokens=args.max_new_tokens,
                               suffix=TEXTVQA_SUFFIX, device=args.device), "n_images": len(images)}

    def score(row, r):
        if args.task == "vstar":
            r["correct"] = float(r["pred"] == row["answer"])
        else:
            r["acc"] = vqa_accuracy(r["pred"], row["answers"])
        return r

    for i, row in enumerate(todo):
        t_start = time.time()
        try:
            question = row["text"] if args.task == "vstar" else row["question"]
            ids = {L: letters_all[L] for L in re.findall(r"\(([A-H])\)", question)} if letters_all else None
            original = Image.open(row["image_path"]).convert("RGB")
            frame, grid = frame_and_grid(spec, original)
            rec = {"qid": row["qid"], "model": args.model, "task": args.task, "mode": mode, "grid": list(grid),
                   "answer": row.get("answer"), "answers": row.get("answers")}
            rec["none"] = score(row, respond(question, [frame], ids))
            if args.task == "vstar":
                gen = answer(model, processor, spec, question, [frame], max_new_tokens=args.max_new_tokens, suffix="",
                             device=args.device)
                rec["none_generated"] = {"text": gen, "letter": mc_letter(gen)}
            for w in WINDOWS:
                res = {}
                for method, m in bundle[row["qid"]].items():
                    if args.methods and method not in args.methods:
                        continue
                    if method.startswith("bits_window_") and method != f"bits_window_{w}":
                        continue
                    win = best_window(m.reshape(grid), w)
                    crop, box = crop_original(original, grid, win)
                    crop_frame, _ = frame_and_grid(spec, crop, upscale=True)      # the crop gets the full token budget
                    images = [frame, crop_frame] if mode == "both" else [crop_frame]
                    try:
                        r = respond(question, images, ids)
                    except Exception:  # noqa: BLE001 - a processor that rejects two images: crop alone
                        if mode != "both":
                            raise
                        r = respond(question, [crop_frame], ids)
                    r["box"] = list(box)
                    res["bits_window" if method.startswith("bits_window_") else method] = score(row, r)
                rec[f"w{w}"] = res
            rec["seconds"] = time.time() - t_start
            (out / f"{row['qid']}.json").write_text(json.dumps(rec))
            print(f"[zoom] {i + 1}/{len(todo)} {row['qid']} {rec['seconds']:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            (out / f"{row['qid']}.error").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            print(f"[zoom] FAILED {row['qid']}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
