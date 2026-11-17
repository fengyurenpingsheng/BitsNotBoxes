#!/usr/bin/env python
"""Per-question attention dumps: the answer-start attention row of every layer and head under the four input
conditions (image or gray image x question or generic prompt), plus the vlmprobe baseline maps.

    python scripts/dump_attention.py --task coco    --model qwen3-vl-2b --questions data/coco/questions.jsonl --out results/dumps
    python scripts/dump_attention.py --task refcoco --model qwen3-vl-2b --questions data/refcoco/questions.jsonl --out results/dumps
    python scripts/dump_attention.py --task textvqa --model qwen3-vl-2b --questions data/textvqa/questions.jsonl --out results/dumps
    python scripts/dump_attention.py --task vstar   --model qwen3-vl-2b --questions data/vstar/questions.jsonl --out results/dumps

Per question: the model's greedy answer, four forward passes (the image/question pass also runs one backward pass to
store gradient statistics) and the baselines (Chefer relevance, attention Grad-CAM, rollout, token activation map).
Output: <out>/<task>/<model><tag>/<qid>.npz (format: wherebits/dump.py). Resumable: finished questions are skipped,
failures leave <qid>.error. Shard large jobs with --shard/--nshards (or a SLURM array, see slurm/).

Sensitivity runs of the supplement: --generic-prompt {describe,see,caption} and --gray 64|192 with a --tag.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.capture import GENERIC_PROMPTS, GRAY, capture_rows  # noqa: E402
from wherebits.data import COCO_TYPES, ground_truth, load_questions, question_text  # noqa: E402
from wherebits.dump import CONDITIONS, pack_rows, save_dump  # noqa: E402
from wherebits.models import (REGISTRY, TokenLogProb, build_prompt, check_grid, frame_and_grid,  # noqa: E402
                              greedy_answer, load_model, prepare)
from wherebits.probe_baselines import token_activation_map, visual_hidden_norms, vlmprobe_baselines  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["coco", "refcoco", "textvqa", "vstar"])
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--types", nargs="*", default=None, help=f"COCO question types (default: {' '.join(COCO_TYPES)})")
    ap.add_argument("--no-baselines", action="store_true", help="skip the vlmprobe baseline maps")
    ap.add_argument("--no-hidden", action="store_true", help="skip the hidden-state norms")
    ap.add_argument("--generic-prompt", default="vicrop", help=f"one of {sorted(GENERIC_PROMPTS)} or a literal prompt")
    ap.add_argument("--gray", type=int, default=None, help=f"gray level of the content-free image (default {GRAY[0]})")
    ap.add_argument("--tag", default="", help="suffix of the output folder, e.g. _p-describe or _g64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="override the model dtype (the paper used float32)")
    return ap.parse_args()


def main():
    args = parse()
    import torch
    from PIL import Image
    from vlm_probing import Prober
    spec = REGISTRY[args.model]
    out = Path(args.out) / args.task / f"{args.model}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    generic = GENERIC_PROMPTS.get(args.generic_prompt, args.generic_prompt)
    gray_rgb = tuple(GRAY) if args.gray is None else (args.gray,) * 3
    types = args.types or (list(COCO_TYPES) if args.task == "coco" else None)
    rows = load_questions(args.questions, types=types, limit=args.limit, shard=(args.shard, args.nshards))
    todo = [r for r in rows if not (out / f"{r['qid']}.npz").exists()]
    print(f"[dump] {args.task}/{args.model} shard {args.shard}/{args.nshards}: {len(rows)} questions, {len(todo)} to do",
          flush=True)
    if not todo:
        return
    model, processor = load_model(spec, device=args.device, dtype=args.dtype)
    probe = Prober(model, processor)
    layers = sorted(probe.spec.attentions)
    prompt_g = build_prompt(processor, generic)
    for i, row in enumerate(todo):
        t_start = time.time()
        timing, skipped = {}, {}
        try:
            original = Image.open(row["image_path"]).convert("RGB")
            frame, grid = frame_and_grid(spec, original)
            gray = Image.new("RGB", frame.size, gray_rgb)
            text = question_text(args.task, row)
            prompt_t = build_prompt(processor, text)
            in_xt = prepare(probe, spec, prompt_t, frame, device=args.device)
            check_grid(probe, in_xt, spec, grid)
            answer_ids, answer_text = greedy_answer(model, processor, in_xt, max_new_tokens=spec.answer_tokens)
            target = answer_ids[0]
            metric = TokenLogProb(target)
            t0 = time.time()
            r_xt = capture_rows(probe, in_xt, layers, metric=metric)
            r_0t = capture_rows(probe, prepare(probe, spec, prompt_t, gray, device=args.device), layers)
            r_xg = capture_rows(probe, prepare(probe, spec, prompt_g, frame, device=args.device), layers)
            r_0g = capture_rows(probe, prepare(probe, spec, prompt_g, gray, device=args.device), layers)
            timing["rows"] = time.time() - t0
            arrays = pack_rows(r_xt, r_0t, r_xg, r_0g)
            n = grid[0] * grid[1]
            if arrays["la"].shape[-1] != n:
                raise RuntimeError(f"{arrays['la'].shape[-1]} visual keys, the grid needs {n}")
            base_names = []
            if not args.no_baselines:
                base, bt, sk = vlmprobe_baselines(probe, in_xt, metric)
                timing.update(bt)
                skipped.update(sk)
                try:
                    t0 = time.time()
                    base["tokenact"] = token_activation_map(probe, in_xt, target)
                    timing["tokenact"] = time.time() - t0
                except Exception as e:  # noqa: BLE001 - optional baseline
                    skipped["tokenact"] = f"{type(e).__name__}: {e}"
                vecs = []
                for name, v in base.items():
                    v = torch.nan_to_num(v.float().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
                    if v.numel() == n:
                        base_names.append(name)
                        vecs.append(v)
                    else:
                        skipped[name] = f"map has {v.numel()} entries, the grid needs {n}"
                if vecs:
                    arrays["base"] = torch.stack(vecs).half().numpy()
            if not args.no_hidden:
                try:
                    arrays["hn"] = visual_hidden_norms(probe, in_xt).half().numpy()
                except Exception as e:  # noqa: BLE001 - optional diagnostic
                    skipped["hidden_norms"] = f"{type(e).__name__}: {e}"
            gt = ground_truth(args.task, row, frame.size, grid)
            if gt is not None:
                arrays["gt"] = gt.numpy().astype(bool)
            L, H, N = arrays["la"].shape[1:]
            meta = {
                "task": args.task, "model": args.model, "qid": row["qid"], "type": row.get("type", args.task),
                "category": row.get("category"), "question": text, "image_id": row.get("image_id"),
                "image_path": row["image_path"], "width": original.width, "height": original.height,
                "answer_text": answer_text, "answer_ids": answer_ids, "target": target,
                "target_text": processor.tokenizer.decode([target]), "score_logp": r_xt.score,
                "grid": list(grid), "frame": list(frame.size), "n_layers": int(L), "n_heads": int(H), "n_visual": int(N),
                "conditions": list(CONDITIONS), "generic_prompt": generic, "gray": list(gray_rgb),
                "base_names": base_names, "answers": row.get("answers", [row["answer"]] if "answer" in row else None),
                "bbox": row.get("bbox", row.get("bboxes")), "expression": row.get("expression"),
                "split": row.get("split"), "timing": timing, "skipped": skipped,
            }
            meta["seconds"] = time.time() - t_start
            save_dump(out / f"{row['qid']}.npz", arrays, meta)
            print(f"[dump] {i + 1}/{len(todo)} {row['qid']} answer={answer_text!r} L={L} H={H} N={N} "
                  f"{meta['seconds']:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 - keep the shard alive
            (out / f"{row['qid']}.error").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            print(f"[dump] FAILED {row['qid']}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
