#!/usr/bin/env python
"""V* Bench (Wu and Xie, 2024; Hugging Face: craigwu/vstar_bench) -> data/vstar/questions.jsonl.

191 high-resolution images (115 direct-attribute + 76 relative-position questions) whose targets are small.
Each per-image annotation json carries the question, the options (the correct one is listed FIRST, as in the
official evaluation) and, when present, the target objects' boxes [x, y, w, h] in original pixels.
We present the options in a deterministic shuffled order (seeded by the image name), so the correct letter is not
always "A", and keep every box (relative-position questions have two targets) as ground truth for localization.

    python scripts/prepare_vstar.py --out data/vstar

data/paper_questions/vstar.tsv lists the 191 question ids and correct letters of the paper.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

LETTERS = "ABCDEFGH"
INSTRUCTION = "Answer with the option's letter from the given choices directly."


def find_boxes(ann: dict):
    for key in ("bbox", "bboxes", "boxes", "target_bbox"):
        if key in ann and ann[key]:
            b = ann[key]
            if isinstance(b[0], (int, float)):
                b = [b]
            return [[float(v) for v in box[:4]] for box in b]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="craigwu/vstar_bench")
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    from huggingface_hub import snapshot_download
    snapshot_download(args.repo, repo_type="dataset", local_dir=str(raw),
                      allow_patterns=["test_questions.jsonl", "direct_attributes/*", "relative_position/*"])
    rows, skipped = [], []
    for cat in ("direct_attributes", "relative_position"):
        folder = raw / cat
        for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            ann_path = img.with_suffix(".json")
            if not ann_path.exists():
                skipped.append((img.name, "no json")); continue
            ann = json.loads(ann_path.read_text())
            q, opts = ann.get("question"), ann.get("options")
            if not q or not opts or len(opts) < 2:
                skipped.append((img.name, f"keys={sorted(ann)}")); continue
            order = list(range(len(opts)))
            random.Random(f"vstar:{cat}:{img.stem}").shuffle(order)
            shown = [opts[i] for i in order]
            letter = LETTERS[order.index(0)]                    # the correct option is opts[0]
            text = q.strip() + "\n" + "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(shown)) + "\n" + INSTRUCTION
            with Image.open(img) as im:
                W, H = im.size
            rows.append({"qid": f"vstar_{cat}_{img.stem}", "type": cat, "category": cat,
                         "image_path": str(img.resolve()), "width": W, "height": H,
                         "question": q.strip(), "text": text, "options": shown, "answer": letter,
                         "answers": [letter], "bboxes": find_boxes(ann),
                         "target_object": ann.get("target_object")})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "questions.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    by = {c: sum(r["category"] == c for r in rows) for c in ("direct_attributes", "relative_position")}
    nb = sum(bool(r["bboxes"]) for r in rows)
    letters = {L: sum(r["answer"] == L for r in rows) for L in LETTERS[:4]}
    print(f"[vstar] {len(rows)} questions {by}; with boxes: {nb}; correct-letter counts {letters}; skipped {len(skipped)}")
    if skipped:
        print("[vstar] skipped examples:", skipped[:5])
    if rows:
        ex = dict(rows[0]); ex["text"] = ex["text"].replace("\n", " | ")
        print("[vstar] example:", json.dumps(ex)[:600])
    if not rows or nb == 0:
        print("[vstar] WARNING: no usable questions or no boxes found - check the dataset files", file=sys.stderr)
        sys.exit(1 if not rows else 0)


if __name__ == "__main__":
    main()
