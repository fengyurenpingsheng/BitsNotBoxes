#!/usr/bin/env python
"""Download a RefCOCO (UNC) testA + testB subset from lmms-lab/RefCOCO -> images + questions.jsonl.

    python scripts/prepare_refcoco.py --out data/refcoco --n-per-split 300

Each row of lmms-lab/RefCOCO is one object: image, bbox (COCO [x, y, w, h], original pixels),
answer = list of referring expressions. We keep one expression per object (seeded choice).
testA = people, testB = other objects. Fast path with pyarrow (pip install pyarrow); fallback: the Hugging Face datasets-server REST API.
The defaults (300 per split, seed 0) are the settings of the paper; data/paper_questions/refcoco.tsv lists the
600 expressions we used and their split (calibration: odd image key, evaluation: even).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

REPO = "lmms-lab/RefCOCO"


def parquet_rows(split: str):
    import pyarrow.parquet as pq  # noqa: F401  (ImportError -> REST fallback)
    from huggingface_hub import hf_hub_download, list_repo_files
    pat = re.compile(rf"(^|/){split}[-_.]")
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet") and pat.search(f))
    if not files:
        raise RuntimeError(f"no parquet for split {split} in {REPO}")
    for f in files:
        path = hf_hub_download(REPO, f, repo_type="dataset")
        table = pq.read_table(path)
        print(f"[refcoco] {f}: {table.num_rows} rows, columns={table.column_names}", flush=True)
        for rec in table.to_pylist():
            img = rec.get("image")
            yield rec, (img.get("bytes") if isinstance(img, dict) else img)


def rest_rows(split: str, limit: int):
    base = "https://datasets-server.huggingface.co/rows?"
    offset = 0
    while offset < limit:
        q = urllib.parse.urlencode({"dataset": REPO, "config": "default", "split": split, "offset": offset, "length": 100})
        with urllib.request.urlopen(base + q, timeout=120) as r:
            batch = json.loads(r.read()).get("rows", [])
        if not batch:
            break
        for item in batch:
            rec = item["row"]
            img = rec.get("image")
            src = img.get("src") if isinstance(img, dict) else None
            data = None
            if src:
                with urllib.request.urlopen(src, timeout=120) as r:
                    data = r.read()
            yield rec, data
        offset += len(batch)


def to_xywh(bbox, width, height):
    x, y, a, b = [float(v) for v in bbox]
    if a > x and b > y and (x + a > width + 1 or y + b > height + 1):   # looks like [x1, y1, x2, y2]
        return [x, y, a - x, b - y]
    return [x, y, a, b]


def expressions(rec):
    ans = rec.get("answer")
    if isinstance(ans, str):
        try:
            parsed = json.loads(ans.replace("'", '"'))
            ans = parsed if isinstance(parsed, list) else [ans]
        except Exception:  # noqa: BLE001
            ans = [ans]
    return [a for a in (ans or []) if isinstance(a, str) and a.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["testA", "testB"])
    ap.add_argument("--n-per-split", type=int, default=300)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area", type=float, default=0.01, help="drop boxes smaller than this image fraction")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    written = 0
    with open(out / "questions.jsonl", "w") as f:
        for split in a.splits:
            try:
                src = parquet_rows(split)
                first = next(src)
                print(f"[refcoco] {split}: parquet + pyarrow", flush=True)
            except ImportError:
                print("[refcoco] pyarrow not importable -> REST fallback (faster: pip install pyarrow)", flush=True)
                src = rest_rows(split, a.pool)
                first = next(src)

            def chain(first=first, src=src):
                yield first
                yield from src

            pool = []
            for rec, data in chain():
                exprs = expressions(rec)
                if data is None or not exprs or rec.get("bbox") is None:
                    continue
                pool.append((rec, data, exprs))
                if len(pool) >= a.pool:
                    break
            rng.shuffle(pool)
            kept = 0
            for rec, data, exprs in pool:
                if kept >= a.n_per_split:
                    break
                img = Image.open(io.BytesIO(data)).convert("RGB")
                box = to_xywh(rec["bbox"], img.width, img.height)
                if box[2] * box[3] < a.min_area * img.width * img.height:
                    continue
                key = str(rec.get("question_id") or f"{split}{kept}")
                path = out / "images" / f"{split}_{key}.jpg"
                img.save(path, quality=95)
                expr = rng.choice(exprs)
                f.write(json.dumps({"qid": f"refcoco_{split}_{key}", "type": f"refcoco_{split}", "split": split,
                                    "expression": expr, "all_expressions": exprs, "bbox": box,
                                    "width": img.width, "height": img.height, "image_path": str(path.resolve()),
                                    "file_name": rec.get("file_name"), "image_id": rec.get("file_name")}) + "\n")
                if kept == 0:
                    print(f"[refcoco] sample {split}: {expr!r} bbox(xywh)={box} image={img.size}", flush=True)
                kept += 1
            written += kept
            print(f"[refcoco] {split}: kept {kept}", flush=True)
    print(f"[refcoco] wrote {written} expressions to {out / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
