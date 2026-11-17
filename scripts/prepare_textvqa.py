#!/usr/bin/env python
"""Download a TextVQA validation subset (lmms-lab/textvqa) -> images + questions.jsonl.

    python scripts/prepare_textvqa.py --out data/textvqa --n 500

The defaults (500 questions sampled with seed 0 from the first 1,600 rows) are the settings of the paper;
data/paper_questions/textvqa.tsv lists the question ids we used.

Primary path: one validation parquet shard (~310 MB) read with pyarrow.
Fallback (no pyarrow): the Hugging Face datasets-server REST API (rows + image URLs).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

REPO = "lmms-lab/textvqa"


def rows_from_parquet(n_shards: int):
    import pyarrow.parquet as pq  # noqa: F401  (raises ImportError -> REST fallback)
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if "validation" in f and f.endswith(".parquet"))
    if not files:
        raise RuntimeError("no validation parquet files found in " + REPO)
    for f in files[:n_shards]:
        path = hf_hub_download(REPO, f, repo_type="dataset")
        table = pq.read_table(path)
        print(f"[textvqa] {f}: {table.num_rows} rows, columns={table.column_names}", flush=True)
        for rec in table.to_pylist():
            img = rec.get("image")
            data = img.get("bytes") if isinstance(img, dict) else img
            yield rec, data


def rows_from_rest(limit: int):
    base = "https://datasets-server.huggingface.co/rows?"
    offset = 0
    while offset < limit:
        q = urllib.parse.urlencode({"dataset": REPO, "config": "default", "split": "validation",
                                    "offset": offset, "length": 100})
        with urllib.request.urlopen(base + q, timeout=120) as r:
            payload = json.loads(r.read())
        batch = payload.get("rows", [])
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool", type=int, default=1600, help="sample n questions from the first `pool` rows")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    try:
        source = rows_from_parquet(a.shards)
        first = next(source)
        print("[textvqa] using parquet + pyarrow", flush=True)
    except ImportError:
        print("[textvqa] pyarrow not importable -> REST fallback (faster: pip install pyarrow)", flush=True)
        source = rows_from_rest(a.pool)
        first = next(source)

    def chain():
        yield first
        yield from source

    pool = []
    for rec, data in chain():
        if data is None or not rec.get("question") or not rec.get("answers"):
            continue
        pool.append((rec, data))
        if len(pool) >= a.pool:
            break
    random.Random(a.seed).shuffle(pool)
    chosen = pool[: a.n]
    with open(out / "questions.jsonl", "w") as f:
        for k, (rec, data) in enumerate(chosen):
            qid = str(rec.get("question_id") or f"tvqa{k}")
            img = Image.open(io.BytesIO(data)).convert("RGB")
            path = out / "images" / f"{qid}.jpg"
            img.save(path, quality=95)
            f.write(json.dumps({"qid": f"tvqa_{qid}", "question": rec["question"], "answers": list(rec["answers"]),
                                "image_path": str(path.resolve()), "width": img.width, "height": img.height,
                                "image_id": str(rec.get("image_id", "")), "type": "textvqa"}) + "\n")
    print(f"[textvqa] wrote {len(chosen)} questions to {out / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
