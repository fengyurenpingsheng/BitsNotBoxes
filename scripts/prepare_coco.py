#!/usr/bin/env python
"""Download COCO val2017 (images and instance annotations) and build the COCO localization questions.

    python scripts/prepare_coco.py --data data/coco --out data/coco/questions.jsonl

500 images and seed 0 are the settings of the paper. Its localization questions are the presence and color questions
(778; those on images with odd ids form the calibration split); data/paper_questions/coco.tsv lists them.
Already downloaded files are skipped.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.data import build_coco_questions  # noqa: E402

COCO_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch(url, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest}")
        return dest
    print(f"[download] {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="COCO folder (val2017/ and annotations/ are created here)")
    ap.add_argument("--out", required=True, help="questions JSONL")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data)
    if not (data / "val2017").exists():
        zipfile.ZipFile(fetch(COCO_IMAGES, data / "val2017.zip")).extractall(data)
    ann = data / "annotations" / "instances_val2017.json"
    if not ann.exists():
        with zipfile.ZipFile(fetch(COCO_ANN, data / "annotations_trainval2017.zip")) as z:
            z.extract("annotations/instances_val2017.json", data)
    n = build_coco_questions(str(ann), str(data / "val2017"), args.out, n_images=args.n_images, seed=args.seed)
    print(f"[coco] wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()
