"""Benchmarks: question files, prompts, ground-truth token masks and the calibration/evaluation split.

Every benchmark is a JSONL file with one question per line (written by ``scripts/prepare_*.py``):

    coco     image_path, image_id, width, height, qid, type ("exist_pos" | "exist_neg" | "color"), category,
             question, polygons (COCO segmentation), boxes
    refcoco  image_path, image_id, width, height, qid, split ("testA" | "testB"), expression, bbox [x, y, w, h]
    textvqa  image_path, image_id, width, height, qid, question, answers (10 human answers)
    vstar    image_path, width, height, qid, type, text (question + lettered options + instruction),
             answer (letter), bboxes (target boxes [x, y, w, h])

Calibration/evaluation split: questions on images with odd ids form the calibration split (layer prior, tuning of
label-based baselines) and questions on images with even ids are reported (``split_parity``).
"""
from __future__ import annotations

import json
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

COCO_QUESTION = {
    "exist": "Is there a {cat} in the image? Answer yes or no.",
    "color": "What color is the {cat}? Answer with one word.",
}
COCO_TYPES = ("exist_pos", "color")              # question types with a target region (used in the paper)
REFCOCO_PROMPT = "Where is the {expr} in the image? Answer briefly."
TEXTVQA_SUFFIX = "\nAnswer the question using a single word or phrase."


def question_text(task: str, row: dict) -> str:
    """The text given to the model for a question of ``task``."""
    if task == "coco":
        return row["question"]
    if task == "textvqa":
        return row["question"] + TEXTVQA_SUFFIX
    if task == "vstar":
        return row["text"]
    if task == "refcoco":
        return REFCOCO_PROMPT.format(expr=row["expression"].strip().rstrip("."))
    raise ValueError(f"unknown task {task!r}")


# ----------------------------------------------------------------------------- COCO questions
def build_coco_questions(annotation_file: str, image_dir: str, out_file: str, *, n_images: int = 500,
                         min_area_frac: float = 0.02, max_area_frac: float = 0.5, seed: int = 0) -> int:
    """COCO val2017 questions with ground-truth polygons (500 images and seed 0 give the paper's questions).

    exist_pos  "Is there a {cat} in the image?" for a present category covering 2-50% of the image
    exist_neg  the same question for an absent, frequently co-occurring category (no target region)
    color      "What color is the {cat}?" when the category has a single instance
    Only polygon segmentations are used (crowd RLE regions are skipped).
    """
    rng = random.Random(seed)
    coco = json.loads(Path(annotation_file).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) or not isinstance(a["segmentation"], list):
            continue
        anns[a["image_id"]].append(a)
    presence = {iid: {a["category_id"] for a in lst} for iid, lst in anns.items()}
    popularity = Counter(c for s in presence.values() for c in s)
    cooc = defaultdict(Counter)
    for s in presence.values():
        for c in s:
            for d in s:
                if c != d:
                    cooc[c][d] += 1
    ids = sorted(anns)
    rng.shuffle(ids)
    rows = []
    for iid in ids:
        if len({r["image_id"] for r in rows}) >= n_images:
            break
        im = images[iid]
        area = im["width"] * im["height"]
        by_cat = defaultdict(list)
        for a in anns[iid]:
            by_cat[a["category_id"]].append(a)
        candidates = [(c, lst) for c, lst in by_cat.items()
                      if min_area_frac <= sum(a["area"] for a in lst) / area <= max_area_frac]
        if not candidates:
            continue
        c, lst = rng.choice(candidates)
        base = {"image_id": iid, "file_name": im["file_name"], "width": im["width"], "height": im["height"]}
        rows.append({**base, "qid": f"{iid}_exist_pos", "type": "exist_pos", "category": cats[c],
                     "question": COCO_QUESTION["exist"].format(cat=cats[c]), "expected": "yes",
                     "polygons": [p for a in lst for p in a["segmentation"]],
                     "boxes": [a["bbox"] for a in lst]})
        absent = [d for d, _ in cooc[c].most_common() if d not in presence[iid]]
        if not absent:
            absent = [d for d, _ in popularity.most_common() if d not in presence[iid]]
        if absent:
            d = absent[0]
            rows.append({**base, "qid": f"{iid}_exist_neg", "type": "exist_neg", "category": cats[d],
                         "question": COCO_QUESTION["exist"].format(cat=cats[d]), "expected": "no",
                         "polygons": [], "boxes": []})
        if len(lst) == 1:
            rows.append({**base, "qid": f"{iid}_color", "type": "color", "category": cats[c],
                         "question": COCO_QUESTION["color"].format(cat=cats[c]), "expected": None,
                         "polygons": [p for p in lst[0]["segmentation"]], "boxes": [lst[0]["bbox"]]})
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r in rows:
            r["image_path"] = str(Path(image_dir) / r["file_name"])
            f.write(json.dumps(r) + "\n")
    return len(rows)


def load_questions(path: str, *, types=None, limit=None, shard: tuple | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if types:
        rows = [r for r in rows if r["type"] in set(types)]
    if shard is not None:
        index, count = shard
        rows = rows[index::count]
    if limit:
        rows = rows[:limit]
    return rows


# ----------------------------------------------------------------------------- split
def split_key(meta: dict) -> int:
    """Integer image key whose parity defines the split (odd: calibration, even: evaluation).

    COCO uses the image id. RefCOCO file names ("COCO_train2014_000000296635_2.jpg") are mapped to their image, so
    all expressions of one image fall into the same split, and hashed with CRC32 (stable across processes).
    """
    iid = meta.get("image_id")
    try:
        return int(iid)
    except (TypeError, ValueError):
        key = str(iid if iid is not None else meta["qid"])
        key = re.sub(r"_\d+\.jpg$", ".jpg", key)
        return zlib.crc32(key.encode()) % 1000003


def split_parity(metas) -> tuple:
    """Boolean masks (calibration, evaluation) over a list of question metadata."""
    import numpy as np
    keys = np.array([split_key(m) for m in metas])
    return keys % 2 == 1, keys % 2 == 0


# ----------------------------------------------------------------------------- ground truth on the token grid
def frame_size(size) -> tuple:
    """``size`` as (width, height): an int means a square frame, a pair is taken as is."""
    return (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))


def rasterize_mask(row: dict, size) -> Image.Image:
    """COCO polygons rasterized in the model frame as a 0/1 PIL "L" image."""
    fw, fh = frame_size(size)
    mask = Image.new("L", (fw, fh), 0)
    if not row.get("polygons"):
        return mask
    draw = ImageDraw.Draw(mask)
    sx, sy = fw / row["width"], fh / row["height"]
    for poly in row["polygons"]:
        pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    return mask


def box_mask(row: dict, size) -> Image.Image:
    """RefCOCO box [x, y, w, h] (original pixels) rasterized in the model frame."""
    fw, fh = frame_size(size)
    mask = Image.new("L", (fw, fh), 0)
    x, y, w, h = row["bbox"]
    sx, sy = fw / row["width"], fh / row["height"]
    ImageDraw.Draw(mask).rectangle([x * sx, y * sy, (x + w) * sx - 1, (y + h) * sy - 1], fill=1)
    return mask


def boxes_mask(row: dict, size) -> Image.Image:
    """V*: union of the target boxes [x, y, w, h] (original pixels) rasterized in the model frame."""
    fw, fh = frame_size(size)
    mask = Image.new("L", (fw, fh), 0)
    sx, sy = fw / row["width"], fh / row["height"]
    draw = ImageDraw.Draw(mask)
    for x, y, w, h in row.get("bboxes") or []:
        draw.rectangle([x * sx, y * sy, max(x * sx, (x + w) * sx - 1), max(y * sy, (y + h) * sy - 1)], fill=1)
    return mask


def mask_to_grid(mask: Image.Image, grid: tuple, threshold: float = 0.25):
    """Token-level ground truth: a token is a target when at least ``threshold`` of its cell is covered.

    Returns (bool [rows, cols], coverage [rows, cols]) as torch tensors.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    arr = torch.from_numpy(np.asarray(mask, dtype="float32"))[None, None]
    frac = F.adaptive_avg_pool2d(arr, grid)[0, 0]
    return frac >= threshold, frac


def ground_truth(task: str, row: dict, frame_size_wh, grid):
    """Token-level ground truth of a localization question (None for TextVQA or questions without a region)."""
    if task == "coco" and row.get("polygons"):
        return mask_to_grid(rasterize_mask(row, frame_size_wh), grid)[0]
    if task == "refcoco":
        return mask_to_grid(box_mask(row, frame_size_wh), grid)[0]
    if task == "vstar" and row.get("bboxes"):
        return mask_to_grid(boxes_mask(row, frame_size_wh), grid, threshold=1e-6)[0]   # small targets: any overlap
    return None
