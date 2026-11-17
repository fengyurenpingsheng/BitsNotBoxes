#!/usr/bin/env python
"""WhereBits on one image and question: map, zoom-in crop and the model's answers without and with the crop.

    python scripts/demo.py --model qwen3-vl-2b --image http://images.cocodataset.org/val2017/000000039769.jpg \
        --question "What color is the remote control?" --out demo.png --device cuda

Saves a figure with the map and the crop window, the crop itself (<out>_crop.png) and prints the answers.
--passes 2 uses the two-pass variant (no gray-image passes at test time).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.models import REGISTRY  # noqa: E402


def open_image(src: str):
    from PIL import Image
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(src).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-vl-2b", choices=sorted(REGISTRY))
    ap.add_argument("--image", required=True, help="path or URL")
    ap.add_argument("--question", required=True)
    ap.add_argument("--out", default="wherebits_demo.png")
    ap.add_argument("--frac", type=float, default=0.5, help="crop window side as a fraction of the image side")
    ap.add_argument("--rule", default="map", choices=["map", "bits"], help="crop rule: map sum (primary) or bits window")
    ap.add_argument("--passes", type=int, default=3, choices=[2, 3])
    ap.add_argument("--prior", default="default", help='"default" (shipped prior), a prior JSON or "none"')
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help='e.g. "bfloat16" on a GPU (the paper used float32)')
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    from wherebits.models import frame_and_grid
    from wherebits.pipeline import WhereBitsVLM
    from wherebits.visualize import show_map
    from wherebits.zoom import answer
    image = open_image(args.image)
    wb = WhereBitsVLM(args.model, prior=None if args.prior == "none" else args.prior, device=args.device,
                      dtype=args.dtype, passes=args.passes)
    out = wb(image, args.question)
    window = out.window(args.frac, args.rule)
    crop, box = out.crop(args.frac, args.rule)
    top = sorted(range(len(out.result.layer_weights)), key=lambda l: -out.result.layer_weights[l])[:3]
    print(f"[demo] grid {out.grid}, layers with the largest prior weight: "
          + ", ".join(f"{l} ({out.result.layer_weights[l]:.2f})" for l in top))
    print(f"[demo] crop window (tokens) {window}, box (pixels) {box}")
    path = Path(args.out)
    show_map(out.frame, out.map, window, title=f"WhereBits: {args.question}", path=path)
    crop.save(path.with_name(path.stem + "_crop.png"))
    print(f"[demo] saved {path} and {path.with_name(path.stem + '_crop.png')}")
    spec, model, processor = wb.spec, wb.model, wb.processor
    frame, _ = frame_and_grid(spec, image)
    crop_frame, _ = frame_and_grid(spec, crop, upscale=True)
    images = [crop_frame] if spec.family == "llava" else [frame, crop_frame]
    a0 = answer(model, processor, spec, args.question, [frame], max_new_tokens=args.max_new_tokens, suffix="",
                device=args.device)
    a1 = answer(model, processor, spec, args.question, images, max_new_tokens=args.max_new_tokens, suffix="",
                device=args.device)
    print(f"[demo] answer without zoom: {a0!r}")
    print(f"[demo] answer with zoom:    {a1!r}")


if __name__ == "__main__":
    main()
