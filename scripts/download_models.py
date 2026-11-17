#!/usr/bin/env python
"""Download model weights (Hugging Face) and the CompressAI codecs ahead of time, e.g. on a login node with internet
before running jobs offline (HF_HUB_OFFLINE=1).

    python scripts/download_models.py --models qwen3-vl-2b llava-1.5-7b
    python scripts/download_models.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wherebits.models import REGISTRY  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], choices=sorted(REGISTRY))
    ap.add_argument("--all", action="store_true", help="every model of the registry")
    ap.add_argument("--codecs", nargs="*", default=["mbt2018-mean:4"],
                    help="name:quality (the paper's bit axis uses mbt2018-mean:4; the supplement adds 2, 6 and "
                         "bmshj2018-hyperprior:4)")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    ids = sorted({REGISTRY[k].model_id for k in (REGISTRY if args.all else args.models)})
    for model_id in ids:
        print(f"[hf] {model_id}", flush=True)
        snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py",
                                                    "*.tiktoken", "*.jinja"])
    if args.codecs:
        from compressai.zoo import bmshj2018_hyperprior, mbt2018_mean
        for item in args.codecs:
            name, q = item.split(":")
            {"mbt2018-mean": mbt2018_mean, "bmshj2018-hyperprior": bmshj2018_hyperprior}[name](quality=int(q),
                                                                                               pretrained=True)
            print(f"[codec] {name} quality {q}", flush=True)


if __name__ == "__main__":
    main()
