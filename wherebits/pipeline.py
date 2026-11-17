"""WhereBits on a Hugging Face VLM: image + question -> map and zoom-in crop. Needs torch, transformers and vlmprobe.

    from PIL import Image
    from wherebits import WhereBitsVLM

    wb = WhereBitsVLM("qwen3-vl-2b", device="cuda")          # loads the model and its shipped layer prior
    out = wb(Image.open("street.jpg"), "What is the color of the umbrella?")
    out.map                    # [rows, cols] map over the visual-token grid
    crop, box = out.crop(0.5)  # window of half the image side with the largest positive map mass
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .capture import GENERIC_PROMPT, GRAY, capture_rows, visual_attention
from .core import WhereBitsResult, from_attention, wherebits
from .crop import best_window, bits_window, crop_original
from .models import REGISTRY, build_prompt, check_grid, frame_and_grid, load_model, prepare
from .prior import load_prior


@dataclass
class Explanation:
    image: object                     # original PIL image
    frame: object                     # the model's input frame (PIL)
    grid: tuple                       # (rows, cols) of visual tokens
    result: WhereBitsResult
    log2a: dict = field(repr=False)   # condition -> log2 attention [L, H, N]
    mass: dict = field(repr=False)    # condition -> visual mass [L, H]

    @property
    def map(self) -> np.ndarray:
        """Final map M on the token grid [rows, cols]."""
        return self.result.as_grid(self.grid)

    def window(self, frac: float = 0.5, rule: str = "map"):
        """Token window (i, j, h, w) of side ``frac``: rule "map" (primary) or "bits" (bits window)."""
        if rule == "map":
            return best_window(self.map, frac)
        if rule == "bits":
            if not {"0t", "0g"} <= set(self.log2a):
                raise ValueError("the bits window needs the gray-image passes (passes=3)")
            return bits_window(self.log2a, self.mass, self.result.layer_weights, self.grid, frac)
        raise ValueError(f"unknown rule {rule!r}")

    def crop(self, frac: float = 0.5, rule: str = "map"):
        """Crop of the original image for the chosen window: (PIL image, box in original pixels)."""
        return crop_original(self.image, self.grid, self.window(frac, rule))


class WhereBitsVLM:
    """WhereBits for a model of ``wherebits.models.REGISTRY``.

    prior:  "default" (the shipped prior of this model), a prior JSON path, an array [L], or None (the question's own
            top-3 layers, no calibration needed).
    passes: 3 (paper default: image/question, image/generic prompt, gray image/question; the gray image with the
            generic prompt is cached per frame size) or 2 (head weights from the visual mass; needs a prior).
    """

    def __init__(self, model: str, prior="default", *, device: str = "cpu", dtype: str | None = None, passes: int = 3,
                 generic_prompt: str = GENERIC_PROMPT, gray=GRAY):
        from vlm_probing import Prober
        if passes not in (2, 3):
            raise ValueError("passes must be 2 or 3")
        self.spec = REGISTRY[model]
        self.device, self.passes = device, passes
        self.model, self.processor = load_model(self.spec, device=device, dtype=dtype)
        self.probe = Prober(self.model, self.processor)
        self.layers = sorted(self.probe.spec.attentions)
        if isinstance(prior, str) and prior == "default":
            self.prior = load_prior(model)
        elif isinstance(prior, (str, os.PathLike)):
            self.prior = load_prior(prior)
        else:
            self.prior = None if prior is None else np.asarray(prior, np.float64)
        if passes == 2 and self.prior is None:
            raise ValueError("the two-pass variant needs a layer prior")
        self.prompt_g = build_prompt(self.processor, generic_prompt)
        self.gray = tuple(gray)
        self._gray_generic = {}       # frame size -> visual attention of the gray image with the generic prompt

    def _attention(self, prompt: str, image, grid=None):
        inputs = prepare(self.probe, self.spec, prompt, image, device=self.device)
        if grid is not None:
            check_grid(self.probe, inputs, self.spec, grid)
        return visual_attention(capture_rows(self.probe, inputs, self.layers))

    def __call__(self, image, question: str) -> Explanation:
        image = image.convert("RGB")
        frame, grid = frame_and_grid(self.spec, image)
        prompt_t = build_prompt(self.processor, question)
        attn = {"xt": self._attention(prompt_t, frame, grid), "xg": self._attention(self.prompt_g, frame)}
        if self.passes == 3:
            from PIL import Image
            gray = Image.new("RGB", frame.size, self.gray)
            attn["0t"] = self._attention(prompt_t, gray)
            if frame.size not in self._gray_generic:
                self._gray_generic[frame.size] = self._attention(self.prompt_g, gray)
            attn["0g"] = self._gray_generic[frame.size]
        log2a, mass = from_attention(attn)
        result = wherebits(log2a, mass, self.prior, weighting="bits" if self.passes == 3 else "mass")
        return Explanation(image=image, frame=frame, grid=tuple(grid), result=result, log2a=log2a, mass=mass)
