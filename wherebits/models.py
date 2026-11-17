"""Model registry, loading, image frames and prompts.

Every model sees one image tile of a fixed size (or, for the native-resolution settings, a frame with the image's
aspect ratio and a token budget), so the visual-token grid is known exactly and maps, crops and ground truth share
one coordinate frame.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class ModelSpec:
    key: str
    model_id: str                # Hugging Face model id
    family: str
    image_size: int | None       # side of the square frame in pixels (None: native aspect ratio, see max_tokens)
    grid: tuple | None           # (rows, cols) of visual tokens (None: per image, from frame_and_grid)
    dtype: str = "float32"
    processor_kwargs: dict = field(default_factory=dict)      # AutoProcessor.from_pretrained kwargs
    call_kwargs: dict = field(default_factory=dict)           # per-call processor kwargs
    answer_tokens: int = 4       # tokens generated for the model's own answer (its first token is explained)
    note: str = ""
    max_tokens: int = 0          # native frames: cap on the number of visual tokens
    token_px: int = 32           # native frames: pixels per visual token side (patch x merge)

    @property
    def native(self) -> bool:
        return self.grid is None

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)


_QWEN_896 = {"size": {"shortest_edge": 4 * 32 * 32, "longest_edge": 896 * 896}}
_QWEN_NATIVE = {"size": {"shortest_edge": 4 * 32 * 32, "longest_edge": 1280 * 32 * 32}}

REGISTRY: dict[str, ModelSpec] = {
    "smolvlm-256m": ModelSpec("smolvlm-256m", "HuggingFaceTB/SmolVLM-256M-Instruct", "idefics3", 512, (8, 8),
                              call_kwargs={"do_image_splitting": False},
                              note="one 512 px tile, pixel shuffle x4: 8x8 = 64 tokens of 64 px"),
    "smolvlm-500m": ModelSpec("smolvlm-500m", "HuggingFaceTB/SmolVLM-500M-Instruct", "idefics3", 512, (8, 8),
                              call_kwargs={"do_image_splitting": False},
                              note="one 512 px tile, pixel shuffle x4: 8x8 = 64 tokens of 64 px"),
    "internvl3-1b": ModelSpec("internvl3-1b", "OpenGVLab/InternVL3-1B-hf", "internvl", 448, (16, 16),
                              call_kwargs={"crop_to_patches": False},
                              note="one 448 px tile, pixel unshuffle: 16x16 = 256 tokens of 28 px"),
    "qwen3-vl-2b": ModelSpec("qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", 448, (14, 14),
                             note="patch 16, merge 2: 14x14 = 196 tokens of 32 px"),
    "qwen3-vl-8b": ModelSpec("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", "qwen3_vl", 448, (14, 14),
                             note="same frame as qwen3-vl-2b; 36 layers x 32 heads"),
    "llava-1.5-7b": ModelSpec("llava-1.5-7b", "llava-hf/llava-1.5-7b-hf", "llava", 336, (24, 24),
                              processor_kwargs={"use_fast": False},
                              note="CLIP-L/14 at 336 px: 24x24 = 576 tokens of 14 px"),
    # held-out family: used only after the method was frozen
    "janus-pro-1b": ModelSpec("janus-pro-1b", "deepseek-community/Janus-Pro-1B", "janus", 384, (24, 24),
                              note="SigLIP-L/16 at 384 px: 24x24 = 576 tokens of 16 px"),
    "janus-pro-7b": ModelSpec("janus-pro-7b", "deepseek-community/Janus-Pro-7B", "janus", 384, (24, 24),
                              note="same frame as janus-pro-1b"),
    # higher input resolutions of Qwen3-VL (the processor's pixel range is set so that it keeps our frame)
    "qwen3-vl-2b-896": ModelSpec("qwen3-vl-2b-896", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", 896, (28, 28),
                                 call_kwargs=_QWEN_896, note="one 896 px tile: 28x28 = 784 tokens"),
    "qwen3-vl-8b-896": ModelSpec("qwen3-vl-8b-896", "Qwen/Qwen3-VL-8B-Instruct", "qwen3_vl", 896, (28, 28),
                                 call_kwargs=_QWEN_896, note="one 896 px tile: 28x28 = 784 tokens"),
    "qwen3-vl-2b-native": ModelSpec("qwen3-vl-2b-native", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", None, None,
                                    call_kwargs=_QWEN_NATIVE, max_tokens=1280, token_px=32,
                                    note="native aspect ratio, at most 1280 tokens"),
    "qwen3-vl-8b-native": ModelSpec("qwen3-vl-8b-native", "Qwen/Qwen3-VL-8B-Instruct", "qwen3_vl", None, None,
                                    call_kwargs=_QWEN_NATIVE, max_tokens=1280, token_px=32,
                                    note="native aspect ratio, at most 1280 tokens"),
}


def _env_threads():
    import torch
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "0")) or 0)
    if n > 0:
        torch.set_num_threads(n)


def load_model(spec: ModelSpec, device: str = "cpu", local_files_only: bool | None = None, dtype: str | None = None):
    """Load the model and processor with eager attention (attention probabilities are read from every layer).

    The paper's numbers were computed in float32; ``dtype`` (e.g. "bfloat16") overrides the spec on a GPU.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    _env_threads()
    if local_files_only is None:
        local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    torch_dtype = getattr(torch, dtype) if dtype else spec.torch_dtype
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id, dtype=torch_dtype, device_map={"": device}, attn_implementation="eager",
        local_files_only=local_files_only).eval()
    processor = AutoProcessor.from_pretrained(spec.model_id, **spec.processor_kwargs, local_files_only=local_files_only)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[models] loaded {spec.model_id} in {time.time() - t0:.1f}s on {device} ({torch_dtype})", flush=True)
    return model, processor


def square_image(image: Image.Image, size: int) -> Image.Image:
    """Resize (no crop) to the model's square frame."""
    return image.convert("RGB").resize((size, size), Image.BICUBIC)


def native_frame(image: Image.Image, *, token_px: int = 32, max_tokens: int = 1280, upscale: bool = False):
    """Resize to the native aspect ratio with both sides multiples of ``token_px`` and at most ``max_tokens`` tokens.

    ``upscale`` fills the token budget even for a small image, as a zoom-in crop should. Returns (frame, (rows, cols)).
    """
    W, H = image.size
    scale = (max_tokens * token_px ** 2 / float(W * H)) ** 0.5
    if not upscale:
        scale = min(1.0, scale)
    w = max(token_px, int(round(W * scale / token_px)) * token_px)
    h = max(token_px, int(round(H * scale / token_px)) * token_px)
    while (w // token_px) * (h // token_px) > max_tokens:          # rounding can overshoot by one row or column
        if w >= h:
            w -= token_px
        else:
            h -= token_px
    frame = image.convert("RGB").resize((w, h), Image.BICUBIC)
    return frame, (h // token_px, w // token_px)


def frame_and_grid(spec: ModelSpec, image: Image.Image, *, upscale: bool = False):
    """The model frame of ``image`` and its visual-token grid (rows, cols)."""
    if spec.native:
        return native_frame(image, token_px=spec.token_px, max_tokens=spec.max_tokens, upscale=upscale)
    return square_image(image, spec.image_size), tuple(spec.grid)


def build_prompt(processor, question: str) -> str:
    """Chat prompt with the image before the text."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare(probe, spec: ModelSpec, prompt: str, image: Image.Image, device="cpu", dtype=None):
    """vlmprobe inputs with floating-point tensors cast to the model dtype."""
    import torch
    inputs = probe.prepare(prompt=prompt, image=image, device=device, **spec.call_kwargs)
    target = dtype or next(probe.model.parameters()).dtype
    for key, value in list(inputs.kwargs.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs.kwargs[key] = value.to(dtype=target)
    return inputs


def check_grid(probe, inputs, spec: ModelSpec, grid: tuple | None = None):
    """Check that the processor produced exactly the expected number of visual tokens."""
    grid = tuple(grid) if grid is not None else spec.grid
    if grid is None:
        raise ValueError("a native-resolution spec needs the grid from frame_and_grid")
    expected = grid[0] * grid[1]
    layout = probe._layout(inputs, inputs.kwargs["input_ids"]) if "input_ids" in inputs.kwargs else None
    if layout is None or layout.visual is None:
        return None
    n = int(layout.visual.sum())
    if n != expected:
        raise RuntimeError(f"{spec.key}: processor produced {n} visual tokens, expected {expected} = {grid}; "
                           f"check the frame size and call_kwargs {spec.call_kwargs}")
    return layout


def greedy_answer(model, processor, inputs, max_new_tokens=4):
    """The model's own greedy answer: (token ids, text)."""
    import torch
    with torch.no_grad():
        out = model.generate(**dict(inputs.kwargs), max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, inputs.kwargs["input_ids"].shape[1]:]
    return [int(t) for t in new.tolist()], processor.tokenizer.decode(new, skip_special_tokens=True)


class TokenLogProb:
    """log p(target | prompt) at the last prompt position (a vlmprobe metric: ``score(logits, layout) -> [B]``)."""

    def __init__(self, target: int, *, kind="logprob", position="last_prompt"):
        self.target, self.kind, self.position = int(target), kind, position

    def score(self, logits, layout):
        import torch.nn.functional as F
        mask = layout.select(self.position)
        if not (mask.sum(-1) == 1).all():
            raise ValueError("TokenLogProb needs exactly one prediction position per example")
        row = logits[mask].float()
        if self.kind == "logit":
            return row[:, self.target]
        logp = F.log_softmax(row, dim=-1)[:, self.target]
        return logp if self.kind == "logprob" else logp.exp()


def answer_distribution(probe, inputs):
    """Next-token log-distribution at the last prompt position: [V]."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        trace = probe._run(inputs)
    layout = probe._layout(inputs, trace.logits)
    mask = layout.select("last_prompt")
    return F.log_softmax(trace.logits[mask].float(), dim=-1)[0]
