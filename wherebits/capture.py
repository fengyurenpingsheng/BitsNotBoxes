"""Capture the answer-start attention row of every layer with vlmprobe (needs torch).

The four input conditions pair the image or a uniform gray image of the same size with the question or the generic
prompt. Because the image precedes the text, the visual keys are the same tokens under all four conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

GENERIC_PROMPT = "Write a general description of the image."   # ViCrop's generic prompt
GRAY = (124, 124, 124)       # gray level of the content-free image (nonzero after every model's normalization)

# alternative generic prompts of the sensitivity study (supplement); "vicrop" is the default
GENERIC_PROMPTS = {
    "vicrop": GENERIC_PROMPT,
    "describe": "Describe this image.",
    "see": "What do you see in this image?",
    "caption": "Provide a short caption for the image.",
}


@dataclass
class Rows:
    """Attention of one query row (the answer start) over all keys, for every layer."""

    attn: "torch.Tensor"              # [L, H, T] float32
    visual: "torch.Tensor"            # [T] bool, visual key positions (row-major image order)
    query_index: int
    grad: "torch.Tensor | None" = None     # [L, H, T] d(target score)/d(attention) of the same row
    score: float | None = None

    @property
    def vis(self):                    # [L, H, N]
        return self.attn[..., self.visual]


def _query_index(layout, query) -> int:
    if isinstance(query, int):
        return query
    idx = layout.select(query)[0].nonzero()
    if idx.numel() == 0:
        raise ValueError(f"query selector {query!r} selected no position")
    return int(idx[-1])


def capture_rows(probe, inputs, layers, *, metric=None, query="last_prompt") -> Rows:
    """One forward pass keeping only the query row of every attention map in ``layers``.

    With a ``metric`` (e.g. ``TokenLogProb``), one backward pass also records the gradient of the target score with
    respect to the same attention row (stored in the dumps; WhereBits itself is gradient-free).
    """
    import torch
    from vlm_probing.adapters import CapabilityError
    from vlm_probing.api.common import evaluate
    sites = [probe.spec.attentions[i] for i in layers]
    if metric is None:
        with torch.no_grad():
            trace = probe._run(inputs, capture=sites)
            layout = probe._layout(inputs, trace.logits)
            q = _query_index(layout, query)
            attn = torch.stack([trace.activations[s][0, :, q, :].float() for s in sites]).cpu()
        del trace
        return Rows(attn, layout.visual[0].cpu(), q)
    with torch.enable_grad():
        edits = {}
        if probe.spec.embeddings is not None:
            edits[probe.spec.embeddings] = lambda x: x if x.requires_grad else x.detach().requires_grad_(True)
        trace = probe._run(inputs, capture=sites, interventions=edits, grad=True)
        layout = probe._layout(inputs, trace.logits)
        q = _query_index(layout, query)
        mats = [trace.activations[s] for s in sites]
        score = evaluate(metric, trace.logits, layout)
        if not score.requires_grad:
            raise CapabilityError("target score is not differentiable with respect to attention")
        grads = torch.autograd.grad(score.sum(), mats, allow_unused=True)
        attn = torch.stack([m[0, :, q, :].detach().float() for m in mats]).cpu()
        grad = torch.stack([(g[0, :, q, :].detach().float() if g is not None
                             else torch.zeros_like(m[0, :, q, :], dtype=torch.float32))
                            for g, m in zip(grads, mats)]).cpu()
        value = float(score.detach().float().sum())
    del trace, mats, grads
    return Rows(attn, layout.visual[0].cpu(), q, grad, value)


def visual_attention(rows: Rows):
    """Visual attention rows as a float64 numpy array [L, H, N]."""
    return rows.vis.clamp_min(0).double().numpy()
