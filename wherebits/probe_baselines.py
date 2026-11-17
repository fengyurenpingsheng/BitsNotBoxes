"""Gradient-based and propagation baselines computed with vlmprobe at dump time (needs torch).

relevance   Chefer et al. (2021) generic attention-model explainability, target = the model's first answer token
grad_cam    attention Grad-CAM at the last layer
rollout     attention rollout
tokenact    token activation map: the target token's logit read out at every visual position through the final norm
            and LM head (base map of Li et al., 2025, re-implemented)
All maps are returned as [N] vectors in visual-token order.
"""
from __future__ import annotations

import time

METHODS = ("grad_cam", "relevance", "rollout")


def vlmprobe_baselines(probe, inputs, metric, *, methods=METHODS, query="last_prompt"):
    from vlm_probing.adapters import CapabilityError
    layout = probe._layout(inputs, inputs.kwargs["input_ids"])
    qmask, kmask = layout.select(query), layout.visual
    q = int(qmask[0].nonzero()[-1])
    last = max(probe.spec.attentions)
    out, timing, skipped = {}, {}, {}

    def row(t):                       # [B, Q, K] map -> [N]
        return t[0, q][kmask[0].to(t.device)].float().cpu()

    jobs = {
        "grad_cam": lambda: row(probe.attention.grad_cam(layers=[last], queries=query, keys="visual")
                                .run(inputs, metric=metric).tensors["cam"][0]),
        "relevance": lambda: row(probe.attention.relevance().run(inputs, metric=metric).tensors["relevance"]),
        "rollout": lambda: row(probe.attention.rollout().run(inputs).tensors["rollout"]),
    }
    for name in methods:
        t0 = time.time()
        try:
            out[name] = jobs[name]()
            timing[name] = time.time() - t0
        except CapabilityError as e:
            skipped[name] = str(e)
    return out, timing, skipped


def token_activation_map(probe, inputs, target: int):
    """The target token's logit at every visual position (logit lens at the last residual site): [N]."""
    import torch
    last = max(probe.spec.residuals)
    with torch.no_grad():
        res = probe.lens.logit(layers=[last], tokens="visual").run(inputs)
    logits = res.tensors["logits"]
    logits = logits.reshape(-1, logits.shape[-1]) if logits.ndim > 2 else logits
    return logits[:, target].float().cpu()


def visual_hidden_norms(probe, inputs):
    """L2 norm of every visual token's residual stream at every residual site: [S, N]."""
    import torch
    sites = [probe.spec.residuals[i] for i in sorted(probe.spec.residuals)]
    with torch.no_grad():
        trace = probe._run(inputs, capture=sites)
        layout = probe._layout(inputs, trace.logits)
        vis = layout.visual[0]
        norms = []
        for s in sites:
            h = trace.activations[s]
            h = h[0] if h.ndim == 3 else h
            norms.append(h[vis.to(h.device)].float().norm(dim=-1).cpu())
    del trace
    return torch.stack(norms)
