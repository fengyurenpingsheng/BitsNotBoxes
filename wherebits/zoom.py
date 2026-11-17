"""Training-free zoom-in (ViCrop protocol): crop the window a map selects from the original image, give the model the
image and the crop (the crop alone for LLaVA-1.5), and answer again. Needs torch.

TextVQA answers are generated greedily (at most 10 new tokens). V* is scored by forced choice: one forward pass, and
the option letter with the highest next-token probability is the prediction.
"""
from __future__ import annotations

from .data import TEXTVQA_SUFFIX


def _cast(inputs: dict, dtype):
    import torch
    return {k: (v.to(dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in inputs.items()}


def _model_inputs(processor, spec, question: str, images: list, dtype, device=None):
    """Processor inputs for one or more images placed before the text. Yields both image-list conventions
    (flat, then nested), because processors differ in which one they accept for several images."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
    attempts = [images] if len(images) == 1 else [images, [images]]
    for imgs in attempts:
        def make(imgs=imgs):
            import torch
            inputs = _cast(dict(processor(text=prompt, images=imgs, return_tensors="pt", **spec.call_kwargs)), dtype)
            if device is not None:
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            return inputs
        yield make


def answer(model, processor, spec, question: str, images: list, *, max_new_tokens: int = 10,
           suffix: str = TEXTVQA_SUFFIX, device=None) -> str:
    """Greedy answer with one or more images."""
    import torch
    dtype = next(model.parameters()).dtype
    last_error = None
    for make in _model_inputs(processor, spec, question + suffix, images, dtype, device):
        try:
            inputs = make()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new = out[0, inputs["input_ids"].shape[1]:]
            return processor.tokenizer.decode(new, skip_special_tokens=True).strip()
        except Exception as e:  # noqa: BLE001 - try the other image-list convention
            last_error = e
    raise last_error


def letter_token_ids(tokenizer) -> dict:
    """Token ids that spell each option letter as a whole token ("A", " A", "▁A", "ĠA")."""
    vocab = tokenizer.get_vocab()
    out = {}
    for L in "ABCDEFGH":
        ids = {vocab[t] for t in (L, "▁" + L, "Ġ" + L, " " + L) if t in vocab}
        for s in (L, " " + L):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if enc and tokenizer.decode([enc[0]]).strip() == L:
                ids.add(enc[0])
        if not ids:
            raise RuntimeError(f"no single token spells the option letter {L!r}")
        out[L] = sorted(ids)
    seen = {}
    for L, ids in out.items():
        for i in ids:
            if i in seen:
                raise RuntimeError(f"token {i} spells both {seen[i]} and {L}")
            seen[i] = L
    return out


def forced_choice(model, processor, spec, question: str, images: list, ids: dict, device=None) -> dict:
    """Log-probability of each option letter at the answer position (one forward pass); "pred" is the argmax."""
    import torch
    dtype = next(model.parameters()).dtype
    last_error = None
    for make in _model_inputs(processor, spec, question, images, dtype, device):
        try:
            inputs = make()
            with torch.no_grad():
                logp = torch.log_softmax(model(**inputs).logits[0, -1].float(), dim=-1)
            lp = {L: float(torch.logsumexp(logp[ii], 0)) for L, ii in ids.items()}
            all_ids = [i for ii in ids.values() for i in ii]
            return {"lp": lp, "pred": max(lp, key=lp.get),
                    "mass": float(torch.logsumexp(logp[all_ids], 0).exp()),
                    "top1_letter": int(logp.argmax()) in set(all_ids)}
        except Exception as e:  # noqa: BLE001 - try the other image-list convention
            last_error = e
    raise last_error
