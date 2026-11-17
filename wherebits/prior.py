"""Layer priors: loading the shipped priors and writing new ones.

A prior file stores, for one model and input size, the mean layer bits Bbar_l over the unlabeled COCO calibration
questions and the normalized prior pi_l = Bbar_l^gamma / sum_j Bbar_j^gamma (gamma = 4).
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from .core import GAMMA, layer_prior


def available_priors() -> list[str]:
    """Model keys with a shipped prior (``wherebits/priors/<key>.json``)."""
    folder = resources.files("wherebits") / "priors"
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


def prior_data(key_or_path) -> dict:
    """Contents of a prior file: a shipped model key or a JSON path."""
    p = Path(str(key_or_path))
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text())
    f = resources.files("wherebits") / "priors" / f"{key_or_path}.json"
    if not f.is_file():
        raise FileNotFoundError(f"no shipped prior for {key_or_path!r}; available: {available_priors()}. "
                                "Calibrate one with scripts/calibrate_prior.py")
    return json.loads(f.read_text())


def load_prior(key_or_path, gamma: float | None = None) -> np.ndarray:
    """Prior pi [L] from a shipped model key (e.g. "qwen3-vl-2b") or a JSON file written by ``save_prior``.

    With ``gamma``, the prior is recomputed from the stored mean layer bits with that exponent.
    """
    data = prior_data(key_or_path)
    if gamma is not None and float(gamma) != float(data["gamma"]):
        return layer_prior(data["layer_bits"], gamma)
    return np.asarray(data["prior"], np.float64)


def save_prior(path, model: str, layer_bits_mean, *, n_questions: int, source: str, gamma: float = GAMMA,
               extra: dict | None = None) -> dict:
    """Write a prior JSON file (mean layer bits, gamma and the normalized prior)."""
    bits = np.asarray(layer_bits_mean, np.float64)
    data = {"model": model, "gamma": gamma, "n_layers": int(bits.size), "n_questions": int(n_questions),
            "source": source, "layer_bits": [float(x) for x in bits],
            "prior": [float(x) for x in layer_prior(bits, gamma)], **(extra or {})}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n")
    return data
