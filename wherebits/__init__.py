"""WhereBits: label-free layer selection for attention maps of vision-language models.

Code of "Bits, Not Boxes: Label-Free Layer Selection for Visual Localization in VLMs".

The estimator (``wherebits.core``), the crop rules (``wherebits.crop``) and the offline analysis of attention dumps
(``wherebits.offline``) need only numpy and scipy. Running a model (``WhereBitsVLM``, the dump and evaluation
scripts) also needs torch, transformers and vlmprobe.
"""
from .core import (CONDITIONS, GAMMA, WhereBitsResult, from_attention, head_weights, layer_maps, layer_prior,
                   token_scores, where_bits, wherebits, whether_bits)
from .crop import best_window, bits_window, crop_original
from .prior import available_priors, load_prior

__version__ = "1.0.0"

__all__ = ["CONDITIONS", "GAMMA", "WhereBitsResult", "from_attention", "head_weights", "layer_maps", "layer_prior",
           "token_scores", "where_bits", "wherebits", "whether_bits", "best_window", "bits_window", "crop_original",
           "available_priors", "load_prior", "WhereBitsVLM"]


def __getattr__(name):
    if name == "WhereBitsVLM":                 # imported lazily: needs torch, transformers and vlmprobe
        from .pipeline import WhereBitsVLM
        return WhereBitsVLM
    raise AttributeError(name)
