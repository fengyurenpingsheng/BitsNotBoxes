"""Coding cost of image regions from a learned hyperprior codec (CompressAI), for the bit axis of the faithfulness
evaluation. Needs torch and compressai.

The codec's entropy model gives the bits spent at every 16 x 16 latent cell, B[h, w] = sum_c -log2 p(y_hat | z_hat).
Spread uniformly over the cell's pixels and summed inside each visual-token cell, these bits add up to the image's
coded size (side information excluded).
"""
from __future__ import annotations

import math

LATENT_STRIDE = 16   # the analysis transform downsamples by 2^4
PAD_MULTIPLE = 64    # the hyper-analysis adds 2^2 more


def load_codec(name: str = "mbt2018-mean", quality: int = 4, device: str = "cpu"):
    """A pretrained CompressAI codec ("mbt2018-mean" or "bmshj2018-hyperprior")."""
    from compressai.zoo import bmshj2018_hyperprior, mbt2018_mean
    factories = {"bmshj2018-hyperprior": bmshj2018_hyperprior, "mbt2018-mean": mbt2018_mean}
    if name not in factories:
        raise ValueError(f"codec must be one of {sorted(factories)}")
    net = factories[name](quality=quality, pretrained=True).eval().to(device)
    net.update(force=True)
    for p in net.parameters():
        p.requires_grad_(False)
    net._wherebits_name = name
    return net


def pil_to_tensor(image):
    import numpy as np
    import torch
    arr = torch.from_numpy(np.asarray(image.convert("RGB"), dtype="uint8").copy())
    return arr.permute(2, 0, 1).float().div(255.)[None]          # [1, 3, H, W] in [0, 1]


def _bits_map(codec, x):
    """Bits of the latents at every latent cell of the padded image: [h, w], and the side-information bits."""
    import torch
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    H = math.ceil(h / PAD_MULTIPLE) * PAD_MULTIPLE
    W = math.ceil(w / PAD_MULTIPLE) * PAD_MULTIPLE
    xp = F.pad(x, (0, W - w, 0, H - h), mode="replicate")
    with torch.no_grad():
        y = codec.g_a(xp)
        if codec._wherebits_name == "bmshj2018-hyperprior":
            z = codec.h_a(torch.abs(y))
            z_hat, z_lik = codec.entropy_bottleneck(z)
            _, y_lik = codec.gaussian_conditional(y, codec.h_s(z_hat))
        else:
            z = codec.h_a(y)
            z_hat, z_lik = codec.entropy_bottleneck(z)
            scales, means = codec.h_s(z_hat).chunk(2, 1)
            _, y_lik = codec.gaussian_conditional(y, scales, means=means)
    return (-torch.log2(y_lik)).sum(1)[0], float((-torch.log2(z_lik)).sum())


def latent_to_token(values, grid: tuple, image_hw: tuple):
    """Spread latent-cell bits [h, w] over their pixels and sum them inside each visual-token cell: [rows, cols]."""
    import torch.nn.functional as F
    H, W = image_hw
    per_pixel = values[None, None] / float(LATENT_STRIDE ** 2)
    pix = F.interpolate(per_pixel, scale_factor=LATENT_STRIDE, mode="nearest")[..., :H, :W]
    gh, gw = grid
    return F.adaptive_avg_pool2d(pix, (gh, gw))[0, 0] * ((H / gh) * (W / gw))


def token_bits(codec, image, grid: tuple):
    """Bits the codec spends on each visual-token cell of ``image`` (the model frame): ([rows, cols], bpp)."""
    x = pil_to_tensor(image)
    bits_map, z_bits = _bits_map(codec, x)
    bits = latent_to_token(bits_map, grid, tuple(x.shape[-2:]))
    return bits, (float(bits_map.sum()) + z_bits) / (x.shape[-1] * x.shape[-2])
