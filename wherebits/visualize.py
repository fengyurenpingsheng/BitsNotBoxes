"""Overlay a map and a crop window on the model frame (needs matplotlib)."""
from __future__ import annotations

import numpy as np


def show_map(frame, map2d, window=None, *, title: str = "", path=None, positive_only: bool = True):
    """Draw ``map2d`` ([rows, cols]) over ``frame`` (PIL); ``window`` = (i, j, h, w) in tokens. Saves to ``path``."""
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    m = np.asarray(map2d, np.float64)
    if positive_only:
        m = np.clip(m, 0, None)
    gh, gw = m.shape
    W, H = frame.size
    fig, ax = plt.subplots(figsize=(5, 5 * H / W))
    ax.imshow(frame)
    ax.imshow(m, cmap="inferno", alpha=0.55, extent=(0, W, H, 0), interpolation="nearest")
    if window is not None:
        i, j, wh, ww = window
        ax.add_patch(Rectangle((j * W / gw, i * H / gh), ww * W / gw, wh * H / gh, fill=False, lw=2.5, ls="--",
                               edgecolor="cyan"))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
