"""em3d.vis — visualization utilities for RCS and EM fields.

Public API
----------
plot_rcs(phi, sigma, *, title=None, filename=None) -> (fig, ax)
plot_rcs_polar(phi, sigma, *, db=False, title=None, filename=None) -> (fig, ax)
"""
from __future__ import annotations

import numpy as np

__all__ = ["plot_rcs", "plot_rcs_polar"]


def _require_matplotlib():
    """Return matplotlib.pyplot, or raise ImportError with install hint."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install em3d[vis]"
        ) from None


def plot_rcs(
    phi,
    sigma,
    *,
    title: str | None = None,
    filename: str | None = None,
) -> tuple:
    """Cartesian sigma(phi) plot.

    Parameters
    ----------
    phi      : array (n,) float — angles in radians [0, 2pi)
    sigma    : array (n,) float — RCS values >= 0
    title    : optional figure title
    filename : if given, save to this path at dpi=150

    Returns
    -------
    (fig, ax) — matplotlib Figure and Axes
    """
    plt = _require_matplotlib()
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fig, ax = plt.subplots()
    ax.plot(phi, sigma)
    ax.set_xlabel("phi (rad)")
    ax.set_ylabel("sigma")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_rcs_polar(
    phi,
    sigma,
    *,
    db: bool = False,
    title: str | None = None,
    filename: str | None = None,
) -> tuple:
    """Polar sigma(phi) plot, linear or dB scale.

    Parameters
    ----------
    phi      : array (n,) float — angles in radians
    sigma    : array (n,) float — RCS values >= 0
    db       : if False (default) plot sigma as radius;
               if True plot 10*log10(sigma/sigma_max) — relative dB, max = 0 dB
    title    : optional figure title
    filename : if given, save to this path at dpi=150

    Returns
    -------
    (fig, ax) — matplotlib Figure and polar Axes
    """
    plt = _require_matplotlib()
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    if db:
        r = 10.0 * np.log10(sigma / sigma.max() + 1e-30)
        ax.plot(phi, r)
        ax.set_ylabel("sigma (dB rel. max)")
    else:
        ax.plot(phi, sigma)
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax
