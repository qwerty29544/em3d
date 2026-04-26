"""em3d.vis — visualization utilities for RCS and EM fields.

Public API
----------
plot_rcs(phi, sigma, *, title=None, filename=None) -> (fig, ax)
plot_rcs_polar(phi, sigma, *, db=False, title=None, filename=None) -> (fig, ax)
plot_field_slice(u, grid, *, plane="xy", idx=None, part="real", stride=1, cmap="RdBu_r", title=None, filename=None) -> (fig, ax)
plot_field_volume(u, grid, *, part="real", stride=2, elev=30.0, azim=-60.0, cmap="RdBu_r", title=None, filename=None) -> (fig, ax)
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["plot_rcs", "plot_rcs_polar", "plot_field_slice", "plot_field_volume"]


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


# ── Private helpers for field visualization ───────────────────────────────

def _validate_u(u) -> np.ndarray:
    """Assert u has shape (3, Nx, Ny, Nz) and return as numpy array."""
    u = np.asarray(u)
    if u.ndim != 4:
        raise ValueError(
            f"u must have shape (3, Nx, Ny, Nz), got ndim={u.ndim}"
        )
    if u.shape[0] != 3:
        raise ValueError(
            f"u must have shape (3, Nx, Ny, Nz), got shape={u.shape}"
        )
    return u


def _extract_part(u: np.ndarray, part: str) -> np.ndarray:
    """Return real-valued (3, Nx, Ny, Nz) array for the chosen part."""
    if part == "real":
        return u.real.astype(np.float64)
    if part == "imag":
        return u.imag.astype(np.float64)
    if part == "abs":
        return np.abs(u).astype(np.float64)
    raise ValueError(f"part must be 'real', 'imag', or 'abs', got {part!r}")


# ── Public field visualization ────────────────────────────────────────────

def plot_field_slice(
    u,
    grid,
    *,
    plane: str = "xy",
    idx: int | None = None,
    part: str = "real",
    stride: int = 1,
    cmap: str = "RdBu_r",
    title: str | None = None,
    filename: str | None = None,
) -> tuple:
    """2-D cross-section of the complex vector field u.

    Parameters
    ----------
    u        : array (3, Nx, Ny, Nz) complex — field from solver result.u
    grid     : em3d.Grid — must expose .coords(), .dv, and .N
    plane    : "xy" | "xz" | "yz" — orientation of the slice
    idx      : index along the normal axis; None uses N//2 (middle)
    part     : "real" | "imag" | "abs" — which part of u to display
    stride   : show every stride-th grid point per in-plane axis (default 1)
    cmap     : matplotlib colormap name (default "RdBu_r")
    title    : optional figure title
    filename : if given, save figure to this path at dpi=150

    Returns
    -------
    (fig, ax) — matplotlib Figure and 2-D Axes
    """
    plt = _require_matplotlib()
    import matplotlib.colors as mcolors

    u = _validate_u(u)
    if plane not in ("xy", "xz", "yz"):
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    F = _extract_part(u, part)          # (3, Nx, Ny, Nz), float64

    # Select slice axis, 1-D coordinate arrays, and in-plane F components
    if plane == "xy":
        axis_len = F.shape[3]           # Nz
        if idx is None:
            idx = axis_len // 2
        if idx < 0 or idx >= axis_len:
            raise ValueError(f"idx {idx} out of range [0, {axis_len})")
        horiz = np.asarray(grid.x)      # (Nx,)
        vert  = np.asarray(grid.y)      # (Ny,)
        F2d   = F[:, :, :, idx]         # (3, Nx, Ny)
        hlabel, vlabel = "x", "y"
    elif plane == "xz":
        axis_len = F.shape[2]           # Ny
        if idx is None:
            idx = axis_len // 2
        if idx < 0 or idx >= axis_len:
            raise ValueError(f"idx {idx} out of range [0, {axis_len})")
        horiz = np.asarray(grid.x)      # (Nx,)
        vert  = np.asarray(grid.z)      # (Nz,)
        F2d   = F[:, :, idx, :]         # (3, Nx, Nz)
        hlabel, vlabel = "x", "z"
    else:  # yz
        axis_len = F.shape[1]           # Nx
        if idx is None:
            idx = axis_len // 2
        if idx < 0 or idx >= axis_len:
            raise ValueError(f"idx {idx} out of range [0, {axis_len})")
        horiz = np.asarray(grid.y)      # (Ny,)
        vert  = np.asarray(grid.z)      # (Nz,)
        F2d   = F[:, idx, :, :]         # (3, Ny, Nz)
        hlabel, vlabel = "y", "z"

    # Background: Full 3-D field magnitude at the slice (all three components including out-of-plane).
    norm2d = np.sqrt(F2d[0] ** 2 + F2d[1] ** 2 + F2d[2] ** 2)  # (Nh, Nv)
    vmax = float(norm2d.max()) if norm2d.max() > 0 else 1.0

    fig, ax = plt.subplots()

    # pcolormesh background
    H, V = np.meshgrid(horiz, vert, indexing="ij")
    pcm = ax.pcolormesh(H, V, norm2d, cmap=cmap, vmin=0.0, vmax=vmax,
                        shading="auto")
    fig.colorbar(pcm, ax=ax)

    # Quiver arrows with stride decimation
    s = stride
    Hs  = horiz[::s]
    Vs  = vert[::s]
    if plane == "xy":
        U2d = F2d[0, ::s, ::s]         # x-component
        V2d = F2d[1, ::s, ::s]         # y-component
    elif plane == "xz":
        U2d = F2d[0, ::s, ::s]         # x-component
        V2d = F2d[2, ::s, ::s]         # z-component
    else:  # yz
        U2d = F2d[1, ::s, ::s]         # y-component
        V2d = F2d[2, ::s, ::s]         # z-component

    F_max = float(np.sqrt(U2d ** 2 + V2d ** 2).max())
    if F_max > 0:
        # Use the geometric mean of the two in-plane cell spacings so that
        # arrow scale is correct on non-square grids (isotropic grid: same as dx).
        if plane == "xy":
            dh = float(np.sqrt((grid.L[0] / grid.N[0]) * (grid.L[1] / grid.N[1])))
        elif plane == "xz":
            dh = float(np.sqrt((grid.L[0] / grid.N[0]) * (grid.L[2] / grid.N[2])))
        else:  # yz
            dh = float(np.sqrt((grid.L[1] / grid.N[1]) * (grid.L[2] / grid.N[2])))
        scale = F_max / (stride * dh * 0.9)
        arrow_norm = np.sqrt(U2d ** 2 + V2d ** 2)
        Hg, Vg = np.meshgrid(Hs, Vs, indexing="ij")
        ax.quiver(
            Hg, Vg, U2d, V2d, arrow_norm,
            cmap=cmap,
            norm=mcolors.Normalize(vmin=0.0, vmax=F_max),
            scale=scale,
            scale_units="xy",
        )

    ax.set_xlabel(hlabel)
    ax.set_ylabel(vlabel)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_field_volume(
    u,
    grid,
    *,
    part: str = "real",
    stride: int = 2,
    elev: float = 30.0,
    azim: float = -60.0,
    cmap: str = "RdBu_r",
    title: str | None = None,
    filename: str | None = None,
) -> tuple:
    """3-D quiver plot of the complex vector field u.

    Parameters
    ----------
    u        : array (3, Nx, Ny, Nz) complex — field from solver result.u
    grid     : em3d.Grid — must expose .coords(), .dv, and .N
    part     : "real" | "imag" | "abs" — which part of u to display
    stride   : show every stride-th point per axis (default 2)
    elev     : elevation angle in degrees for 3-D view (default 30)
    azim     : azimuth angle in degrees for 3-D view (default -60)
    cmap     : matplotlib colormap name (default "RdBu_r")
    title    : optional figure title
    filename : if given, save figure to this path at dpi=150

    Returns
    -------
    (fig, ax) — matplotlib Figure and Axes3D

    Notes
    -----
    Arrow colours encode the local norm ‖(U, V, W)‖ through `cmap`.
    If the decimated grid has more than 2 000 arrows a UserWarning is emitted.

    Examples
    --------
    >>> fig, ax = plot_field_volume(u, grid, elev=45, azim=30)   # isometric
    >>> fig, ax = plot_field_volume(u, grid, elev=90, azim=0)    # top-down
    >>> fig, ax = plot_field_volume(u, grid, elev=0,  azim=0)    # front view
    """
    plt = _require_matplotlib()
    import matplotlib.colors as mcolors

    u = _validate_u(u)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    F = _extract_part(u, part)          # (3, Nx, Ny, Nz), float64; raises on bad part

    sl = (slice(None, None, stride),) * 3
    X, Y, Z = grid.coords()
    Xs = np.asarray(X[sl])
    Ys = np.asarray(Y[sl])
    Zs = np.asarray(Z[sl])
    U  = np.asarray(F[0][sl])
    V  = np.asarray(F[1][sl])
    W  = np.asarray(F[2][sl])

    if Xs.size == 0:
        warnings.warn(
            f"plot_field_volume: stride={stride} exceeds grid dimensions after decimation; "
            f"no arrows to draw",
            UserWarning,
            stacklevel=2,
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if title is not None:
            ax.set_title(title)
        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
        return fig, ax

    n_arrows = Xs.size
    if n_arrows > 2_000:
        warnings.warn(
            f"plot_field_volume: {n_arrows} arrows after decimation; "
            f"consider increasing stride",
            UserWarning,
            stacklevel=2,
        )

    # Arrow length: longest arrow spans 90% of one decimated cell
    dv13     = float(grid.dv ** (1.0 / 3.0))
    norm_max = float(np.sqrt(U ** 2 + V ** 2 + W ** 2).max())
    if norm_max > 0:
        length = stride * dv13 * 0.9 / norm_max
    else:
        length = stride * dv13

    # Per-arrow colour via local norm mapped through cmap
    arrow_norms = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=norm_max if norm_max > 0 else 1.0),
    )
    colors_rgba = sm.to_rgba(arrow_norms.ravel())   # (N, 4)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.quiver(
        Xs.ravel(), Ys.ravel(), Zs.ravel(),
        U.ravel(),  V.ravel(),  W.ravel(),
        length=length, normalize=False, colors=colors_rgba,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax
