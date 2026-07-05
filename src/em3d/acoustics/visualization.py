"""Visualization helpers for scalar acoustic fields."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..vis import _require_matplotlib


def _field_part(u, part: str) -> np.ndarray:
    arr = np.asarray(u)
    if arr.ndim != 3:
        raise ValueError(f"u must have shape (Nx, Ny, Nz), got {arr.shape}")
    if part == "abs":
        return np.abs(arr).astype(np.float64)
    if part == "real":
        return arr.real.astype(np.float64)
    if part == "imag":
        return arr.imag.astype(np.float64)
    if part == "angle":
        return np.angle(arr).astype(np.float64)
    raise ValueError(f"part must be 'abs', 'real', 'imag', or 'angle', got {part!r}")


def _axis_to_host(grid, axis):
    return np.asarray(grid.backend.to_host(axis), dtype=np.float64)


def _slice_scalar(values: np.ndarray, grid, plane: str, idx: int | None):
    if plane == "xy":
        idx = values.shape[2] // 2 if idx is None else int(idx)
        if idx < 0 or idx >= values.shape[2]:
            raise ValueError(f"idx {idx} out of range [0, {values.shape[2]})")
        return _axis_to_host(grid, grid.x), _axis_to_host(grid, grid.y), values[:, :, idx], "x", "y"
    if plane == "xz":
        idx = values.shape[1] // 2 if idx is None else int(idx)
        if idx < 0 or idx >= values.shape[1]:
            raise ValueError(f"idx {idx} out of range [0, {values.shape[1]})")
        return _axis_to_host(grid, grid.x), _axis_to_host(grid, grid.z), values[:, idx, :], "x", "z"
    if plane == "yz":
        idx = values.shape[0] // 2 if idx is None else int(idx)
        if idx < 0 or idx >= values.shape[0]:
            raise ValueError(f"idx {idx} out of range [0, {values.shape[0]})")
        return _axis_to_host(grid, grid.y), _axis_to_host(grid, grid.z), values[idx, :, :], "y", "z"
    raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")


def plot_scalar_slice(
    u,
    grid,
    *,
    plane: str = "xy",
    idx: int | None = None,
    part: str = "abs",
    cmap: str = "viridis",
    title: str | None = None,
    filename: str | Path | None = None,
) -> tuple:
    """Plot one scalar acoustic field slice."""
    plt = _require_matplotlib()
    values = _field_part(u, part)
    horiz, vert, scalar, hlabel, vlabel = _slice_scalar(values, grid, plane, idx)
    H, V = np.meshgrid(horiz, vert, indexing="ij")
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(H, V, scalar, cmap=cmap, shading="auto")
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(hlabel)
    ax.set_ylabel(vlabel)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(str(filename), dpi=150, bbox_inches="tight")
    return fig, ax


def plot_scalar_slices(
    u,
    grid,
    *,
    output_dir: str | Path,
    prefix: str = "acoustic",
    parts: tuple[str, ...] = ("abs", "real", "imag", "angle"),
) -> list[Path]:
    """Save xy/xz/yz scalar slices for selected field parts."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for part in parts:
        for plane in ("xy", "xz", "yz"):
            path = root / f"{prefix}_{part}_{plane}.png"
            plot_scalar_slice(
                u,
                grid,
                plane=plane,
                part=part,
                title=f"{prefix} {part} {plane}",
                filename=path,
            )
            written.append(path)
    return written


def plot_pattern(
    phi,
    sigma,
    *,
    filename: str | Path | None = None,
    polar: bool = False,
    title: str | None = None,
) -> tuple:
    """Plot acoustic scattering pattern in Cartesian or polar axes."""
    plt = _require_matplotlib()
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if phi.shape != sigma.shape:
        raise ValueError(f"phi and sigma must have the same shape, got {phi.shape} and {sigma.shape}")
    if polar:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel("phi (rad)")
        ax.set_ylabel("normalized pattern")
    ax.plot(phi, sigma)
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(str(filename), dpi=150, bbox_inches="tight")
    return fig, ax
