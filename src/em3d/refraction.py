"""Refractive-index / permittivity profiles on the grid."""
from __future__ import annotations

from typing import Literal, Tuple

from .grid import Grid


def _blank(grid: Grid):
    be = grid.backend
    return be.zeros(grid.N, kind="complex")


def cylinder_refraction(
    grid: Grid,
    *,
    eps_real: float,
    eps_imag: float,
    radius: float,
    axis: Literal["x", "y", "z"] = "z",
) -> object:
    """Infinite cylinder along `axis`, radius in grid length units."""
    be = grid.backend
    X, Y, Z = grid.coords()
    if axis == "z":
        r2 = X * X + Y * Y
    elif axis == "y":
        r2 = X * X + Z * Z
    elif axis == "x":
        r2 = Y * Y + Z * Z
    else:
        raise ValueError(f"axis must be 'x'|'y'|'z', got {axis!r}")
    eta_value = complex(eps_real - 1.0, eps_imag)
    out = _blank(grid)
    mask = r2 <= radius * radius
    out = be.xp.where(mask, be.xp.asarray(eta_value, dtype=be.complex_dtype), out)
    return out


def step_refraction(
    grid: Grid,
    *,
    eps_real: float,
    eps_imag: float,
    z_min: float,
    z_max: float,
) -> object:
    """Slab between z_min and z_max (inclusive)."""
    be = grid.backend
    _, _, Z = grid.coords()
    eta_value = complex(eps_real - 1.0, eps_imag)
    mask = (Z >= z_min) & (Z <= z_max)
    out = _blank(grid)
    out = be.xp.where(mask, be.xp.asarray(eta_value, dtype=be.complex_dtype), out)
    return out


def ellipsis_refraction(
    grid: Grid,
    *,
    eps_real: float,
    eps_imag: float,
    center: Tuple[float, float, float],
    radius: Tuple[float, float, float],
) -> object:
    """Axis-aligned ellipsoid."""
    be = grid.backend
    X, Y, Z = grid.coords()
    cx, cy, cz = center
    rx, ry, rz = radius
    metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    eta_value = complex(eps_real - 1.0, eps_imag)
    out = _blank(grid)
    mask = metric <= 1.0
    out = be.xp.where(mask, be.xp.asarray(eta_value, dtype=be.complex_dtype), out)
    return out


def apply_refraction(grid: Grid, *, scalar_eta=None, tensor_eta=None) -> object:
    """Return a (3, 3, Nx, Ny, Nz) complex tensor."""
    be = grid.backend
    if (scalar_eta is None) == (tensor_eta is None):
        raise ValueError("apply_refraction requires exactly one of scalar_eta or tensor_eta")
    if scalar_eta is not None:
        if scalar_eta.shape != grid.N:
            raise ValueError(f"scalar_eta.shape {scalar_eta.shape} != grid.N {grid.N}")
        out = be.zeros((3, 3) + grid.N, kind="complex")
        for i in range(3):
            out[i, i] = scalar_eta
        return out
    if tensor_eta.shape != (3, 3) + grid.N:
        raise ValueError(
            f"tensor_eta.shape {tensor_eta.shape} != {(3, 3) + grid.N}"
        )
    return tensor_eta.astype(be.complex_dtype, copy=False)
