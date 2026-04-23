"""Plane wave sampled on a grid."""
from __future__ import annotations

from typing import Tuple

from .grid import Grid


def flat_wave_vec(
    grid: Grid,
    *,
    k: float,
    orient: Tuple[float, float, float],
    amplitude: Tuple[float, float, float],
    phi0: float = 0.0,
    sign: int = 1,
) -> object:
    """Return plane wave field A · exp(i sign·(k·(orient·r) + phi0)) on the grid.

    Shape: (3, Nx, Ny, Nz), dtype = backend.complex_dtype.
    `orient` must be a unit vector.
    """
    be = grid.backend
    xp = be.xp
    norm2 = orient[0] ** 2 + orient[1] ** 2 + orient[2] ** 2
    if abs(norm2 - 1.0) > 1e-9:
        raise ValueError(f"orient must be a unit vector, got norm² = {norm2}")
    X, Y, Z = grid.coords()
    phase = sign * (k * (orient[0] * X + orient[1] * Y + orient[2] * Z) + phi0)
    carrier = xp.exp(1j * phase).astype(be.complex_dtype, copy=False)
    out = be.zeros((3,) + grid.N, kind="complex")
    for i, a in enumerate(amplitude):
        if a != 0.0:
            out[i] = be.complex_dtype(a) * carrier
    return out
