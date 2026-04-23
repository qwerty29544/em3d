"""FFT-accelerated volume-integral operator on a doubled parallelepiped Π₂."""
from __future__ import annotations

import numpy as np

from .backend import Backend
from .grid import Grid


def _kernel_tensor_on_doubled_grid(grid: Grid, k: float, volume: float):
    """Sample the 3×3 kernel on the doubled grid Π₂ (2Nx × 2Ny × 2Nz cells).

    The (a, b) block is an isotropic scalar kernel ⋅ δ_{ab} in this minimal version.
    Anisotropic refinements can hook in here; they are not needed for the FFT-vs-dense
    integration test because the dense matrix is assembled consistently.
    """
    be = grid.backend
    xp = be.xp
    Nx, Ny, Nz = grid.N
    Lx, Ly, Lz = grid.L
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    # separations on Π₂: [0, dx, 2dx, ..., (N-1)dx, -N·dx, -(N-1)dx, ..., -dx]  — typical periodisation
    sx = xp.concatenate([xp.arange(Nx) * dx, -(xp.arange(Nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(Ny) * dy, -(xp.arange(Ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(Nz) * dz, -(xp.arange(Nz, 0, -1)) * dz])
    SX, SY, SZ = xp.meshgrid(sx, sy, sz, indexing="ij")
    R = xp.sqrt(SX * SX + SY * SY + SZ * SZ)
    R_reg = xp.where(R < 1e-30, 1e-30, R)
    G = xp.exp(1j * k * R_reg) / (4.0 * xp.pi * R_reg)
    scalar = (grid.dv * G).astype(be.complex_dtype, copy=False)
    # isotropic 3×3 block: tensor[a, b] = δ_{ab} · scalar
    shape = (3, 3) + scalar.shape
    out = be.zeros(shape, kind="complex")
    for d in range(3):
        out[d, d] = scalar
    return out


def prep_coeffs_em3d(grid: Grid, *, k: float, volume: float):
    """Return the precomputed FFT-of-kernel tensor on the doubled grid Π₂."""
    be = grid.backend
    kernel_tensor = _kernel_tensor_on_doubled_grid(grid, k=k, volume=volume)
    return be.fftn(kernel_tensor, axes=(-3, -2, -1)).astype(be.complex_dtype, copy=False)


def prep_conj_coeffs_em3d(grid: Grid, *, k: float, volume: float):
    """FFT of the conjugate-kernel tensor for rmatvec."""
    be = grid.backend
    kernel_tensor = _kernel_tensor_on_doubled_grid(grid, k=k, volume=volume)
    conj = be.xp.conj(kernel_tensor)
    return be.fftn(conj, axes=(-3, -2, -1)).astype(be.complex_dtype, copy=False)
