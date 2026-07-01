"""Scalar Helmholtz kernel coefficients for acoustic integral equations."""
from __future__ import annotations

from ..grid import Grid


def self_cell_coefficient(*, k0: float, dv: float) -> complex:
    """Equivalent-sphere approximation of ``k0^2 integral_cell G(r) dV``."""
    import numpy as np

    k = float(k0)
    cell_volume = float(dv)
    if k <= 0.0:
        raise ValueError(f"k0 must be positive, got {k0}")
    if cell_volume <= 0.0:
        raise ValueError(f"dv must be positive, got {dv}")
    a = (3.0 * cell_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    return complex(np.exp(1j * k * a) * (1.0 - 1j * k * a) - 1.0)


def kernel_on_doubled_grid(grid: Grid, *, k0: float):
    """Sample scalar ``k0^2 G`` cell coefficients on a doubled grid."""
    be = grid.backend
    xp = be.xp
    Nx, Ny, Nz = grid.N
    dx = grid.L[0] / Nx
    dy = grid.L[1] / Ny
    dz = grid.L[2] / Nz
    sx = xp.concatenate([xp.arange(Nx) * dx, -(xp.arange(Nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(Ny) * dy, -(xp.arange(Ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(Nz) * dz, -(xp.arange(Nz, 0, -1)) * dz])
    SX, SY, SZ = xp.meshgrid(sx, sy, sz, indexing="ij")
    R = xp.sqrt(SX * SX + SY * SY + SZ * SZ)
    is_self = R < 1e-15
    R_safe = xp.where(is_self, xp.ones_like(R), R)
    ik = be.complex_dtype(1j * float(k0))
    green_coeff = (float(k0) ** 2) * grid.dv * xp.exp(ik * R_safe) / (4.0 * xp.pi * R_safe)
    self_value = be.complex_dtype(self_cell_coefficient(k0=float(k0), dv=grid.dv))
    out = xp.where(is_self, self_value, green_coeff)
    return out.astype(be.complex_dtype, copy=False)


def prep_coeffs_acoustic(grid: Grid, *, k0: float):
    """Return FFT of scalar acoustic kernel on the doubled grid."""
    be = grid.backend
    return be.fftn(kernel_on_doubled_grid(grid, k0=k0), axes=(-3, -2, -1)).astype(
        be.complex_dtype,
        copy=False,
    )


def prep_conj_coeffs_acoustic(grid: Grid, *, k0: float):
    """Return FFT of conjugate scalar acoustic kernel for adjoint convolution."""
    be = grid.backend
    return be.fftn(be.xp.conj(kernel_on_doubled_grid(grid, k0=k0)), axes=(-3, -2, -1)).astype(
        be.complex_dtype,
        copy=False,
    )
