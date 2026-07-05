"""FFT-backed scalar acoustic operator."""
from __future__ import annotations

import numpy as np

from .dense import H_scalar_matrix
from .kernel import prep_coeffs_acoustic, prep_conj_coeffs_acoustic
from .problem import AcousticProblem


def _pad_to_doubled(xp, u, N):
    """Zero-pad scalar field to a doubled parallelepiped."""
    Nx, Ny, Nz = N
    out = xp.zeros((2 * Nx, 2 * Ny, 2 * Nz), dtype=u.dtype)
    out[:Nx, :Ny, :Nz] = u
    return out


def _crop_from_doubled(u_big, N):
    """Extract the original scalar field from a doubled parallelepiped."""
    Nx, Ny, Nz = N
    return u_big[:Nx, :Ny, :Nz]


class AcousticOperator:
    """FFT-backed acoustic operator ``H u = u - B((eta - 1)u)``."""

    def __init__(self, problem: AcousticProblem):
        self.problem = problem
        self._be = problem.grid.backend
        self._N = problem.grid.N
        self._K_hat = prep_coeffs_acoustic(problem.grid, k0=problem.k0)
        self._K_hat_conj = prep_conj_coeffs_acoustic(problem.grid, k0=problem.k0)
        self._chi_conj = self._be.xp.conj(problem.chi)

    @property
    def backend(self):
        return self._be

    def matvec(self, u):
        """Apply ``u - B(chi*u)`` to a scalar field."""
        be = self._be
        xp = be.xp
        u_arr = xp.asarray(u, dtype=be.complex_dtype)
        if u_arr.shape != self._N:
            raise ValueError(f"u.shape {u_arr.shape} != expected {self._N}")
        source = self.problem.chi * u_arr
        padded = _pad_to_doubled(xp, source, self._N)
        hat = be.fftn(padded, axes=(-3, -2, -1))
        applied_big = be.ifftn(self._K_hat * hat, axes=(-3, -2, -1))
        return (u_arr - _crop_from_doubled(applied_big, self._N)).astype(
            be.complex_dtype,
            copy=False,
        )

    def rmatvec(self, v):
        """Apply adjoint ``v - conj(chi)*B* v`` to a scalar field."""
        be = self._be
        xp = be.xp
        v_arr = xp.asarray(v, dtype=be.complex_dtype)
        if v_arr.shape != self._N:
            raise ValueError(f"v.shape {v_arr.shape} != expected {self._N}")
        padded = _pad_to_doubled(xp, v_arr, self._N)
        hat = be.fftn(padded, axes=(-3, -2, -1))
        applied_big = be.ifftn(self._K_hat_conj * hat, axes=(-3, -2, -1))
        B_star_v = _crop_from_doubled(applied_big, self._N)
        return (v_arr - self._chi_conj * B_star_v).astype(be.complex_dtype, copy=False)

    def to_dense(self) -> np.ndarray:
        """Return dense ``H`` matrix; requires numpy backend."""
        if self._be.xp is not np:
            raise RuntimeError("AcousticOperator.to_dense requires numpy backend")
        return H_scalar_matrix(self.problem)
