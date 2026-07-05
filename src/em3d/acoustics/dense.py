"""Dense scalar acoustic operators for small-grid verification."""
from __future__ import annotations

import numpy as np

from ..grid import Grid
from .kernel import self_cell_coefficient
from .problem import AcousticProblem


def _cell_centres(grid: Grid) -> np.ndarray:
    """Return cell-centre coordinates as ``(Nx*Ny*Nz, 3)`` in row-major order."""
    be = grid.backend
    x = np.asarray(be.to_host(grid.x), dtype=np.float64)
    y = np.asarray(be.to_host(grid.y), dtype=np.float64)
    z = np.asarray(be.to_host(grid.z), dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def flatten_scalar_field(u) -> np.ndarray:
    """Flatten scalar field in row-major order."""
    return np.asarray(u, dtype=np.complex128).reshape(-1)


def unflatten_scalar_field(values, N: tuple[int, int, int]) -> np.ndarray:
    """Restore scalar field from a row-major vector."""
    return np.asarray(values, dtype=np.complex128).reshape(tuple(N))


def B_scalar_matrix(grid: Grid, *, k0: float) -> np.ndarray:
    """Assemble dense scalar convolution matrix ``B`` without eta multiplication."""
    centres = _cell_centres(grid)
    n_cells = centres.shape[0]
    B = np.zeros((n_cells, n_cells), dtype=np.complex128)
    k = float(k0)
    for i in range(n_cells):
        for j in range(n_cells):
            R = float(np.linalg.norm(centres[i] - centres[j]))
            if R < 1e-15:
                B[i, j] = self_cell_coefficient(k0=k, dv=grid.dv)
            else:
                B[i, j] = (k * k) * grid.dv * np.exp(1j * k * R) / (4.0 * np.pi * R)
    return B


def H_scalar_matrix(problem: AcousticProblem) -> np.ndarray:
    """Build dense scalar ``H = I - B diag(eta - 1)``."""
    B = B_scalar_matrix(problem.grid, k0=problem.k0)
    chi = flatten_scalar_field(problem.grid.backend.to_host(problem.chi))
    return np.eye(B.shape[0], dtype=np.complex128) - B @ np.diag(chi)
