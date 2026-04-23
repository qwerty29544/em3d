"""Reference dense assembly of the volume-integral operator.

Only used for integration testing on small grids. Always numpy.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .kernel import b_coeff


def flatten_block_matrix(T4: np.ndarray) -> np.ndarray:
    """Unpack a (N, N, m, m) block tensor into a (N·m, N·m) matrix (row-major blocks)."""
    N, N2, m, m2 = T4.shape
    if N != N2 or m != m2:
        raise ValueError(f"expected (N,N,m,m) tensor, got {T4.shape}")
    # reshape: (N, N, m, m) -> (N, m, N, m) -> (Nm, Nm)
    return T4.transpose(0, 2, 1, 3).reshape(N * m, N * m)


def _cell_centres(grid: Grid) -> np.ndarray:
    """Return array of shape (Nx·Ny·Nz, 3) with cell-centre coordinates in row-major order."""
    be = grid.backend
    x = np.asarray(be.to_host(grid.x))
    y = np.asarray(be.to_host(grid.y))
    z = np.asarray(be.to_host(grid.z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def B_operator_matrix(grid: Grid, *, k: float, volume: float) -> np.ndarray:
    """Assemble the dense 3Ncells × 3Ncells matrix of the volume-integral operator.

    Uses scalar b_coeff and a block-diagonal 3×3 identity per cell pair (isotropic
    kernel, as in the notebook's `B_operator_matrix` without ε multiplication).
    `volume` is accepted for API symmetry with the FFT operator but is unused here;
    cell volume is always taken from grid.dv.
    """
    centres = _cell_centres(grid)
    Ncells = centres.shape[0]
    dv = grid.dv
    M = np.zeros((3 * Ncells, 3 * Ncells), dtype=np.complex128)
    identity3 = np.eye(3, dtype=np.complex128)
    for i in range(Ncells):
        for j in range(Ncells):
            coeff = b_coeff(centres[i], centres[j], k=k, dv=dv)
            M[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = coeff * identity3
    return M
