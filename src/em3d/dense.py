from __future__ import annotations

import numpy as np

from .grid import Grid
from .kernel import b_coeff


def flatten_block_matrix(tensor: np.ndarray) -> np.ndarray:
    rows, columns, block_rows, block_columns = tensor.shape
    if rows != columns or block_rows != block_columns:
        raise ValueError(f"expected (N, N, m, m) tensor, got {tensor.shape}")
    return tensor.transpose(0, 2, 1, 3).reshape(
        rows * block_rows, columns * block_columns
    )


def _cell_centres(grid: Grid) -> np.ndarray:
    backend = grid.backend
    x = np.asarray(backend.to_host(grid.x))
    y = np.asarray(backend.to_host(grid.y))
    z = np.asarray(backend.to_host(grid.z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def dense_kernel_matrix(grid: Grid, *, k: float) -> np.ndarray:
    centres = _cell_centres(grid)
    cells = centres.shape[0]
    matrix = np.zeros((3 * cells, 3 * cells), dtype=np.complex128)
    for row in range(cells):
        for column in range(cells):
            matrix[
                3 * row : 3 * row + 3,
                3 * column : 3 * column + 3,
            ] = b_coeff(
                centres[row], centres[column], k=k, dv=grid.dv
            )
    return matrix


def B_operator_matrix(
    grid: Grid,
    *,
    k: float,
    volume: float | None = None,
) -> np.ndarray:
    """Backward-compatible name for :func:`dense_kernel_matrix`."""

    return dense_kernel_matrix(grid, k=k)


def contrast_block_matrix(contrast_tensor, *, backend=None) -> np.ndarray:
    if backend is not None:
        contrast = np.asarray(backend.to_host(contrast_tensor), dtype=np.complex128)
    else:
        contrast = np.asarray(contrast_tensor, dtype=np.complex128)
    if contrast.ndim != 5 or contrast.shape[:2] != (3, 3):
        raise ValueError(
            "contrast_tensor must have shape (3, 3, Nx, Ny, Nz), "
            f"got {contrast.shape}"
        )
    cells = int(np.prod(contrast.shape[2:]))
    blocks = contrast.transpose(2, 3, 4, 0, 1).reshape(cells, 3, 3)
    matrix = np.zeros((3 * cells, 3 * cells), dtype=np.complex128)
    for cell, block in enumerate(blocks):
        matrix[3 * cell : 3 * cell + 3, 3 * cell : 3 * cell + 3] = block
    return matrix


def dense_operator_matrix(problem) -> np.ndarray:
    """Build the complete collocation matrix ``A = I - B chi``."""

    kernel = dense_kernel_matrix(problem.grid, k=problem.k0)
    contrast = contrast_block_matrix(
        problem.eps_tensor, backend=problem.grid.backend
    )
    return np.eye(kernel.shape[0], dtype=np.complex128) - kernel @ contrast
