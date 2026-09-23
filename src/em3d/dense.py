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


def _cell_centres_backend(grid: Grid):
    """Cell centres as a backend array of shape ``(M, 3)``."""

    xp = grid.backend.xp
    X, Y, Z = grid.coords()
    return xp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=-1)


def dense_kernel_blocks_backend(grid: Grid, *, k: float):
    """Return dyadic kernel blocks on the selected NumPy/CuPy backend.

    The result has shape ``(M, M, 3, 3)`` where the first pair of indices
    identifies receiver/source cells.  The implementation is algebraically
    identical to :func:`em3d.kernel.b_coeff`, including the ``-I/3``
    self-interaction block, but is vectorized so that full spectra on moderate
    grids can be evaluated on a CUDA device.
    """

    be = grid.backend
    xp = be.xp
    centres = _cell_centres_backend(grid)
    difference = centres[:, None, :] - centres[None, :, :]
    radius = xp.linalg.norm(difference, axis=-1)
    is_self = radius < 1e-15
    radius_safe = xp.where(is_self, xp.ones_like(radius), radius)

    alpha = difference / radius_safe[..., None]
    alpha_outer = alpha[..., :, None] * alpha[..., None, :]
    inverse_radius = 1.0 / radius_safe
    inverse_radius_squared = inverse_radius * inverse_radius
    ik = be.complex_dtype(1j * float(k))
    coefficient_1 = (
        3.0 * inverse_radius_squared
        - 3.0 * ik * inverse_radius
        - float(k) * float(k)
    )
    coefficient_2 = (
        float(k) * float(k)
        + ik * inverse_radius
        - inverse_radius_squared
    )
    green = xp.exp(ik * radius_safe) / (4.0 * xp.pi * radius_safe)
    identity = xp.eye(3, dtype=be.complex_dtype)
    blocks = (
        green[..., None, None]
        * be.real_dtype(grid.dv)
        * (
            coefficient_1[..., None, None] * alpha_outer
            + coefficient_2[..., None, None] * identity
        )
    )
    self_block = (-1.0 / 3.0) * identity
    blocks = xp.where(is_self[..., None, None], self_block, blocks)
    return blocks.astype(be.complex_dtype, copy=False)


def dense_kernel_matrix_backend(grid: Grid, *, k: float):
    """Full kernel matrix in cell-major/component-minor ordering."""

    blocks = dense_kernel_blocks_backend(grid, k=k)
    cells = int(np.prod(grid.N))
    return blocks.transpose(0, 2, 1, 3).reshape(3 * cells, 3 * cells)


def dense_kernel_matrix(grid: Grid, *, k: float) -> np.ndarray:
    """Host representation of the full dyadic kernel matrix."""

    return np.asarray(
        grid.backend.to_host(dense_kernel_matrix_backend(grid, k=k)),
        dtype=np.complex128,
    )


def dense_kernel_matrix_reference(grid: Grid, *, k: float) -> np.ndarray:
    """Slow scalar reference used by regression tests.

    Keeping this implementation separate makes it possible to verify the
    vectorized CPU/GPU construction against the original per-block formula.
    """

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


def contrast_blocks_backend(contrast_tensor, *, backend):
    contrast = backend.asarray_of_kind(contrast_tensor, kind="complex")
    if contrast.ndim != 5 or contrast.shape[:2] != (3, 3):
        raise ValueError(
            "contrast_tensor must have shape (3, 3, Nx, Ny, Nz), "
            f"got {contrast.shape}"
        )
    cells = int(np.prod(contrast.shape[2:]))
    return contrast.transpose(2, 3, 4, 0, 1).reshape(cells, 3, 3)


def contrast_block_matrix_backend(contrast_tensor, *, backend):
    blocks = contrast_blocks_backend(contrast_tensor, backend=backend)
    cells = int(blocks.shape[0])
    matrix = backend.xp.zeros(
        (3 * cells, 3 * cells), dtype=backend.complex_dtype
    )
    indices = backend.xp.arange(cells)
    for row in range(3):
        for column in range(3):
            matrix[3 * indices + row, 3 * indices + column] = blocks[:, row, column]
    return matrix


def contrast_block_matrix(contrast_tensor, *, backend=None) -> np.ndarray:
    if backend is None:
        contrast = np.asarray(contrast_tensor, dtype=np.complex128)
        if contrast.ndim != 5 or contrast.shape[:2] != (3, 3):
            raise ValueError(
                "contrast_tensor must have shape (3, 3, Nx, Ny, Nz), "
                f"got {contrast.shape}"
            )
        cells = int(np.prod(contrast.shape[2:]))
        blocks = contrast.transpose(2, 3, 4, 0, 1).reshape(cells, 3, 3)
        matrix = np.zeros((3 * cells, 3 * cells), dtype=np.complex128)
        indices = np.arange(cells)
        for row in range(3):
            for column in range(3):
                matrix[3 * indices + row, 3 * indices + column] = blocks[:, row, column]
        return matrix
    return np.asarray(
        backend.to_host(
            contrast_block_matrix_backend(contrast_tensor, backend=backend)
        ),
        dtype=np.complex128,
    )


def dense_operator_matrix_backend(problem):
    """Complete collocation matrix ``A = I - B chi`` on the problem backend."""

    be = problem.grid.backend
    xp = be.xp
    kernel_blocks = dense_kernel_blocks_backend(problem.grid, k=problem.k0)
    contrast_blocks = contrast_blocks_backend(
        problem.eps_tensor, backend=be
    )
    interaction = xp.einsum(
        "ijab,jbd->ijad", kernel_blocks, contrast_blocks, optimize=True
    )
    cells = int(np.prod(problem.grid.N))
    identity = xp.eye(3, dtype=be.complex_dtype)
    cell_identity = xp.eye(cells, dtype=be.complex_dtype)
    operator_blocks = (
        cell_identity[..., None, None] * identity - interaction
    )
    return operator_blocks.transpose(0, 2, 1, 3).reshape(
        3 * cells, 3 * cells
    )


def dense_operator_matrix(problem) -> np.ndarray:
    """Host representation of ``A = I - B chi``."""

    return np.asarray(
        problem.grid.backend.to_host(dense_operator_matrix_backend(problem)),
        dtype=np.complex128,
    )
