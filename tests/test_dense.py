import numpy as np

from em3d.grid import Grid
from em3d.dense import B_operator_matrix, flatten_block_matrix
from em3d.kernel import b_coeff


def test_B_operator_matrix_shape_small_grid(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(0.5, 0.5, 0.5), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    M = B_operator_matrix(grid, k=1.0, volume=grid.dv * 8)
    # 2·2·2 = 8 cells, 3 components each → 24×24
    assert M.shape == (24, 24)
    assert M.dtype == np.complex128


def test_B_operator_matrix_uses_dyadic_blocks(backend_numpy_double):
    """Dense reference must preserve full 3x3 dyadic B blocks, not scalar I blocks."""
    grid = Grid(N=(2, 2, 2), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    M = B_operator_matrix(grid, k=1.0, volume=grid.dv * 8)
    centres = []
    for x in grid.x:
        for y in grid.y:
            for z in grid.z:
                centres.append(np.array([x, y, z], dtype=np.float64))

    block01 = M[0:3, 3:6]
    expected01 = b_coeff(centres[0], centres[1], k=1.0, dv=grid.dv)
    np.testing.assert_allclose(block01, expected01, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(M[0:3, 0:3], (-1.0 / 3.0) * np.eye(3), atol=1e-14)


def test_flatten_block_matrix_roundtrip():
    rng = np.random.default_rng(0)
    N = 3
    m = 2
    T4 = rng.standard_normal((N, N, m, m)) + 1j * rng.standard_normal((N, N, m, m))
    flat = flatten_block_matrix(T4)
    assert flat.shape == (N * m, N * m)
    # sample a couple of entries to verify layout
    for i in range(N):
        for j in range(N):
            for a in range(m):
                for b in range(m):
                    assert flat[i * m + a, j * m + b] == T4[i, j, a, b]
