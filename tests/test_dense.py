import numpy as np

from em3d.grid import Grid
from em3d.dense import B_operator_matrix, flatten_block_matrix


def test_B_operator_matrix_shape_small_grid(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(0.5, 0.5, 0.5), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    M = B_operator_matrix(grid, k=1.0, volume=grid.dv * 8)
    # 2·2·2 = 8 cells, 3 components each → 24×24
    assert M.shape == (24, 24)
    assert M.dtype == np.complex128


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
