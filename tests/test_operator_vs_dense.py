import numpy as np

from em3d.grid import Grid
from em3d.operator import prep_coeffs_em3d


def test_prep_coeffs_shape_and_dtype(backend_numpy_double):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    coeffs = prep_coeffs_em3d(grid, k=1.0, volume=grid.dv * 64)
    # Π₂ doubling: FFT tensor on (2Nx, 2Ny, 2Nz) with 3×3 block structure
    assert coeffs.shape == (3, 3) + tuple(2 * n for n in grid.N)
    assert coeffs.dtype == backend_numpy_double.complex_dtype
