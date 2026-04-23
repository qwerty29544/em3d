import numpy as np
import pytest

from em3d.grid import Grid


def test_grid_dv_and_shape(backend_numpy_double):
    grid = Grid(N=(4, 5, 6), L=(1.0, 2.0, 3.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    assert grid.dv == pytest.approx((1.0 / 4) * (2.0 / 5) * (3.0 / 6))
    X, Y, Z = grid.coords()
    assert X.shape == Y.shape == Z.shape == (4, 5, 6)


def test_grid_coords_center_offset(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1.0, 1.0, 1.0), center=(10.0, 0.0, 0.0), backend=backend_numpy_double)
    X, _, _ = grid.coords()
    # cell centres should be offset by 10 in x
    assert float(X.min()) > 9.0
    assert float(X.max()) < 11.0


def test_grid_dtype_matches_backend(backend_cpu):
    grid = Grid(N=(3, 3, 3), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_cpu)
    X, _, _ = grid.coords()
    assert X.dtype == backend_cpu.real_dtype


def test_grid_rejects_non_positive_N(backend_numpy_double):
    with pytest.raises(ValueError):
        Grid(N=(0, 2, 2), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)


def test_grid_rejects_non_positive_L(backend_numpy_double):
    with pytest.raises(ValueError):
        Grid(N=(2, 2, 2), L=(1.0, -1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
