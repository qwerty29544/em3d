import numpy as np

import em3d
from em3d.acoustics import eta_homogeneous, make_acoustic_problem
from em3d.acoustics.dense import (
    B_scalar_matrix,
    H_scalar_matrix,
    flatten_scalar_field,
    unflatten_scalar_field,
)
from em3d.acoustics.kernel import kernel_on_doubled_grid, self_cell_coefficient


def _grid(N=(3, 2, 2)):
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=N, L=(1.0, 0.8, 0.6), center=(0.0, 0.0, 0.0), backend=be)


def test_acoustic_self_cell_is_finite_and_low_frequency_small():
    value = self_cell_coefficient(k0=1e-3, dv=1e-6)
    assert np.isfinite(value.real)
    assert np.isfinite(value.imag)
    assert abs(value) < 1e-6


def test_acoustic_kernel_doubled_grid_shape_and_self_term():
    grid = _grid()
    kernel = kernel_on_doubled_grid(grid, k0=2.0)
    assert kernel.shape == (2 * grid.N[0], 2 * grid.N[1], 2 * grid.N[2])
    assert np.isclose(kernel[0, 0, 0], self_cell_coefficient(k0=2.0, dv=grid.dv))


def test_acoustic_dense_identity_when_eta_is_background():
    grid = _grid()
    eta = eta_homogeneous(grid, 1.0)
    problem = make_acoustic_problem(grid, eta, k0=1.5)
    H = H_scalar_matrix(problem)
    assert H.shape == (np.prod(grid.N), np.prod(grid.N))
    assert np.allclose(H, np.eye(H.shape[0]))


def test_acoustic_dense_flatten_roundtrip():
    grid = _grid()
    u = np.arange(np.prod(grid.N), dtype=np.complex128).reshape(grid.N)
    flat = flatten_scalar_field(u)
    assert flat.shape == (np.prod(grid.N),)
    assert np.allclose(unflatten_scalar_field(flat, grid.N), u)


def test_acoustic_dense_matrix_has_nonzero_scattering_for_eta_not_one():
    grid = _grid((2, 2, 2))
    eta = eta_homogeneous(grid, 2.0 + 0.1j)
    problem = make_acoustic_problem(grid, eta, k0=1.0)
    B = B_scalar_matrix(grid, k0=problem.k0)
    H = H_scalar_matrix(problem)
    assert B.shape == H.shape
    assert not np.allclose(H, np.eye(H.shape[0]))
