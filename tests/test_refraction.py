import numpy as np

from em3d.grid import Grid
from em3d.refraction import (
    cylinder_refraction,
    step_refraction,
    ellipsis_refraction,
    apply_refraction,
)


def _grid(be):
    return Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


def test_cylinder_mask_inside_outside(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_real = 2.25
    mask = cylinder_refraction(grid, eps_real=eps_real, eps_imag=0.0, radius=0.49, axis="z")
    assert mask.shape == (4, 4, 4)
    centre = mask[grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2]
    corner = mask[0, 0, 0]
    assert abs(centre.real - (eps_real - 1.0)) < 1e-12
    assert abs(corner) < 1e-12


def test_step_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    mask = step_refraction(
        grid, eps_real=2.0, eps_imag=0.1, z_min=-0.25, z_max=0.25
    )
    assert mask.shape == (4, 4, 4)
    mid = grid.N[2] // 2
    assert abs(mask[0, 0, mid]) > 0
    assert abs(mask[0, 0, 0]) < 1e-12
    assert abs(mask[0, 0, -1]) < 1e-12


def test_ellipsis_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    mask = ellipsis_refraction(
        grid, eps_real=1.5, eps_imag=0.0, center=(0.0, 0.0, 0.0), radius=(0.3, 0.4, 0.5)
    )
    assert mask.shape == (4, 4, 4)
    assert abs(mask[grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2]) > 0


def test_apply_refraction_scalar_returns_isotropic_tensor(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.49, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    assert eta.shape == (3, 3) + grid.N
    assert eta.dtype == backend_numpy_double.complex_dtype
    np.testing.assert_allclose(eta[0, 0], scalar)
    np.testing.assert_allclose(eta[1, 1], scalar)
    np.testing.assert_allclose(eta[2, 2], scalar)
    np.testing.assert_allclose(eta[0, 1], np.zeros_like(scalar))
