import numpy as np
import pytest

from em3d.grid import Grid
from em3d.refraction import (
    cylinder_refraction,
    step_refraction,
    ellipsis_refraction,
    apply_refraction,
)


def _grid(be):
    return Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


# --- Updated geometry tests: shape now (3,3,Nx,Ny,Nz) ---

def test_cylinder_mask_inside_outside(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_real = 2.25
    eta = cylinder_refraction(grid, eps_real=eps_real, eps_imag=0.0, radius=0.49, axis="z")
    assert eta.shape == (3, 3, 4, 4, 4)
    cx, cy = grid.N[0] // 2, grid.N[1] // 2
    expected_eta = eps_real - 1.0
    for d in range(3):
        np.testing.assert_allclose(eta[d, d, cx, cy, 0].real, expected_eta, atol=1e-12)
    np.testing.assert_allclose(eta[0, 1], np.zeros(grid.N))     # off-diagonal = 0
    np.testing.assert_allclose(eta[0, 0, 0, 0, :], np.zeros(grid.N[2]))  # corner = 0


def test_step_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eta = step_refraction(grid, eps_real=2.0, eps_imag=0.1, z_min=-0.25, z_max=0.25)
    assert eta.shape == (3, 3, 4, 4, 4)
    mid = grid.N[2] // 2  # index 2 → z=0.125, inside z_min=-0.25..z_max=0.25
    assert abs(eta[0, 0, 0, 0, mid]) > 0
    np.testing.assert_allclose(eta[0, 0, 0, 0, 0], 0, atol=1e-12)
    np.testing.assert_allclose(eta[0, 0, 0, 0, -1], 0, atol=1e-12)


def test_ellipsis_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eta = ellipsis_refraction(
        grid, eps_real=1.5, eps_imag=0.0, center=(0.0, 0.0, 0.0), radius=(0.3, 0.4, 0.5)
    )
    assert eta.shape == (3, 3, 4, 4, 4)
    cx, cy, cz = grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2
    np.testing.assert_allclose(eta[0, 0, cx, cy, cz].real, 1.5 - 1.0, atol=1e-12)


# --- apply_refraction still works with manually-built scalar field ---

def test_apply_refraction_scalar_returns_isotropic_tensor(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_real = 2.0
    scalar = np.full(grid.N, complex(eps_real - 1.0, 0.0))
    eta = apply_refraction(grid, scalar_eta=scalar)
    assert eta.shape == (3, 3) + grid.N
    assert eta.dtype == backend_numpy_double.complex_dtype
    np.testing.assert_allclose(eta[0, 0], scalar)
    np.testing.assert_allclose(eta[1, 1], scalar)
    np.testing.assert_allclose(eta[2, 2], scalar)
    np.testing.assert_allclose(eta[0, 1], np.zeros_like(scalar))


# --- New: anisotropic input ---

def test_ellipsis_anisotropic(backend_numpy_double):
    """Diagonal anisotropic eps: each component of eta diagonal differs."""
    grid = _grid(backend_numpy_double)
    eps_r = np.diag([2.0, 1.5, 1.2])
    eps_i = np.zeros((3, 3))
    eta = ellipsis_refraction(
        grid, eps_real=eps_r, eps_imag=eps_i,
        center=(0, 0, 0), radius=(0.3, 0.3, 0.3),
    )
    assert eta.shape == (3, 3) + grid.N
    cx, cy, cz = grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2
    np.testing.assert_allclose(eta[0, 0, cx, cy, cz], 1.0, atol=1e-12)  # 2.0 - 1
    np.testing.assert_allclose(eta[1, 1, cx, cy, cz], 0.5, atol=1e-12)  # 1.5 - 1
    np.testing.assert_allclose(eta[2, 2, cx, cy, cz], 0.2, atol=1e-12)  # 1.2 - 1
    np.testing.assert_allclose(eta[0, 1, cx, cy, cz], 0.0, atol=1e-12)  # off-diagonal
    np.testing.assert_allclose(eta[:, :, 0, 0, 0], np.zeros((3, 3)), atol=1e-12)


# --- Guard: passing geometry-function result to scalar_eta raises ValueError ---

def test_apply_refraction_rejects_tensor_as_scalar(backend_numpy_double):
    """Passing a (3,3,Nx,Ny,Nz) tensor to scalar_eta= raises a helpful ValueError."""
    grid = _grid(backend_numpy_double)
    eta_tensor = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.49, axis="z")
    assert eta_tensor.shape == (3, 3) + grid.N  # sanity-check the fixture
    with pytest.raises(ValueError, match=r"apply_refraction is not needed"):
        apply_refraction(grid, scalar_eta=eta_tensor)


# --- New: type mismatch raises TypeError ---

def test_to_eta_mat_type_mismatch(backend_numpy_double):
    """Mixing scalar eps_real with matrix eps_imag must raise TypeError."""
    grid = _grid(backend_numpy_double)
    with pytest.raises(TypeError):
        ellipsis_refraction(
            grid, eps_real=2.0, eps_imag=np.zeros((3, 3)),
            center=(0, 0, 0), radius=(0.1, 0.1, 0.1),
        )
