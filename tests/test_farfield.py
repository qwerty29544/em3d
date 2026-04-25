"""Tests for em3d.farfield — scatter_integral, rcs, rcs_plane."""
import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.grid import Grid
from em3d.refraction import apply_refraction, ellipsis_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.farfield import scatter_integral, rcs, rcs_plane


def _be():
    return Backend.numpy(Precision.DOUBLE)


def _make_problem(N=(4, 4, 4), eps_real=2.0, eps_imag=0.0, k0=1.0):
    be = _be()
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    scalar = ellipsis_refraction(
        grid, eps_real=eps_real, eps_imag=eps_imag,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=k0, volume=volume)


# --- Test 1: zero contrast (direct method) ---

def test_zero_contrast():
    """eta=0 → F=0, sigma=0."""
    be = _be()
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = be.zeros((3, 3) + grid.N, kind="complex")
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 64)
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = scatter_integral(wave, problem, directions, method="direct")
    np.testing.assert_allclose(F, 0.0, atol=1e-14)


# --- Test 2: non-negative ---

def test_rcs_nonnegative():
    """RCS >= 0 for arbitrary non-zero field."""
    problem = _make_problem()
    direction = np.array([1.0, 0.0, 0.0])
    result = rcs(problem.wave, problem, direction)
    assert result >= 0.0


# --- Test 3: single cell analytic ---

def test_single_cell_analytic():
    """1x1x1 grid at origin: F = dv * eta @ u (phase=1 since r_cell=0)."""
    be = _be()
    grid = Grid(N=(1, 1, 1), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    # Grid cell is at r=(0,0,0): start = center - L/2 + dx/2 = 0 - 0.5 + 0.5 = 0
    eta_val = complex(1.5, 0.1)
    scalar = np.full(grid.N, eta_val, dtype=np.complex128)
    eta = apply_refraction(grid, scalar_eta=scalar)
    u = np.zeros((3, 1, 1, 1), dtype=np.complex128)
    u[0, 0, 0, 0] = 1.0 + 0.5j          # x-component only
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv)
    direction = np.array([1.0, 0.0, 0.0])
    F = scatter_integral(u, problem, direction[np.newaxis, :])[0]   # (3,)
    # analytic: r_cell=(0,0,0), phase=exp(0)=1
    # J = eta_val * u  (isotropic diagonal eta)
    # F = dv * J = 1.0 * eta_val * (1+0.5j, 0, 0)
    # dv = 1.0
    F_analytic = np.array([eta_val * (1.0 + 0.5j), 0.0, 0.0])
    np.testing.assert_allclose(F, F_analytic, rtol=1e-12)


# --- Test 5: rcs_plane shape ---

def test_rcs_plane_shape():
    """rcs_plane returns (phi, sigma) of shape (n_phi,)."""
    problem = _make_problem(k0=0.5)
    phi, sigma = rcs_plane(problem.wave, problem, n_phi=12, plane="xy")
    assert phi.shape == (12,)
    assert sigma.shape == (12,)
