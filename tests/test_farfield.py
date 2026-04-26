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
    eta = ellipsis_refraction(
        grid, eps_real=eps_real, eps_imag=eps_imag,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    wave = flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=k0, volume=volume)


# --- Test 1: zero contrast (direct method) ---

@pytest.mark.parametrize("method", ["direct", "fft"])
def test_zero_contrast(method):
    """eta=0 → F=0, sigma=0."""
    be = _be()
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = be.zeros((3, 3) + grid.N, kind="complex")
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 64)
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = scatter_integral(wave, problem, directions, method=method)
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


# --- Test 4: fft vs direct agreement ---

def test_fft_vs_direct_agreement():
    """FFT backend matches direct to atol=1e-4 on an 8x8x8 grid."""
    be = _be()
    grid = Grid(N=(8, 8, 8), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 512)
    rng = np.random.default_rng(42)
    u = (rng.standard_normal((3, 8, 8, 8))
         + 1j * rng.standard_normal((3, 8, 8, 8))).astype(np.complex128)
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ])
    F_direct = scatter_integral(u, problem, directions, method="direct")
    F_fft    = scatter_integral(u, problem, directions, method="fft")
    np.testing.assert_allclose(F_fft, F_direct, atol=1e-4,
                                err_msg="FFT and direct backends disagree")


# --- Test 5: rcs_plane shape ---

def test_rcs_plane_shape():
    """rcs_plane returns (phi, sigma) of shape (n_phi,)."""
    problem = _make_problem(k0=0.5)
    phi, sigma = rcs_plane(problem.wave, problem, n_phi=12, plane="xy")
    assert phi.shape == (12,)
    assert sigma.shape == (12,)


# --- Test 6: rcs_plane symmetry ---

def test_rcs_plane_symmetry():
    """Isotropic sphere, real eta, uniform x-field:
    sigma(phi) == sigma(phi + pi) in xy-plane.
    Proof: for real eta and u=(1,0,0), F(phi+pi) = conj(F(phi)),
    so |e_p(phi+pi) x F(phi+pi)|^2 = |e_p(phi) x F(phi)|^2.
    """
    be = _be()
    N = (8, 8, 8)
    grid = Grid(N=N, L=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.4, 0.4, 0.4),
    )
    wave = flat_wave_vec(grid, k=0.1, orient=(0, 0, 1), amplitude=(1, 0, 0))
    u = np.zeros((3,) + N, dtype=np.complex128)
    u[0] = 1.0                                       # uniform x-polarized field
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.1,
                      volume=grid.dv * int(np.prod(N)))
    n_phi = 24
    phi, sigma = rcs_plane(u, problem, n_phi=n_phi, plane="xy")
    half = n_phi // 2
    np.testing.assert_allclose(
        sigma[:half], sigma[half:], rtol=1e-10, atol=1e-35,
        err_msg="RCS not symmetric: sigma(phi) != sigma(phi+pi)",
    )
