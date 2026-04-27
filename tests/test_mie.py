"""Tests for em3d.mie: Mie scattering analytical solution."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import em3d


# ---------------------------------------------------------------------------
# Task 1 tests: coefficients and cross sections
# ---------------------------------------------------------------------------

def test_mie_coefficients_rayleigh_limit():
    """a1 must match Rayleigh limit: a1 ≈ -2i/3 * (m²-1)/(m²+2) * x³."""
    eps_r = 2.0
    a = 0.1
    k0 = 0.05 / a  # x = k0*a = 0.05 (deep Rayleigh)
    coeffs = em3d.mie.mie_coefficients(a, eps_r, k0)
    a1 = coeffs["a"][0]
    m = np.sqrt(complex(eps_r))
    x = k0 * a
    a1_rayleigh = (-2j / 3) * ((m**2 - 1) / (m**2 + 2)) * x**3
    rel_err = abs(a1 - a1_rayleigh) / abs(a1_rayleigh)
    assert rel_err < 1e-4, f"Rayleigh limit error {rel_err:.2e}"


def test_mie_cross_sections_lossless_zero_absorption():
    """For real eps_r, sigma_abs must be negligibly small."""
    cross = em3d.mie.mie_cross_sections(a=0.3, eps_r=2.0, k0=1.0 / 0.3)
    assert cross["abs"] < 1e-10 * cross["scat"], (
        f"sigma_abs={cross['abs']:.3e} not negligible vs scat={cross['scat']:.3e}"
    )


def test_mie_cross_sections_rayleigh():
    """sigma_scat matches Rayleigh formula within 1% for small x."""
    a = 0.05
    k0 = 0.1 / a  # x = 0.1
    eps_r = 2.0
    cross = em3d.mie.mie_cross_sections(a=a, eps_r=eps_r, k0=k0)
    m = np.sqrt(complex(eps_r))
    lam = 2 * np.pi / k0
    sigma_rayleigh = (128 * np.pi**5 * a**6) / (3 * lam**4) * abs((m**2 - 1) / (m**2 + 2))**2
    rel_err = abs(cross["scat"] - sigma_rayleigh) / sigma_rayleigh
    assert rel_err < 0.01, f"Rayleigh cross section error {rel_err:.2%}"


def test_mie_coefficients_invalid_inputs():
    """ValueError for bad inputs."""
    with pytest.raises(ValueError, match="a must be > 0"):
        em3d.mie.mie_coefficients(a=-1.0, eps_r=2.0, k0=1.0)
    with pytest.raises(ValueError, match="k0 must be > 0"):
        em3d.mie.mie_coefficients(a=1.0, eps_r=2.0, k0=0.0)
    with pytest.raises(ValueError, match="Re\\(eps_r\\) must be > 0"):
        em3d.mie.mie_coefficients(a=1.0, eps_r=-1.0, k0=1.0)


def test_mie_coefficients_large_x_warning():
    """UserWarning when x > 10."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        em3d.mie.mie_coefficients(a=1.0, eps_r=2.0, k0=11.0)
        assert any("x=" in str(wi.message) for wi in w), "Expected UserWarning for x > 10"


# ---------------------------------------------------------------------------
# Task 2 tests: far-field RCS
# ---------------------------------------------------------------------------

def test_mie_rcs_plane_symmetry():
    """RCS in xy-plane has pi-periodicity: sigma(phi) == sigma(phi + pi)."""
    phi, sigma = em3d.mie.mie_rcs_plane(a=0.3, eps_r=2.0, k0=1.0/0.3, n_phi=180, plane="xy")
    half = len(phi) // 2
    np.testing.assert_allclose(sigma[:half], sigma[half:], rtol=1e-10,
                               err_msg="RCS should be pi-periodic in xy plane")


def test_mie_rcs_plane_invalid_inputs():
    """ValueError for bad plane or n_phi."""
    with pytest.raises(ValueError, match="plane must be"):
        em3d.mie.mie_rcs_plane(a=0.3, eps_r=2.0, k0=1.0/0.3, plane="zx")
    with pytest.raises(ValueError, match="n_phi must be"):
        em3d.mie.mie_rcs_plane(a=0.3, eps_r=2.0, k0=1.0/0.3, n_phi=0)


# ---------------------------------------------------------------------------
# Task 3 tests: near-field
# ---------------------------------------------------------------------------

def test_mie_field_center_equals_incident():
    """Field at the sphere centre (r=0) equals the incident wave direction."""
    a = 0.3
    eps_r = 2.0
    k0 = 1.0 / a
    # Default: amplitude=(1,0,0), orient=(0,0,1)
    # At r=0 the Mie series limit gives E = E0 * x̂ = (1,0,0)
    E = em3d.mie.mie_field_at(np.array([[0.0, 0.0, 0.0]]), a, eps_r, k0)
    np.testing.assert_allclose(E[0], [1.0, 0.0, 0.0], atol=1e-10,
                               err_msg="Field at sphere centre should equal incident wave")


def test_mie_field_matches_grid_wrapper(backend_numpy_double):
    """mie_field_at and mie_field(grid) give identical values at grid nodes."""
    a = 0.3
    eps_r = 2.0
    k0 = 1.0 / a
    be = backend_numpy_double
    grid = em3d.Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0, 0, 0), backend=be)
    X, Y, Z = grid.coords()
    xyz = np.stack([np.asarray(X).ravel(), np.asarray(Y).ravel(), np.asarray(Z).ravel()], axis=1)
    E_at = em3d.mie.mie_field_at(xyz, a, eps_r, k0)
    E_grid = em3d.mie.mie_field(grid, a, eps_r, k0)
    E_grid_flat = E_grid.reshape(3, -1).T   # (N**3, 3)
    np.testing.assert_allclose(E_at, E_grid_flat, atol=1e-12,
                               err_msg="mie_field_at and mie_field must agree on grid nodes")


def test_mie_field_at_invalid_inputs():
    """ValueError for bad xyz shape or amplitude ∥ orient."""
    with pytest.raises(ValueError, match="xyz must have shape"):
        em3d.mie.mie_field_at(np.zeros((5,)), a=0.3, eps_r=2.0, k0=1.0/0.3)
    with pytest.raises(ValueError, match="xyz must have shape"):
        em3d.mie.mie_field_at(np.zeros((5, 2)), a=0.3, eps_r=2.0, k0=1.0/0.3)
    with pytest.raises(ValueError, match="parallel"):
        em3d.mie.mie_field_at(np.zeros((1, 3)), a=0.3, eps_r=2.0, k0=1.0/0.3,
                               amplitude=(0, 0, 1), orient=(0, 0, 1))
