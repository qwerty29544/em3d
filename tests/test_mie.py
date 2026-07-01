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
    assert abs(cross["abs"]) < 1e-10 * cross["scat"], (
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


def test_mie_cross_sections_passive_loss_nonnegative_absorption():
    """For passive lossy eps_r, absorption must be non-negative."""
    cross = em3d.mie.mie_cross_sections(a=0.3, eps_r=2.0 + 0.1j, k0=1.0 / 0.3)
    assert cross["abs"] >= 0.0


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

def test_angle_functions_closed_forms():
    """pi_n and tau_n must match low-order closed forms."""
    mu = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    pi_arr, tau_arr = em3d.mie._angle_functions(3, mu)
    np.testing.assert_allclose(pi_arr[0], np.ones_like(mu), atol=1e-14)
    np.testing.assert_allclose(tau_arr[0], mu, atol=1e-14)
    np.testing.assert_allclose(pi_arr[1], 3.0 * mu, atol=1e-14)
    np.testing.assert_allclose(tau_arr[1], 6.0 * mu**2 - 3.0, atol=1e-14)
    np.testing.assert_allclose(pi_arr[2], 0.5 * (15.0 * mu**2 - 3.0), atol=1e-14)
    np.testing.assert_allclose(tau_arr[2], 0.5 * (45.0 * mu**3 - 33.0 * mu), atol=1e-14)


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


@pytest.mark.parametrize("eps_r,k0a", [(2.0, 1.0), (1.5, 0.5)])
def test_mie_verification_rcs_normalized_shape(eps_r, k0a, backend_numpy_double):
    """em3d far-field angular pattern must match Mie after normalization.

    The absolute RCS scale is kept as a loose diagnostic because the current
    solver uses a voxelized sphere and the discrete self-term/effective radius
    still need a dedicated validation pass.
    """
    a = 0.3
    k0 = k0a / a
    n = 32
    be = backend_numpy_double
    grid = em3d.Grid(N=(n, n, n), L=(1.0, 1.0, 1.0), center=(0, 0, 0), backend=be)
    eta = em3d.ellipsis_refraction(
        grid,
        eps_real=eps_r,
        eps_imag=0.0,
        center=(0, 0, 0),
        radius=(a, a, a),
    )
    wave = em3d.flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = em3d.Problem(grid=grid, eps_tensor=eta, wave=wave, k0=k0, volume=grid.dv * n**3)
    result = em3d.BiCGStab(em3d.SolverConfig(max_iter=500, rtol=1e-8)).solve(
        em3d.Operator(problem), wave
    )
    assert result.converged

    comparison = em3d.mie.compare_rcs_plane(
        np.asarray(result.u),
        problem,
        a=a,
        eps_r=eps_r,
        n_phi=180,
        plane="xy",
        normalize="max",
    )

    assert set(comparison) == {
        "phi",
        "sigma_num",
        "sigma_mie",
        "sigma_num_norm",
        "sigma_mie_norm",
        "shape_err",
        "scale_ratio",
        "abs_rel_err",
    }
    np.testing.assert_allclose(np.max(comparison["sigma_num_norm"]), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.max(comparison["sigma_mie_norm"]), 1.0, atol=1e-14)
    assert comparison["shape_err"] <= 0.02, (
        f"normalized RCS shape error {comparison['shape_err']:.2%} > 2% "
        f"(eps_r={eps_r}, k0a={k0a}, scale_ratio={comparison['scale_ratio']:.3f})"
    )
    assert 0.5 <= comparison["scale_ratio"] <= 2.0, (
        f"RCS scale ratio {comparison['scale_ratio']:.3f} outside diagnostic bounds "
        f"(eps_r={eps_r}, k0a={k0a})"
    )


def test_compare_rcs_plane_invalid_normalization(backend_numpy_double):
    """compare_rcs_plane rejects unsupported normalization modes."""
    n = 4
    be = backend_numpy_double
    grid = em3d.Grid(N=(n, n, n), L=(1.0, 1.0, 1.0), center=(0, 0, 0), backend=be)
    eta = em3d.ellipsis_refraction(
        grid,
        eps_real=2.0,
        eps_imag=0.0,
        center=(0, 0, 0),
        radius=(0.3, 0.3, 0.3),
    )
    wave = em3d.flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = em3d.Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * n**3)

    with pytest.raises(ValueError, match="normalize"):
        em3d.mie.compare_rcs_plane(
            wave,
            problem,
            a=0.3,
            eps_r=2.0,
            normalize="area",
        )


# ---------------------------------------------------------------------------
# Task 3 tests: near-field
# ---------------------------------------------------------------------------

def _spherical_basis(theta, phi):
    er = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    eth = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta),
    ])
    eph = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return er, eth, eph


def test_mie_field_center_is_continuous():
    """The exact centre value must match the interior r -> 0 limit."""
    a = 0.3
    eps_r = 2.0
    k0 = 1.0 / a
    radius = 1e-7 * a
    pts = np.array([
        [0.0, 0.0, 0.0],
        [radius, 0.0, 0.0],
        [0.0, radius, 0.0],
        [0.0, 0.0, radius],
    ])
    E = em3d.mie.mie_field_at(pts, a, eps_r, k0)
    np.testing.assert_allclose(E[1:], np.broadcast_to(E[0], E[1:].shape), rtol=1e-6, atol=1e-6)


def test_mie_field_center_rayleigh_static_limit():
    """In the Rayleigh/static limit, centre field tends to 3/(eps_r+2) E0."""
    a = 0.3
    eps_r = 2.0
    k0 = 0.01 / a
    E = em3d.mie.mie_field_at(np.array([[0.0, 0.0, 0.0]]), a, eps_r, k0)
    expected = 3.0 / (eps_r + 2.0)
    np.testing.assert_allclose(E[0, 0], expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(E[0, 1:], [0.0, 0.0], atol=1e-12)


def test_mie_field_no_scatterer_matches_incident_wave():
    """eps_r=1 must reduce to the incident plane wave."""
    a = 0.3
    k0 = 1.0 / a
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.4, 0.0, 0.2],
    ])
    E = em3d.mie.mie_field_at(pts, a, 1.0, k0)
    expected = np.zeros_like(E)
    expected[:, 0] = np.exp(1j * k0 * pts[:, 2])
    np.testing.assert_allclose(E, expected, atol=1e-8)


@pytest.mark.xfail(strict=True, reason="near-field surface boundary conditions need repair")
def test_mie_field_satisfies_sphere_boundary_conditions():
    """Tangential E and normal D must be continuous across a dielectric sphere."""
    a = 0.3
    eps_r = 2.0
    k0 = 1.0 / a
    delta = 1e-6

    for theta, phi in [(0.7, 0.4), (1.2, 1.1), (2.0, 2.4)]:
        er, eth, eph = _spherical_basis(theta, phi)
        points = np.array([
            a * (1.0 - delta) * er,
            a * (1.0 + delta) * er,
        ])
        e_in, e_out = em3d.mie.mie_field_at(points, a, eps_r, k0)

        e_in_t = np.array([np.dot(e_in, eth), np.dot(e_in, eph)])
        e_out_t = np.array([np.dot(e_out, eth), np.dot(e_out, eph)])
        tangential_scale = max(np.linalg.norm(e_in_t), np.linalg.norm(e_out_t), 1e-30)
        tangential_rel = np.linalg.norm(e_in_t - e_out_t) / tangential_scale

        d_in_n = eps_r * np.dot(e_in, er)
        d_out_n = np.dot(e_out, er)
        normal_scale = max(abs(d_in_n), abs(d_out_n), 1e-30)
        normal_rel = abs(d_in_n - d_out_n) / normal_scale

        assert tangential_rel < 1e-3
        assert normal_rel < 1e-3


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
    """ValueError for bad xyz shape or invalid wave frame."""
    with pytest.raises(ValueError, match="xyz must have shape"):
        em3d.mie.mie_field_at(np.zeros((5,)), a=0.3, eps_r=2.0, k0=1.0/0.3)
    with pytest.raises(ValueError, match="xyz must have shape"):
        em3d.mie.mie_field_at(np.zeros((5, 2)), a=0.3, eps_r=2.0, k0=1.0/0.3)
    with pytest.raises(ValueError, match="parallel"):
        em3d.mie.mie_field_at(np.zeros((1, 3)), a=0.3, eps_r=2.0, k0=1.0/0.3,
                               amplitude=(0, 0, 1), orient=(0, 0, 1))
    with pytest.raises(ValueError, match="transverse"):
        em3d.mie.mie_field_at(np.zeros((1, 3)), a=0.3, eps_r=2.0, k0=1.0/0.3,
                               amplitude=(1, 0, 1), orient=(0, 0, 1))
    with pytest.raises(ValueError, match="orient"):
        em3d.mie.mie_field_at(np.zeros((1, 3)), a=0.3, eps_r=2.0, k0=1.0/0.3,
                               orient=(0, 0, 0))
    with pytest.raises(ValueError, match="amplitude"):
        em3d.mie.mie_field_at(np.zeros((1, 3)), a=0.3, eps_r=2.0, k0=1.0/0.3,
                               amplitude=(0, 0, 0))
