import numpy as np
import pytest

from em3d.gamma0 import (
    analyze_spectrum,
    cross,
    sequential_chain,
    compute_circle_two_points,
    compute_circle_three_points,
    circle_contains_points,
    circle_contains_origin,
    coarse_operator_matrix,
    estimate_from_problem,
    find_params,
)
from em3d.grid import Grid
from em3d.problem import Problem
from em3d.refraction import cylinder_refraction
from em3d.wave import flat_wave_vec


def test_cross_positive():
    o = np.array([0.0, 0.0])
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cross(o, a, b) > 0


def test_cross_negative():
    o = np.array([0.0, 0.0])
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 0.0])
    assert cross(o, a, b) < 0


def test_sequential_chain_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
    hull = sequential_chain(pts)
    # hull is the outer square
    assert len(hull) == 4
    coords = sorted((float(p[0]), float(p[1])) for p in hull)
    assert coords == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


def test_sequential_chain_colinear():
    pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    hull = sequential_chain(pts)
    # colinear points — hull should contain only the endpoints
    assert len(hull) == 2


def test_circle_two_points_midpoint():
    p1 = 1.0 + 0j
    p2 = 3.0 + 0j
    centre, radius = compute_circle_two_points(p1, p2)
    assert abs(centre - 2.0) < 1e-12
    assert abs(radius - 1.0) < 1e-12


def test_circle_two_points_uses_gamma0_visible_angle_formula():
    centre, radius = compute_circle_two_points(1.0 + 1.0j, 1.0 - 1.0j)
    assert abs(centre - 2.0) < 1e-12
    assert abs(radius - np.sqrt(2.0)) < 1e-12


def test_circle_three_points_unit_circle():
    p1 = 1.0 + 0j
    p2 = -1.0 + 0j
    p3 = 0.0 + 1j
    centre, radius = compute_circle_three_points(p1, p2, p3)
    assert abs(centre) < 1e-12
    assert abs(radius - 1.0) < 1e-12


def test_circle_contains_points_inside():
    centre = 0.0 + 0.0j
    radius = 2.0
    pts = np.array([1.0 + 1.0j, -1.0 + 0.5j])
    assert circle_contains_points(centre, radius, pts)


def test_circle_contains_points_outside():
    centre = 0.0 + 0.0j
    radius = 1.0
    pts = np.array([1.5 + 0.0j])
    assert not circle_contains_points(centre, radius, pts)


def test_circle_contains_origin_true():
    assert circle_contains_origin(centre=0.5 + 0.0j, radius=1.0) is True


def test_circle_contains_origin_false():
    assert circle_contains_origin(centre=2.0 + 0.0j, radius=1.0) is False


def test_find_params_simple_case():
    # spectrum along the positive real axis from 1 to 3 → optimal μ = 2, radius = 1
    samples = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    result = find_params(samples)
    assert abs(result["mu"] - 2.0) < 1e-6
    assert abs(result["radius"] - 1.0) < 1e-6
    # result is plug-compatible with SolverConfig(**result)
    assert set(result.keys()) == {"mu", "radius"}


def test_find_params_rejects_origin_inside_hull():
    # if samples straddle the origin, γ₀ is ill-defined
    samples = np.array([-1.0 + 0j, 1.0 + 0j, 0.0 + 1j])
    with pytest.raises(ValueError):
        find_params(samples)


def test_find_params_requires_at_least_two():
    with pytest.raises(ValueError):
        find_params(np.array([1.0 + 0j]))


def test_analyze_spectrum_returns_hull_and_rho():
    samples = np.array([2.0 + 0.0j, 3.0 + 1.0j, 4.0 + 0.0j, 3.0 + 0.25j])
    analysis = analyze_spectrum(samples)
    assert analysis.spectrum.shape == samples.shape
    assert len(analysis.hull) == 3
    assert abs(analysis.rho - analysis.radius / abs(analysis.mu)) < 1e-12
    assert circle_contains_points(analysis.mu, analysis.radius, samples)
    assert not circle_contains_origin(analysis.mu, analysis.radius)


def _gamma0_problem(backend, N=(3, 3, 3)):
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend)
    eta = cylinder_refraction(grid, eps_real=1.5, eps_imag=0.1, radius=1.0, axis="z")
    wave = flat_wave_vec(grid, k=0.75, orient=(0, 0, 1), amplitude=(1, 0, 0))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.75, volume=grid.dv * int(np.prod(N)))


def test_coarse_operator_matrix_shape(backend_numpy_double):
    problem = _gamma0_problem(backend_numpy_double, N=(3, 3, 3))
    H = coarse_operator_matrix(problem, coarse_N=(2, 2, 2))
    assert H.shape == (24, 24)
    assert H.dtype == np.complex128


def test_coarse_operator_matrix_uses_original_operator_sign(backend_numpy_double):
    """Gamma0 must analyze the same H = I - B eta operator used by Operator.matvec."""
    from em3d.dense import B_operator_matrix

    problem = _gamma0_problem(backend_numpy_double, N=(2, 2, 2))
    H = coarse_operator_matrix(problem, coarse_N=(2, 2, 2))
    B = B_operator_matrix(problem.grid, k=problem.k0, volume=problem.volume)
    eta = np.zeros_like(B)
    cell = 0
    for ix in range(problem.grid.N[0]):
        for iy in range(problem.grid.N[1]):
            for iz in range(problem.grid.N[2]):
                eta[3 * cell : 3 * cell + 3, 3 * cell : 3 * cell + 3] = problem.eps_tensor[:, :, ix, iy, iz]
                cell += 1
    expected = np.eye(B.shape[0], dtype=np.complex128) - B @ eta
    np.testing.assert_allclose(H, expected, rtol=1e-12, atol=1e-12)


def test_estimate_from_problem_returns_solver_params(backend_numpy_double):
    problem = _gamma0_problem(backend_numpy_double, N=(3, 3, 3))
    analysis = estimate_from_problem(problem, coarse_N=(2, 2, 2))
    assert analysis.spectrum.shape == (24,)
    assert analysis.coarse_N == (2, 2, 2)
    assert analysis.matrix_shape == (24, 24)
    assert analysis.as_solver_config_kwargs() == {"mu": analysis.mu, "radius": analysis.radius}
