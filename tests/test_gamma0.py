import numpy as np
import pytest

from em3d.gamma0 import (
    cross,
    sequential_chain,
    compute_circle_two_points,
    compute_circle_three_points,
    circle_contains_points,
    circle_contains_origin,
    find_params,
)


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
