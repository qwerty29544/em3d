import numpy as np
import pytest

from em3d.gamma0 import cross, sequential_chain, compute_circle_three_points, smallest_enclosing_circle


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


def test_circle_three_points_known():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    c = np.array([0.0, 1.0])
    cx, cy, r = compute_circle_three_points(a, b, c)
    assert abs(cx) < 1e-12 and abs(cy) < 1e-12
    assert abs(r - 1.0) < 1e-10


def test_smallest_enclosing_circle_triangle():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    cx, cy, r = smallest_enclosing_circle(pts)
    assert r < 1.5  # sanity: circle fits the triangle
    for p in pts:
        assert np.linalg.norm(p - np.array([cx, cy])) <= r + 1e-12
