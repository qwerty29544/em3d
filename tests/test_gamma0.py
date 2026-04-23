import numpy as np
import pytest

from em3d.gamma0 import cross, sequential_chain


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
