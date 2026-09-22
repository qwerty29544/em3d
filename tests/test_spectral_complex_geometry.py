import numpy as np
import pytest

from em3d.spectral import (
    convex_hull_complex,
    directed_polygon_distance,
    point_in_convex_polygon,
    point_polygon_distance,
    polygon_hausdorff_distance,
)


def test_complex_hull_is_deterministic_and_excludes_interior_points():
    points = np.array([0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j, 0.5 + 0.5j, 1 + 1j])
    hull = convex_hull_complex(points)
    assert len(hull) == 4
    assert set(np.round(hull, 12)) == {0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j}


def test_directed_distance_is_asymmetric():
    small = np.array([1 + 0j, 2 + 0j, 2 + 1j, 1 + 1j])
    large = np.array([0 + 0j, 3 + 0j, 3 + 2j, 0 + 2j])
    assert directed_polygon_distance(small, large) == pytest.approx(0.0)
    assert directed_polygon_distance(large, small) > 0.0
    assert polygon_hausdorff_distance(small, large) == pytest.approx(
        directed_polygon_distance(large, small)
    )


def test_point_membership_and_distance_for_segment_and_polygon():
    segment = np.array([1 + 0j, 3 + 0j])
    assert point_in_convex_polygon(2 + 0j, segment)
    assert point_polygon_distance(2 + 1j, segment) == pytest.approx(1.0)

    square = np.array([1 + 0j, 3 + 0j, 3 + 2j, 1 + 2j])
    assert point_in_convex_polygon(2 + 1j, square)
    assert point_polygon_distance(0 + 1j, square) == pytest.approx(1.0)
