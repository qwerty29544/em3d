from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HullConfig:
    rounding_decimals: int | None = 13
    cross_tolerance: float = 1e-13
    containment_tolerance: float = 1e-12


def cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull_complex(
    points: np.ndarray,
    *,
    config: HullConfig = HullConfig(),
) -> np.ndarray:
    """Deterministic monotone-chain hull of complex points."""

    spectrum = np.asarray(points, dtype=np.complex128).reshape(-1)
    xy = np.column_stack((spectrum.real, spectrum.imag))
    if config.rounding_decimals is not None:
        xy = np.round(xy, decimals=config.rounding_decimals)
    xy = np.unique(xy, axis=0)
    if len(xy) < 2:
        raise ValueError("at least two distinct points are required")

    ordered = [tuple(point) for point in xy[np.lexsort((xy[:, 1], xy[:, 0]))]]
    if len(ordered) == 2:
        values = np.asarray(ordered, dtype=np.float64)
        return values[:, 0] + 1j * values[:, 1]

    def build(sequence):
        hull: list[tuple[float, float]] = []
        for point in sequence:
            while (
                len(hull) >= 2
                and cross(hull[-2], hull[-1], point) <= config.cross_tolerance
            ):
                hull.pop()
            hull.append(point)
        return hull

    lower = build(ordered)
    upper = build(reversed(ordered))
    hull_points = lower[:-1] + upper[:-1]
    if len(hull_points) < 2:
        hull_points = [ordered[0], ordered[-1]]
    values = np.asarray(hull_points, dtype=np.float64)
    return values[:, 0] + 1j * values[:, 1]


def point_segment_distance(z: complex, a: complex, b: complex) -> float:
    point = np.asarray((complex(z).real, complex(z).imag), dtype=np.float64)
    start = np.asarray((complex(a).real, complex(a).imag), dtype=np.float64)
    end = np.asarray((complex(b).real, complex(b).imag), dtype=np.float64)
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, direction) / denominator)
    fraction = min(1.0, max(0.0, fraction))
    return float(np.linalg.norm(point - (start + fraction * direction)))


def polygon_orientation(polygon: np.ndarray) -> float:
    values = np.asarray(polygon, dtype=np.complex128).reshape(-1)
    return 0.5 * float(
        np.sum(values.real * np.roll(values.imag, -1) - values.imag * np.roll(values.real, -1))
    )


def point_in_convex_polygon(
    z: complex,
    polygon: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> bool:
    values = np.asarray(polygon, dtype=np.complex128).reshape(-1)
    if len(values) == 0:
        return False
    if len(values) == 1:
        return abs(complex(z) - values[0]) <= tolerance
    if len(values) == 2:
        return point_segment_distance(z, values[0], values[1]) <= tolerance

    orientation = np.sign(polygon_orientation(values))
    if orientation == 0.0:
        orientation = 1.0
    point = np.asarray((complex(z).real, complex(z).imag), dtype=np.float64)
    for start, end in zip(values, np.roll(values, -1), strict=True):
        a = np.asarray((start.real, start.imag), dtype=np.float64)
        b = np.asarray((end.real, end.imag), dtype=np.float64)
        edge = b - a
        relative = point - a
        value = edge[0] * relative[1] - edge[1] * relative[0]
        if orientation * value < -tolerance:
            return False
    return True


def point_polygon_distance(z: complex, polygon: np.ndarray) -> float:
    values = np.asarray(polygon, dtype=np.complex128).reshape(-1)
    if len(values) == 0:
        raise ValueError("polygon must be non-empty")
    if point_in_convex_polygon(z, values):
        return 0.0
    if len(values) == 1:
        return float(abs(complex(z) - values[0]))
    return min(
        point_segment_distance(z, start, end)
        for start, end in zip(values, np.roll(values, -1), strict=True)
    )


def directed_polygon_distance(
    source: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Directed distance from ``source`` to ``reference``.

    For convex polygons the maximum of the distance-to-set function is attained
    at a vertex, so evaluating source vertices is sufficient.
    """

    source_values = np.asarray(source, dtype=np.complex128).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if len(source_values) == 0 or len(reference_values) == 0:
        raise ValueError("both polygons must be non-empty")
    return float(
        max(point_polygon_distance(value, reference_values) for value in source_values)
    )


def polygon_hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(directed_polygon_distance(a, b), directed_polygon_distance(b, a))
