"""γ₀ algorithm: convex hull of spectrum samples and bounding circle for the optimal iteration parameter."""
from __future__ import annotations

import numpy as np


def cross(o, a, b) -> float:
    """Signed area of the triangle (o, a, b) × 2. Positive for CCW."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def sequential_chain(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain convex hull. Input: (N, 2). Output: (H, 2) CCW, no duplicates."""
    pts = np.asarray(points, dtype=np.float64)
    pts = np.unique(pts, axis=0)  # sorts lexicographically and deduplicates
    if len(pts) < 2:
        return pts

    def build(seq):
        hull = []
        for p in seq:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(tuple(p))
        return hull

    lower = build(pts)
    upper = build(pts[::-1])
    hull_points = lower[:-1] + upper[:-1]
    if len(hull_points) < 2:
        return np.array(hull_points, dtype=np.float64)
    return np.array(hull_points, dtype=np.float64)


def compute_circle_two_points(z1: complex, z2: complex) -> tuple:
    """Circle through two points with the smaller radius (midpoint, |z2-z1|/2)."""
    centre = 0.5 * (z1 + z2)
    radius = abs(z2 - z1) / 2.0
    return centre, float(radius)


def compute_circle_three_points(z1: complex, z2: complex, z3: complex) -> tuple:
    """Circumscribed circle through three non-collinear complex points."""
    ax, ay = z1.real, z1.imag
    bx, by = z2.real, z2.imag
    cx, cy = z3.real, z3.imag
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-30:
        raise ValueError("three points are collinear")
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    centre = complex(ux, uy)
    radius = abs(centre - z1)
    return centre, float(radius)


def circle_contains_points(centre: complex, radius: float, points, epsilon: float = 1e-8) -> bool:
    pts = np.asarray(points, dtype=np.complex128)
    return bool(np.all(np.abs(pts - centre) <= radius + epsilon))


def circle_contains_origin(centre: complex, radius: float, epsilon: float = 1e-8) -> bool:
    return abs(centre) <= radius + epsilon


def find_params(spectrum_samples: np.ndarray) -> dict:
    """Compute the optimal γ₀ iteration parameter from spectrum samples.

    Returns {'mu': complex, 'radius': float}. The return dict is plug-compatible
    with SolverConfig(**find_params(samples)). Raises ValueError if samples are
    degenerate (fewer than 2 points, origin inside the resulting circle).
    """
    pts = np.asarray(spectrum_samples, dtype=np.complex128)
    if len(pts) < 2:
        raise ValueError("find_params requires at least 2 spectrum samples")

    # Convex hull of the point set in 2D real coordinates
    as_xy = np.column_stack([pts.real, pts.imag])
    hull_xy = sequential_chain(as_xy)
    hull = hull_xy[:, 0] + 1j * hull_xy[:, 1]
    if len(hull) < 2:
        raise ValueError("spectrum samples are degenerate (collinear or identical)")

    # Smallest enclosing circle among: (a) pairs of hull points (diameter), (b) triples.
    best = None  # (radius, centre)
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            c, r = compute_circle_two_points(hull[i], hull[j])
            if circle_contains_points(c, r, hull):
                if best is None or r < best[0]:
                    best = (r, c)
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            for k in range(j + 1, len(hull)):
                try:
                    c, r = compute_circle_three_points(hull[i], hull[j], hull[k])
                except ValueError:
                    continue
                if circle_contains_points(c, r, hull):
                    if best is None or r < best[0]:
                        best = (r, c)
    if best is None:
        raise ValueError("could not find a bounding circle; check spectrum samples")
    radius, mu = best
    if circle_contains_origin(mu, radius):
        raise ValueError(
            "origin lies inside (or on) the bounding circle; γ₀ is ill-defined for this spectrum"
        )
    return {"mu": mu, "radius": float(radius)}
