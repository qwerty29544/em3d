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
