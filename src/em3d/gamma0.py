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


def compute_circle_three_points(a, b, c):
    """Circumscribed circle of triangle (a, b, c). Returns (cx, cy, radius)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-30:
        raise ValueError("Points are colinear — no circumscribed circle")
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    r = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
    return ux, uy, r


def smallest_enclosing_circle(points: np.ndarray):
    """Welzl-inspired smallest enclosing circle for a set of 2D points.

    Returns (cx, cy, radius).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("Empty point set")
    if n == 1:
        return pts[0, 0], pts[0, 1], 0.0
    if n == 2:
        cx = (pts[0, 0] + pts[1, 0]) / 2
        cy = (pts[0, 1] + pts[1, 1]) / 2
        r = np.linalg.norm(pts[0] - pts[1]) / 2
        return cx, cy, r

    # Iterative Welzl (Welzl 1991, randomised)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    pts_shuffled = pts[order]

    def in_circle(cx, cy, r, p):
        return np.linalg.norm(p - np.array([cx, cy])) <= r + 1e-12

    def circle_from_1(p):
        return p[0], p[1], 0.0

    def circle_from_2(p, q):
        cx = (p[0] + q[0]) / 2
        cy = (p[1] + q[1]) / 2
        return cx, cy, np.linalg.norm(p - q) / 2

    def circle_from_3(p, q, r_pt):
        try:
            return compute_circle_three_points(p, q, r_pt)
        except ValueError:
            return circle_from_2(p, r_pt) if np.linalg.norm(p - r_pt) > np.linalg.norm(q - r_pt) else circle_from_2(q, r_pt)

    cx, cy, r = circle_from_1(pts_shuffled[0])
    for i in range(1, n):
        p = pts_shuffled[i]
        if not in_circle(cx, cy, r, p):
            cx, cy, r = circle_from_1(p)
            for j in range(i):
                q = pts_shuffled[j]
                if not in_circle(cx, cy, r, q):
                    cx, cy, r = circle_from_2(p, q)
                    for k in range(j):
                        s = pts_shuffled[k]
                        if not in_circle(cx, cy, r, s):
                            cx, cy, r = circle_from_3(p, q, s)
    return cx, cy, r
