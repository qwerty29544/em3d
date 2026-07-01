"""gamma0 geometry and coarse-spectrum analysis for SIM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Gamma0Analysis:
    """Complete gamma0 analysis result for research workflows."""

    mu: complex
    radius: float
    rho: float
    spectrum: np.ndarray
    hull: np.ndarray
    coarse_N: tuple[int, int, int] | None = None
    matrix_shape: tuple[int, int] | None = None

    def as_solver_config_kwargs(self) -> dict:
        return {"mu": self.mu, "radius": self.radius}


def cross(o, a, b) -> float:
    """Signed area of the triangle (o, a, b) x 2. Positive for CCW."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def sequential_chain(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain convex hull. Input: (N, 2). Output: (H, 2)."""
    pts = np.asarray(points, dtype=np.float64)
    pts = np.unique(pts, axis=0)
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


def mu_2points(z1: complex, z2: complex) -> complex:
    """Step-A gamma0 centre for the circle through two spectrum points."""
    prod = z1 * np.conj(z2)
    denom = 2.0 * (abs(prod) + prod.real)
    if abs(denom) < 1e-30:
        raise ValueError("two-point gamma0 circle is undefined for this pair")
    midpoint = 0.5 * (z1 + z2)
    correction = 1j * (prod.imag * (z2 - z1)) / denom
    return complex(midpoint + correction)


def radius_2points(z1: complex, z2: complex) -> float:
    """Step-A gamma0 radius for the circle through two spectrum points."""
    prod = np.conj(z1) * z2
    denom = 2.0 * (abs(prod) + prod.real)
    if abs(denom) < 1e-30:
        raise ValueError("two-point gamma0 circle is undefined for this pair")
    value = (abs(z1 - z2) ** 2) * abs(prod) / denom
    return float(np.sqrt(max(value.real, 0.0)))


def compute_circle_two_points(z1: complex, z2: complex) -> tuple:
    """Gamma0 Step-A circle for a boundary segment."""
    return mu_2points(z1, z2), radius_2points(z1, z2)


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


def _candidate_rho(centre: complex, radius: float) -> float:
    centre_abs = abs(centre)
    if centre_abs <= 1e-30:
        return float("inf")
    return float(radius / centre_abs)


def _valid_candidate(centre: complex, radius: float, hull: np.ndarray) -> bool:
    return circle_contains_points(centre, radius, hull) and not circle_contains_origin(centre, radius)


def _iter_pairs(points: np.ndarray) -> Iterable[tuple[complex, complex]]:
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            yield points[i], points[j]


def _iter_triples(points: np.ndarray) -> Iterable[tuple[complex, complex, complex]]:
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                yield points[i], points[j], points[k]


def analyze_spectrum(
    spectrum_samples: np.ndarray,
    *,
    coarse_N: tuple[int, int, int] | None = None,
    matrix_shape: tuple[int, int] | None = None,
) -> Gamma0Analysis:
    """Compute gamma0 parameters and diagnostic geometry from spectrum samples."""
    spectrum = np.asarray(spectrum_samples, dtype=np.complex128).reshape(-1)
    if len(spectrum) < 2:
        raise ValueError("analyze_spectrum requires at least 2 spectrum samples")

    as_xy = np.column_stack([spectrum.real, spectrum.imag])
    hull_xy = sequential_chain(as_xy)
    hull = hull_xy[:, 0] + 1j * hull_xy[:, 1]
    if len(hull) < 2:
        raise ValueError("spectrum samples are degenerate")

    best: tuple[float, float, complex] | None = None

    for z1, z2 in _iter_pairs(hull):
        try:
            centre, radius = compute_circle_two_points(z1, z2)
        except ValueError:
            continue
        if _valid_candidate(centre, radius, hull):
            rho = _candidate_rho(centre, radius)
            if best is None or rho < best[0]:
                best = (rho, radius, centre)

    for z1, z2, z3 in _iter_triples(hull):
        try:
            centre, radius = compute_circle_three_points(z1, z2, z3)
        except ValueError:
            continue
        if _valid_candidate(centre, radius, hull):
            rho = _candidate_rho(centre, radius)
            if best is None or rho < best[0]:
                best = (rho, radius, centre)

    if best is None:
        raise ValueError("could not find a gamma0 circle that excludes the origin")

    rho, radius, mu = best
    return Gamma0Analysis(
        mu=mu,
        radius=float(radius),
        rho=float(rho),
        spectrum=spectrum,
        hull=hull,
        coarse_N=coarse_N,
        matrix_shape=matrix_shape,
    )


def find_params(spectrum_samples: np.ndarray) -> dict:
    """Return {'mu': complex, 'radius': float} for SolverConfig compatibility."""
    return analyze_spectrum(spectrum_samples).as_solver_config_kwargs()


def _normalize_grid_shape(coarse_N) -> tuple[int, int, int]:
    if isinstance(coarse_N, int):
        shape = (coarse_N, coarse_N, coarse_N)
    else:
        shape = tuple(int(n) for n in coarse_N)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"coarse_N must be a positive int or 3-tuple, got {coarse_N!r}")
    return shape


def _nearest_indices(source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    source = np.asarray(source_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    indices = np.searchsorted(source, target)
    indices = np.clip(indices, 0, len(source) - 1)
    left = np.clip(indices - 1, 0, len(source) - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(target - source[indices])
    return np.where(choose_left, left, indices)


def _resample_eps_tensor(problem, coarse_grid) -> np.ndarray:
    be = problem.grid.backend
    eps = np.asarray(be.to_host(problem.eps_tensor), dtype=np.complex128)
    ix = _nearest_indices(be.to_host(problem.grid.x), coarse_grid.x)
    iy = _nearest_indices(be.to_host(problem.grid.y), coarse_grid.y)
    iz = _nearest_indices(be.to_host(problem.grid.z), coarse_grid.z)
    return eps[:, :, ix, :, :][:, :, :, iy, :][:, :, :, :, iz]


def _eps_block_matrix(eps_tensor: np.ndarray) -> np.ndarray:
    Nx, Ny, Nz = eps_tensor.shape[2:]
    n_cells = Nx * Ny * Nz
    out = np.zeros((3 * n_cells, 3 * n_cells), dtype=np.complex128)
    cell = 0
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                out[3 * cell : 3 * cell + 3, 3 * cell : 3 * cell + 3] = eps_tensor[:, :, ix, iy, iz]
                cell += 1
    return out


def coarse_operator_matrix(problem, coarse_N=(4, 4, 4)) -> np.ndarray:
    """Build dense H = I - B eta for a nearest-neighbour coarse version of problem."""
    from .backend import Backend
    from .dense import B_operator_matrix
    from .dtypes import Precision
    from .grid import Grid

    shape = _normalize_grid_shape(coarse_N)
    coarse_backend = Backend.numpy(Precision.DOUBLE)
    coarse_grid = Grid(N=shape, L=problem.grid.L, center=problem.grid.center, backend=coarse_backend)
    eps_tensor = _resample_eps_tensor(problem, coarse_grid)
    B = B_operator_matrix(coarse_grid, k=problem.k0, volume=problem.volume)
    eta = _eps_block_matrix(eps_tensor)
    dof = B.shape[0]
    return np.eye(dof, dtype=np.complex128) - B @ eta


def estimate_from_problem(problem, coarse_N=(4, 4, 4)) -> Gamma0Analysis:
    """Estimate gamma0 from the dense spectrum of a coarse-grid problem."""
    shape = _normalize_grid_shape(coarse_N)
    H = coarse_operator_matrix(problem, coarse_N=shape)
    spectrum = np.linalg.eigvals(H)
    return analyze_spectrum(spectrum, coarse_N=shape, matrix_shape=H.shape)


def find_params_from_problem(problem, coarse_N=(4, 4, 4)) -> dict:
    """Return SolverConfig-compatible gamma0 parameters estimated from problem."""
    return estimate_from_problem(problem, coarse_N=coarse_N).as_solver_config_kwargs()
