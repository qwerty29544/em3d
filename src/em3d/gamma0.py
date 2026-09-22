from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral.complex_geometry import HullConfig, convex_hull_complex
from .spectral.localization import (
    LocalizationStatus,
    analyze_spectrum as _analyze_spectrum,
    circle_contains_origin as _circle_contains_origin,
    circle_contains_points as _circle_contains_points,
    circle_from_three_points as _circle_from_three_points,
    circle_from_two_points as _circle_from_two_points,
)


@dataclass(frozen=True)
class Gamma0Analysis:
    mu: complex
    radius: float
    rho: float
    spectrum: np.ndarray
    hull: np.ndarray
    coarse_N: tuple[int, int, int] | None = None
    matrix_shape: tuple[int, int] | None = None

    @property
    def margin(self) -> float:
        return float(abs(self.mu) - self.radius)

    def as_solver_config_kwargs(self) -> dict:
        return {"mu": self.mu, "radius": self.radius}


def cross(o, a, b) -> float:
    return float(
        (a[0] - o[0]) * (b[1] - o[1])
        - (a[1] - o[1]) * (b[0] - o[0])
    )


def sequential_chain(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points, got {values.shape}")
    if len(np.unique(values, axis=0)) < 2:
        return np.unique(values, axis=0)
    complex_points = values[:, 0] + 1j * values[:, 1]
    hull = convex_hull_complex(
        complex_points,
        config=HullConfig(rounding_decimals=None, cross_tolerance=0.0),
    )
    return np.column_stack((hull.real, hull.imag))


def mu_2points(z1: complex, z2: complex) -> complex:
    return _circle_from_two_points(z1, z2)[0]


def radius_2points(z1: complex, z2: complex) -> float:
    return _circle_from_two_points(z1, z2)[1]


def compute_circle_two_points(z1: complex, z2: complex) -> tuple:
    return _circle_from_two_points(z1, z2)


def compute_circle_three_points(
    z1: complex,
    z2: complex,
    z3: complex,
) -> tuple:
    return _circle_from_three_points(z1, z2, z3)


def circle_contains_points(
    centre: complex,
    radius: float,
    points,
    epsilon: float = 1e-8,
) -> bool:
    return _circle_contains_points(
        centre,
        radius,
        np.asarray(points, dtype=np.complex128),
        tolerance=epsilon,
    )


def circle_contains_origin(
    centre: complex,
    radius: float,
    epsilon: float = 1e-8,
) -> bool:
    return _circle_contains_origin(
        centre,
        radius,
        tolerance=epsilon,
    )


def analyze_spectrum(
    spectrum_samples: np.ndarray,
    *,
    coarse_N: tuple[int, int, int] | None = None,
    matrix_shape: tuple[int, int] | None = None,
) -> Gamma0Analysis:
    localization = _analyze_spectrum(spectrum_samples)
    if localization.status is not LocalizationStatus.OK or localization.circle is None:
        if localization.status is LocalizationStatus.ORIGIN_IN_HULL:
            raise ValueError(
                "could not find a gamma0 circle: the origin belongs to the "
                "convex spectral hull"
            )
        if localization.status is LocalizationStatus.DEGENERATE_SPECTRUM:
            raise ValueError("spectrum samples are degenerate")
        raise ValueError("could not find a gamma0 circle that excludes the origin")
    circle = localization.circle
    return Gamma0Analysis(
        mu=circle.mu,
        radius=circle.radius,
        rho=circle.q,
        spectrum=localization.spectrum,
        hull=localization.hull,
        coarse_N=coarse_N,
        matrix_shape=matrix_shape,
    )


def find_params(spectrum_samples: np.ndarray) -> dict:
    return analyze_spectrum(spectrum_samples).as_solver_config_kwargs()


def _normalize_grid_shape(coarse_N) -> tuple[int, int, int]:
    if isinstance(coarse_N, int):
        shape = (coarse_N, coarse_N, coarse_N)
    else:
        shape = tuple(int(n) for n in coarse_N)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(
            f"coarse_N must be a positive int or 3-tuple, got {coarse_N!r}"
        )
    return shape


def _nearest_indices(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    indices = np.searchsorted(source, target)
    indices = np.clip(indices, 0, len(source) - 1)
    left = np.clip(indices - 1, 0, len(source) - 1)
    choose_left = (
        np.abs(target - source[left]) <= np.abs(target - source[indices])
    )
    return np.where(choose_left, left, indices)


def _resample_eps_tensor(problem, coarse_grid) -> np.ndarray:
    backend = problem.grid.backend
    contrast = np.asarray(
        backend.to_host(problem.eps_tensor), dtype=np.complex128
    )
    ix = _nearest_indices(backend.to_host(problem.grid.x), coarse_grid.x)
    iy = _nearest_indices(backend.to_host(problem.grid.y), coarse_grid.y)
    iz = _nearest_indices(backend.to_host(problem.grid.z), coarse_grid.z)
    return contrast[:, :, ix, :, :][:, :, :, iy, :][:, :, :, :, iz]


def _eps_block_matrix(eps_tensor: np.ndarray) -> np.ndarray:
    from .dense import contrast_block_matrix

    return contrast_block_matrix(eps_tensor)


def coarse_operator_matrix(problem, coarse_N=(4, 4, 4)) -> np.ndarray:
    """Backward-compatible coarse matrix based on nearest-cell resampling.

    New reproducible spectral-transfer experiments should rebuild every grid
    from a continuous :class:`~em3d.experiments.spectral_transfer.SpectralCaseDefinition`
    instead of resampling a fine-grid tensor.
    """

    from .backend import Backend
    from .dense import dense_kernel_matrix
    from .dtypes import Precision
    from .grid import Grid

    shape = _normalize_grid_shape(coarse_N)
    coarse_backend = Backend.numpy(Precision.DOUBLE)
    coarse_grid = Grid(
        N=shape,
        L=problem.grid.L,
        center=problem.grid.center,
        backend=coarse_backend,
    )
    eps_tensor = _resample_eps_tensor(problem, coarse_grid)
    kernel = dense_kernel_matrix(coarse_grid, k=problem.k0)
    contrast = _eps_block_matrix(eps_tensor)
    return np.eye(kernel.shape[0], dtype=np.complex128) - kernel @ contrast


def estimate_from_problem(problem, coarse_N=(4, 4, 4)) -> Gamma0Analysis:
    shape = _normalize_grid_shape(coarse_N)
    matrix = coarse_operator_matrix(problem, coarse_N=shape)
    spectrum = np.linalg.eigvals(matrix)
    return analyze_spectrum(
        spectrum,
        coarse_N=shape,
        matrix_shape=matrix.shape,
    )


def find_params_from_problem(problem, coarse_N=(4, 4, 4)) -> dict:
    return estimate_from_problem(
        problem,
        coarse_N=coarse_N,
    ).as_solver_config_kwargs()
