"""Gamma0 estimation for scalar acoustic problems."""
from __future__ import annotations

import numpy as np

from ..backend import Backend
from ..dtypes import Precision
from ..gamma0 import analyze_spectrum
from ..grid import Grid
from .dense import H_scalar_matrix
from .problem import AcousticProblem, make_acoustic_problem


def _normalize_grid_shape(coarse_N) -> tuple[int, int, int]:
    if isinstance(coarse_N, int):
        shape = (int(coarse_N), int(coarse_N), int(coarse_N))
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


def _resample_eta(problem: AcousticProblem, coarse_grid: Grid) -> np.ndarray:
    be = problem.grid.backend
    eta = np.asarray(be.to_host(problem.eta), dtype=np.complex128)
    ix = _nearest_indices(be.to_host(problem.grid.x), coarse_grid.x)
    iy = _nearest_indices(be.to_host(problem.grid.y), coarse_grid.y)
    iz = _nearest_indices(be.to_host(problem.grid.z), coarse_grid.z)
    return eta[ix, :, :][:, iy, :][:, :, iz]


def coarse_operator_matrix(problem: AcousticProblem, coarse_N=(4, 4, 4)) -> np.ndarray:
    """Build dense scalar H for a nearest-neighbour coarse acoustic problem."""
    shape = _normalize_grid_shape(coarse_N)
    coarse_backend = Backend.numpy(Precision.DOUBLE)
    coarse_grid = Grid(N=shape, L=problem.grid.L, center=problem.grid.center, backend=coarse_backend)
    eta = _resample_eta(problem, coarse_grid)
    coarse_problem = make_acoustic_problem(coarse_grid, eta, k0=problem.k0)
    return H_scalar_matrix(coarse_problem)


def estimate_from_problem(problem: AcousticProblem, coarse_N=(4, 4, 4)):
    """Estimate gamma0 from dense spectrum of a coarse scalar acoustic problem."""
    shape = _normalize_grid_shape(coarse_N)
    H = coarse_operator_matrix(problem, coarse_N=shape)
    spectrum = np.linalg.eigvals(H)
    return analyze_spectrum(spectrum, coarse_N=shape, matrix_shape=H.shape)


def find_params_from_problem(problem: AcousticProblem, coarse_N=(4, 4, 4)) -> dict:
    """Return SolverConfig-compatible gamma0 parameters."""
    return estimate_from_problem(problem, coarse_N=coarse_N).as_solver_config_kwargs()
