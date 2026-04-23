import numpy as np
import pytest

from em3d.grid import Grid
from em3d.operator import prep_coeffs_em3d
from em3d.refraction import cylinder_refraction, apply_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.operator import Operator


def test_prep_coeffs_shape_and_dtype(backend_numpy_double):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    coeffs = prep_coeffs_em3d(grid, k=1.0, volume=grid.dv * 64)
    # Π₂ doubling: FFT tensor on (2Nx, 2Ny, 2Nz) with 3×3 block structure
    assert coeffs.shape == (3, 3) + tuple(2 * n for n in grid.N)
    assert coeffs.dtype == backend_numpy_double.complex_dtype


def _toy_problem(backend, N=(4, 4, 4)):
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend)
    scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.3, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    # approximate Q volume as all cells (toy example)
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=volume)


def test_operator_matvec_shape(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double)
    op = Operator(problem)
    u = backend_numpy_double.zeros((3,) + problem.grid.N, kind="complex")
    u[0, 0, 0, 0] = 1.0
    y = op.matvec(u)
    assert y.shape == (3,) + problem.grid.N
    assert y.dtype == backend_numpy_double.complex_dtype


def test_operator_rmatvec_shape(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double)
    op = Operator(problem)
    u = backend_numpy_double.zeros((3,) + problem.grid.N, kind="complex")
    u[0, 0, 0, 0] = 1.0
    y = op.rmatvec(u)
    assert y.shape == u.shape


def test_problem_rejects_wrong_dtype(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1, 1, 1), center=(0, 0, 0), backend=backend_numpy_double)
    eta = np.zeros((3, 3, 2, 2, 2), dtype=np.complex64)  # wrong precision
    wave = np.zeros((3, 2, 2, 2), dtype=np.complex128)
    with pytest.raises(TypeError):
        Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 8)


def test_problem_rejects_wrong_shape(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1, 1, 1), center=(0, 0, 0), backend=backend_numpy_double)
    eta = np.zeros((3, 3, 2, 2, 2), dtype=np.complex128)
    wave = np.zeros((3, 4, 2, 2), dtype=np.complex128)  # wrong shape
    with pytest.raises(ValueError):
        Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 8)
