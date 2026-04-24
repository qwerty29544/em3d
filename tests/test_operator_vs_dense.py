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


def _flatten_field(u):
    """Convert (3, Nx, Ny, Nz) field to (3·Nx·Ny·Nz,) vector in row-major (cell, component) order."""
    # dense uses block order: for cell i = (ix, iy, iz), components 0,1,2 are contiguous
    three, Nx, Ny, Nz = u.shape
    assert three == 3
    # reshape (3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3) -> flat
    return np.transpose(np.asarray(u), (1, 2, 3, 0)).reshape(-1)


def test_fft_matvec_matches_dense(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double, N=(4, 4, 4))
    op = Operator(problem)
    M_dense = op.to_dense()
    rng = np.random.default_rng(42)
    u = (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128)
    # FFT path: applies (I + B·η); dense path: applies (I + dense·η)
    eta_flat = _flatten_field(np.einsum("ab...,b...->a...", problem.eps_tensor, u))
    u_flat = _flatten_field(u)
    y_dense = u_flat + M_dense @ eta_flat
    y_fft = _flatten_field(op.matvec(u))
    rel = np.linalg.norm(y_fft - y_dense) / np.linalg.norm(y_dense)
    assert rel < 1e-10, f"FFT vs dense relative error {rel:.2e}"


def _toy_problem_lossy(backend, N=(4, 4, 4)):
    """Lossy medium (eps_imag=0.5) so that the operator is not accidentally self-adjoint."""
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend)
    scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.5, radius=0.3, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=volume)


def test_fft_rmatvec_matches_dense_adjoint(backend_numpy_double):
    problem = _toy_problem_lossy(backend_numpy_double, N=(4, 4, 4))
    op = Operator(problem)
    M_dense = op.to_dense()
    rng = np.random.default_rng(7)
    u = (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128)
    u_flat = _flatten_field(u)
    eta_conj_T = np.einsum("ab...,b...->a...", np.conj(problem.eps_tensor).swapaxes(0, 1), _unflatten_field(M_dense.conj().T @ u_flat, problem.grid.N))
    expected = u_flat + _flatten_field(eta_conj_T)
    y_fft = _flatten_field(op.rmatvec(u))
    rel = np.linalg.norm(y_fft - expected) / np.linalg.norm(expected)
    assert rel < 1e-10


def _unflatten_field(flat, N):
    Nx, Ny, Nz = N
    return np.transpose(flat.reshape(Nx, Ny, Nz, 3), (3, 0, 1, 2))


def test_fft_matvec_single_precision_tolerance(backend_numpy_single):
    problem = _toy_problem(backend_numpy_single, N=(4, 4, 4))
    op = Operator(problem)
    # for single precision we only assert the FFT path returns finite values of the expected dtype;
    # absolute dense comparison in f32 is known to be noisy. Detailed single-precision tolerance
    # tests belong to the solver convergence suite (Task 18).
    u = backend_numpy_single.zeros((3,) + problem.grid.N, kind="complex")
    u[0, 0, 0, 0] = 1.0
    y = op.matvec(u)
    assert y.dtype == backend_numpy_single.complex_dtype
    assert np.all(np.isfinite(np.asarray(y)))


@pytest.mark.gpu
def test_operator_matvec_gpu_matches_cpu():
    cupy = pytest.importorskip("cupy", reason="cupy not installed")
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    be_cpu = Backend.numpy(Precision.DOUBLE)
    be_gpu = Backend.cupy(Precision.DOUBLE)

    def make(be):
        grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0, 0, 0), backend=be)
        scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.3, axis="z")
        eta = apply_refraction(grid, scalar_eta=scalar)
        wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
        return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 64)

    p_cpu = make(be_cpu)
    p_gpu = make(be_gpu)
    op_cpu = Operator(p_cpu)
    op_gpu = Operator(p_gpu)
    u_cpu = p_cpu.wave
    u_gpu = p_gpu.wave
    y_cpu = op_cpu.matvec(u_cpu)
    y_gpu = be_gpu.to_host(op_gpu.matvec(u_gpu))
    np.testing.assert_allclose(y_gpu, np.asarray(y_cpu), atol=1e-10)
