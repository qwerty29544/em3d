import numpy as np
import pytest
 
from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.grid import Grid
from em3d.refraction import apply_refraction, ellipsis_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.farfield import scatter_integral, rcs, rcs_plane
 
def _be():
    return Backend.numpy(Precision.DOUBLE)
 
def _make_problem(N=(4, 4, 4), eps_real=2.0, eps_imag=0.0, k0=1.0):
    be = _be()
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=eps_real, eps_imag=eps_imag,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    wave = flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=k0, volume=volume)
 
@pytest.mark.parametrize("method", ["direct", "fft"])
def test_zero_contrast(method):
 
    be = _be()
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = be.zeros((3, 3) + grid.N, kind="complex")
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 64)
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = scatter_integral(wave, problem, directions, method=method)
    np.testing.assert_allclose(F, 0.0, atol=1e-14)
 
def test_rcs_nonnegative():
 
    problem = _make_problem()
    direction = np.array([1.0, 0.0, 0.0])
    result = rcs(problem.wave, problem, direction)
    assert result >= 0.0
 
def test_scatter_integral_shape_validation_does_not_force_numpy():
 
    problem = _make_problem()
 
    class NoImplicitArray:
        shape = (2,) + problem.grid.N
 
        def __array__(self, dtype=None):
            raise AssertionError("scatter_integral should not coerce before shape validation")
 
    with pytest.raises(ValueError, match="u must have shape"):
        scatter_integral(NoImplicitArray(), problem, np.array([[1.0, 0.0, 0.0]]), method="direct")
 
def test_single_cell_analytic():
 
    be = _be()
    grid = Grid(N=(1, 1, 1), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
 
    eta_val = complex(1.5, 0.1)
    scalar = np.full(grid.N, eta_val, dtype=np.complex128)
    eta = apply_refraction(grid, scalar_eta=scalar)
    u = np.zeros((3, 1, 1, 1), dtype=np.complex128)
    u[0, 0, 0, 0] = 1.0 + 0.5j
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv)
    direction = np.array([1.0, 0.0, 0.0])
    F = scatter_integral(u, problem, direction[np.newaxis, :])[0]
 
    F_analytic = np.array([eta_val * (1.0 + 0.5j), 0.0, 0.0])
    np.testing.assert_allclose(F, F_analytic, rtol=1e-12)
 
def test_fft_vs_direct_agreement():
 
    be = _be()
    grid = Grid(N=(8, 8, 8), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 512)
    rng = np.random.default_rng(42)
    u = (rng.standard_normal((3, 8, 8, 8))
         + 1j * rng.standard_normal((3, 8, 8, 8))).astype(np.complex128)
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ])
    F_direct = scatter_integral(u, problem, directions, method="direct")
    F_fft    = scatter_integral(u, problem, directions, method="fft")
    np.testing.assert_allclose(F_fft, F_direct, atol=1e-4,
                                err_msg="FFT and direct backends disagree")
 
def test_rcs_plane_shape():
 
    problem = _make_problem(k0=0.5)
    phi, sigma = rcs_plane(problem.wave, problem, n_phi=12, plane="xy")
    assert phi.shape == (12,)
    assert sigma.shape == (12,)
 
def test_rcs_plane_symmetry():
 
    be = _be()
    N = (8, 8, 8)
    grid = Grid(N=N, L=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.4, 0.4, 0.4),
    )
    wave = flat_wave_vec(grid, k=0.1, orient=(0, 0, 1), amplitude=(1, 0, 0))
    u = np.zeros((3,) + N, dtype=np.complex128)
    u[0] = 1.0
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.1,
                      volume=grid.dv * int(np.prod(N)))
    n_phi = 24
    phi, sigma = rcs_plane(u, problem, n_phi=n_phi, plane="xy")
    half = n_phi // 2
    np.testing.assert_allclose(
        sigma[:half], sigma[half:], rtol=1e-10, atol=1e-35,
        err_msg="RCS not symmetric: sigma(phi) != sigma(phi+pi)",
    )

@pytest.mark.gpu
def test_direct_farfield_cuda_matches_numpy():
    import cupy as cp

    cpu_problem = _make_problem(N=(4, 4, 4), eps_real=2.0, k0=0.7)
    cpu_field = np.asarray(cpu_problem.wave)
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
    ])
    expected = scatter_integral(cpu_field, cpu_problem, directions, method="direct")

    be = Backend.cupy(Precision.DOUBLE)
    grid = Grid(
        N=cpu_problem.grid.N,
        L=cpu_problem.grid.L,
        center=cpu_problem.grid.center,
        backend=be,
    )
    gpu_problem = Problem(
        grid=grid,
        eps_tensor=cp.asarray(cpu_problem.eps_tensor),
        wave=cp.asarray(cpu_problem.wave),
        k0=cpu_problem.k0,
        volume=cpu_problem.volume,
    )
    actual = scatter_integral(
        gpu_problem.wave,
        gpu_problem,
        directions,
        method="direct",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
