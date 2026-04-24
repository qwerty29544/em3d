import numpy as np
import pytest

from em3d.grid import Grid
from em3d.refraction import cylinder_refraction, apply_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.operator import Operator
from em3d.solvers.base import SolverConfig, SolverResult
from em3d.solvers.sim import SIM


def test_solver_config_defaults():
    cfg = SolverConfig(max_iter=100, rtol=1e-6)
    assert cfg.max_iter == 100
    assert cfg.rtol == 1e-6
    assert cfg.log is False


def test_solver_result_fields():
    u = np.zeros(4, dtype=np.complex128)
    res = SolverResult(u=u, iterations=5, residual_history=[1.0, 0.5, 0.1], converged=True)
    assert res.iterations == 5
    assert res.converged
    assert res.residual_history[-1] == 0.1


def _toy_problem_for_solver(be):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    scalar = cylinder_refraction(grid, eps_real=1.1, eps_imag=0.0, radius=0.2, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=0.5, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * 64
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.5, volume=volume)


def test_sim_converges_backend_agnostic(backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    # construct rhs from a known u_true so we verify convergence against a ground truth
    rng = np.random.default_rng(0)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.1
    )
    rhs = op.matvec(u_true)
    cfg = SolverConfig(max_iter=500, rtol=1e-8, mu=complex(1.0), radius=0.05)
    result = SIM(cfg).solve(op, rhs)
    assert result.converged, f"SIM did not converge, residuals: {result.residual_history[-5:]}"
    err = np.linalg.norm(np.asarray(result.u) - np.asarray(u_true)) / np.linalg.norm(np.asarray(u_true))
    assert err < 1e-6, f"SIM reconstructed u to {err:.2e}, expected < 1e-6"
