import numpy as np
import pytest

from em3d.grid import Grid
from em3d.refraction import cylinder_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.operator import Operator
from em3d.solvers.base import SolverConfig, SolverResult
from em3d.solvers.sim import SIM
from em3d.solvers.bicgstab import BiCGStab
from em3d.solvers.twostep import TwoStep


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
    eta = cylinder_refraction(grid, eps_real=1.1, eps_imag=0.0, radius=0.2, axis="z")
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


def test_bicgstab_converges(backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    rng = np.random.default_rng(1)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.1
    )
    rhs = op.matvec(u_true)
    cfg = SolverConfig(max_iter=200, rtol=1e-8)
    result = BiCGStab(cfg).solve(op, rhs)
    assert result.converged, f"BiCGStab did not converge, residuals: {result.residual_history[-5:]}"
    err = np.linalg.norm(np.asarray(result.u) - np.asarray(u_true)) / np.linalg.norm(np.asarray(u_true))
    assert err < 1e-6


def test_twostep_converges(backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    rng = np.random.default_rng(2)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.1
    )
    rhs = op.matvec(u_true)
    cfg = SolverConfig(max_iter=300, rtol=1e-8)
    result = TwoStep(cfg).solve(op, rhs)
    assert result.converged, f"TwoStep did not converge, residuals: {result.residual_history[-5:]}"
    err = np.linalg.norm(np.asarray(result.u) - np.asarray(u_true)) / np.linalg.norm(np.asarray(u_true))
    assert err < 1e-6


SOLVERS = [
    ("SIM", lambda: SIM(SolverConfig(max_iter=500, rtol=1e-6, mu=complex(1.0), radius=0.05))),
    ("BiCGStab", lambda: BiCGStab(SolverConfig(max_iter=200, rtol=1e-6))),
    ("TwoStep", lambda: TwoStep(SolverConfig(max_iter=300, rtol=1e-6))),
]


@pytest.mark.parametrize("solver_name,solver_factory", SOLVERS)
def test_solvers_converge_double(solver_name, solver_factory, backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    rng = np.random.default_rng(hash(solver_name) & 0xFFFF)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.05
    )
    rhs = op.matvec(u_true)
    result = solver_factory().solve(op, rhs)
    assert result.converged, f"{solver_name} did not converge"


@pytest.mark.parametrize("solver_name,solver_factory", SOLVERS)
def test_solvers_converge_single(solver_name, solver_factory, backend_numpy_single):
    problem = _toy_problem_for_solver(backend_numpy_single)
    op = Operator(problem)
    rng = np.random.default_rng((hash(solver_name) + 1) & 0xFFFF)
    u_true_double = (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)) * 0.05
    u_true = backend_numpy_single.array(u_true_double.astype(np.complex64))
    rhs = op.matvec(u_true)
    result = solver_factory().solve(op, rhs)
    # f32 single precision: looser convergence threshold
    final_rel = result.residual_history[-1] if result.residual_history else float("inf")
    assert final_rel < 5e-4, f"{solver_name}@single: final residual {final_rel:.2e} too large"
