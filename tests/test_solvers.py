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


class _MatrixOperator:
    def __init__(self, A, backend):
        self.A = np.asarray(A, dtype=np.complex128)
        self.backend = backend

    def matvec(self, u):
        return self.backend.array(self.A @ np.asarray(u))

    def rmatvec(self, u):
        return self.backend.array(self.A.conj().T @ np.asarray(u))


def _manual_twostep_two_updates(A, rhs):
    u0 = np.zeros_like(rhs, dtype=np.complex128)
    r0 = A @ u0 - rhs
    g0 = A.conj().T @ r0
    Hg0 = A @ g0
    h0 = np.vdot(g0, g0).real / np.vdot(Hg0, Hg0).real
    u1 = u0 - h0 * g0

    r1 = A @ u1 - rhs
    g1 = A.conj().T @ r1
    Hg1 = A @ g1
    delta_r = r1 - r0
    a00 = np.vdot(delta_r, delta_r).real
    a01 = np.vdot(delta_r, Hg1).real
    a11 = np.vdot(Hg1, Hg1).real
    b0 = np.vdot(r1, delta_r).real
    b1 = np.vdot(r1, Hg1).real
    det = a00 * a11 - a01 * a01
    t = (b0 * a11 - b1 * a01) / det
    h = (a00 * b1 - a01 * b0) / det
    return u1 - t * (u1 - u0) - h * g1


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


def test_twostep_uses_two_step_recurrence(backend_numpy_double):
    A = np.array([[2.0, 0.5], [1.0, 1.5]], dtype=np.complex128)
    rhs = backend_numpy_double.array(np.array([1.0, -0.5], dtype=np.complex128))
    op = _MatrixOperator(A, backend_numpy_double)

    result = TwoStep(SolverConfig(max_iter=2, rtol=1e-15)).solve(op, rhs)

    expected = _manual_twostep_two_updates(A, np.asarray(rhs))
    assert np.allclose(np.asarray(result.u), expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(A @ np.asarray(result.u) - np.asarray(rhs)) < 1e-12


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
