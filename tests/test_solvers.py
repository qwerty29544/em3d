import numpy as np

from em3d.solvers.base import SolverConfig, SolverResult


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
