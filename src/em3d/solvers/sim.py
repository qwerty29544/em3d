"""Generalised simple iteration (MSGD) driven by γ₀."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


class SIM:
    def __init__(self, config: SolverConfig):
        config.require_gamma()
        self.cfg = config

    def solve(self, operator, rhs) -> SolverResult:
        be = operator.backend
        xp = be.xp
        cfg = self.cfg
        gamma = 1.0 / cfg.mu  # γ₀ = 1/μ for SIM
        u = xp.zeros_like(rhs)
        residuals: list[float] = []
        rhs_norm = float(xp.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return SolverResult(u=u, iterations=0, residual_history=[0.0], converged=True)
        for k in range(cfg.max_iter):
            Au = operator.matvec(u)
            r = Au - rhs
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[SIM] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)
            u = u - be.complex_dtype(gamma) * r
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
