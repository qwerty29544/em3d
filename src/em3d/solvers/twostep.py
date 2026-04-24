"""Two-step gradient descent (MSGD/TwoSGD) using matvec and rmatvec."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


class TwoStep:
    def __init__(self, config: SolverConfig):
        self.cfg = config

    def solve(self, operator, rhs) -> SolverResult:
        be = operator.backend
        xp = be.xp
        cfg = self.cfg
        rhs_norm = float(xp.linalg.norm(rhs))
        residuals: list[float] = []
        u = xp.zeros_like(rhs)
        if rhs_norm == 0.0:
            return SolverResult(u=u, iterations=0, residual_history=[0.0], converged=True)

        for k in range(cfg.max_iter):
            Au = operator.matvec(u)
            r = Au - rhs
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[TwoStep] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)

            # step direction = A* r   (steepest descent for ||A u - f||²)
            p = operator.rmatvec(r)
            Ap = operator.matvec(p)
            p_norm_sq = float(xp.vdot(p, p).real)
            Ap_norm_sq = float(xp.vdot(Ap, Ap).real)
            if Ap_norm_sq == 0.0:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=False)
            tau = p_norm_sq / Ap_norm_sq
            u = u - be.complex_dtype(tau) * p
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
