"""BiCGStab solver ported from the Yurchenkov notebook."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


class BiCGStab:
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

        r = rhs - operator.matvec(u)
        r_hat = xp.asarray(r, dtype=rhs.dtype).copy()  # shadow residual
        rho_prev = 1.0
        alpha = 1.0
        omega = 1.0
        v = xp.zeros_like(rhs)
        p = xp.zeros_like(rhs)
        for k in range(cfg.max_iter):
            rho = complex(xp.vdot(r_hat, r))
            if rho == 0:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=False)
            beta = (rho / rho_prev) * (alpha / omega) if k > 0 else 0.0
            p = r + be.complex_dtype(beta) * (p - be.complex_dtype(omega) * v)
            v = operator.matvec(p)
            alpha = rho / complex(xp.vdot(r_hat, v))
            s = r - be.complex_dtype(alpha) * v
            s_norm = float(xp.linalg.norm(s))
            if s_norm / rhs_norm < cfg.rtol:
                u = u + be.complex_dtype(alpha) * p
                residuals.append(s_norm / rhs_norm)
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)
            t = operator.matvec(s)
            t_norm_sq = complex(xp.vdot(t, t))
            if abs(t_norm_sq) < 1e-30 * rhs_norm * rhs_norm:
                # BiCGStab breakdown: A·s ≈ 0, cannot continue
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=False)
            omega = complex(xp.vdot(t, s)) / t_norm_sq
            u = u + be.complex_dtype(alpha) * p + be.complex_dtype(omega) * s
            r = s - be.complex_dtype(omega) * t
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[BiCGStab] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k + 1, residual_history=residuals, converged=True)
            rho_prev = rho
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
