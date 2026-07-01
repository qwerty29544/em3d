"""Two-step gradient descent (MSGD/TwoSGD) using matvec and rmatvec."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


def _real_inner(xp, x, y) -> float:
    return float(xp.vdot(x, y).real)


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

        previous_u = None
        previous_r = None
        for k in range(cfg.max_iter):
            Au = operator.matvec(u)
            r = Au - rhs
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[TwoStep] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)

            gradient = operator.rmatvec(r)
            H_gradient = operator.matvec(gradient)
            H_gradient_norm_sq = _real_inner(xp, H_gradient, H_gradient)
            if H_gradient_norm_sq == 0.0:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=False)

            if previous_u is None:
                # First MSGD step: one-dimensional steepest descent.
                gradient_norm_sq = _real_inner(xp, gradient, gradient)
                h = gradient_norm_sq / H_gradient_norm_sq
                next_u = u - be.complex_dtype(h) * gradient
            else:
                # Two-step MSGD recurrence from the local 2x2 minimization.
                delta_r = r - previous_r
                a00 = _real_inner(xp, delta_r, delta_r)
                a01 = _real_inner(xp, delta_r, H_gradient)
                a11 = H_gradient_norm_sq
                b0 = _real_inner(xp, r, delta_r)
                b1 = _real_inner(xp, r, H_gradient)
                det = a00 * a11 - a01 * a01
                det_scale = max(abs(a00 * a11), abs(a01 * a01), 1.0)
                if abs(det) <= 1e-14 * det_scale:
                    # Degenerate two-dimensional subspace: fall back to the
                    # one-dimensional residual minimizer along H* r_k.
                    t = 0.0
                    h = b1 / a11
                else:
                    t = (b0 * a11 - b1 * a01) / det
                    h = (a00 * b1 - a01 * b0) / det
                next_u = u - be.complex_dtype(t) * (u - previous_u) - be.complex_dtype(h) * gradient

            previous_u = u
            previous_r = r
            u = next_u
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
