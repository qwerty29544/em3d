from __future__ import annotations

import numpy as np

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
        rhs_norm = float(be.to_host(xp.linalg.norm(rhs)))
        residuals: list[float] = []
        residual_action_counts: list[int] = []
        u = xp.zeros_like(rhs)
        if rhs_norm == 0.0:
            return SolverResult(
                u=u,
                iterations=0,
                residual_history=[0.0],
                converged=True,
                matvec_count=0,
                rmatvec_count=0,
                residual_action_counts=[0],
                status="converged",
                true_final_residual=0.0,
            )

        tolerance = max(float(cfg.rtol), float(cfg.atol) / rhs_norm)
        previous_u = None
        previous_r = None
        updates = 0
        matvec_count = 0
        rmatvec_count = 0
        status = "max_iter"

        # As for SIM, the final residual is explicitly evaluated after the last
        # permitted update.  Action counts include both A and A* applications.
        while True:
            Au = operator.matvec(u)
            matvec_count += 1
            r = Au - rhs
            relative = float(be.to_host(xp.linalg.norm(r))) / rhs_norm
            residuals.append(relative)
            residual_action_counts.append(matvec_count + rmatvec_count)
            if cfg.log:
                print(
                    f"[TwoStep] updates={updates}, "
                    f"actions={matvec_count + rmatvec_count}, "
                    f"rel_res={relative:.3e}"
                )
            if not np.isfinite(relative):
                status = "nonfinite"
                break
            if relative < tolerance:
                status = "converged"
                break
            if (
                cfg.divergence_guard is not None
                and relative > float(cfg.divergence_guard)
            ):
                status = "divergence_guard"
                break
            if (
                cfg.max_operator_actions is not None
                and matvec_count + rmatvec_count >= int(cfg.max_operator_actions)
            ):
                status = "max_operator_actions"
                break
            if updates >= int(cfg.max_iter):
                status = "max_iter"
                break

            if (
                cfg.max_operator_actions is not None
                and matvec_count + rmatvec_count + 2
                > int(cfg.max_operator_actions)
            ):
                status = "max_operator_actions"
                break
            gradient = operator.rmatvec(r)
            rmatvec_count += 1
            H_gradient = operator.matvec(gradient)
            matvec_count += 1
            H_gradient_norm_sq = _real_inner(xp, H_gradient, H_gradient)
            if not np.isfinite(H_gradient_norm_sq) or H_gradient_norm_sq <= 0.0:
                status = "breakdown_gradient"
                break

            if previous_u is None:
                gradient_norm_sq = _real_inner(xp, gradient, gradient)
                h = gradient_norm_sq / H_gradient_norm_sq
                next_u = u - be.complex_dtype(h) * gradient
            else:
                delta_r = r - previous_r
                a00 = _real_inner(xp, delta_r, delta_r)
                a01 = _real_inner(xp, delta_r, H_gradient)
                a11 = H_gradient_norm_sq
                b0 = _real_inner(xp, r, delta_r)
                b1 = _real_inner(xp, r, H_gradient)
                det = a00 * a11 - a01 * a01
                det_scale = max(abs(a00 * a11), abs(a01 * a01), 1.0)
                if abs(det) <= 1e-14 * det_scale:
                    t = 0.0
                    h = b1 / a11
                else:
                    t = (b0 * a11 - b1 * a01) / det
                    h = (a00 * b1 - a01 * b0) / det
                next_u = (
                    u
                    - be.complex_dtype(t) * (u - previous_u)
                    - be.complex_dtype(h) * gradient
                )

            previous_u = u
            previous_r = r
            u = next_u
            updates += 1

        converged = status == "converged"
        return SolverResult(
            u=u,
            iterations=updates,
            residual_history=residuals,
            converged=converged,
            matvec_count=matvec_count,
            rmatvec_count=rmatvec_count,
            residual_action_counts=residual_action_counts,
            status=status,
            true_final_residual=float(residuals[-1]),
        )
