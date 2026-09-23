from __future__ import annotations

import numpy as np

from .base import SolverConfig, SolverResult


class BiCGStab:
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
                residual_action_counts=[0],
                status="converged",
                true_final_residual=0.0,
            )

        tolerance = max(float(cfg.rtol), float(cfg.atol) / rhs_norm)
        matvec_count = 0
        r = rhs - operator.matvec(u)
        matvec_count += 1
        initial_relative = float(be.to_host(xp.linalg.norm(r))) / rhs_norm
        residuals.append(initial_relative)
        residual_action_counts.append(matvec_count)
        if not np.isfinite(initial_relative):
            return SolverResult(
                u=u,
                iterations=0,
                residual_history=residuals,
                converged=False,
                matvec_count=matvec_count,
                residual_action_counts=residual_action_counts,
                status="nonfinite",
                true_final_residual=initial_relative,
            )
        if initial_relative < tolerance:
            return SolverResult(
                u=u,
                iterations=0,
                residual_history=residuals,
                converged=True,
                matvec_count=matvec_count,
                residual_action_counts=residual_action_counts,
                status="converged",
                true_final_residual=initial_relative,
            )

        r_hat = xp.asarray(r, dtype=rhs.dtype).copy()
        rho_prev = 1.0 + 0.0j
        alpha = 1.0 + 0.0j
        omega = 1.0 + 0.0j
        v = xp.zeros_like(rhs)
        p = xp.zeros_like(rhs)
        status = "max_iter"
        completed_updates = 0

        for iteration in range(int(cfg.max_iter)):
            rho = complex(be.to_host(xp.vdot(r_hat, r)))
            if abs(rho) == 0.0:
                status = "breakdown_rho"
                break
            if iteration == 0:
                p = r.copy()
            else:
                if abs(omega) == 0.0:
                    status = "breakdown_omega"
                    break
                beta = (rho / rho_prev) * (alpha / omega)
                p = r + be.complex_dtype(beta) * (
                    p - be.complex_dtype(omega) * v
                )

            v = operator.matvec(p)
            matvec_count += 1
            denominator = complex(be.to_host(xp.vdot(r_hat, v)))
            if abs(denominator) == 0.0:
                status = "breakdown_alpha"
                break
            alpha = rho / denominator
            s = r - be.complex_dtype(alpha) * v
            s_relative = float(be.to_host(xp.linalg.norm(s))) / rhs_norm
            u_alpha = u + be.complex_dtype(alpha) * p

            if not np.isfinite(s_relative):
                u = u_alpha
                completed_updates = iteration + 1
                residuals.append(s_relative)
                residual_action_counts.append(matvec_count)
                status = "nonfinite"
                break
            if (
                cfg.divergence_guard is not None
                and s_relative > float(cfg.divergence_guard)
            ):
                u = u_alpha
                completed_updates = iteration + 1
                residuals.append(s_relative)
                residual_action_counts.append(matvec_count)
                status = "divergence_guard"
                break
            if s_relative < tolerance:
                u = u_alpha
                completed_updates = iteration + 1
                residuals.append(s_relative)
                residual_action_counts.append(matvec_count)
                status = "converged"
                break

            t = operator.matvec(s)
            matvec_count += 1
            t_norm_sq = complex(be.to_host(xp.vdot(t, t)))
            if abs(t_norm_sq) < 1e-30 * rhs_norm * rhs_norm:
                u = u_alpha
                completed_updates = iteration + 1
                residuals.append(s_relative)
                residual_action_counts.append(matvec_count)
                status = "breakdown_t"
                break
            omega = complex(be.to_host(xp.vdot(t, s))) / t_norm_sq
            if abs(omega) == 0.0:
                u = u_alpha
                completed_updates = iteration + 1
                residuals.append(s_relative)
                residual_action_counts.append(matvec_count)
                status = "breakdown_omega"
                break

            u = u_alpha + be.complex_dtype(omega) * s
            r = s - be.complex_dtype(omega) * t
            completed_updates = iteration + 1
            relative = float(be.to_host(xp.linalg.norm(r))) / rhs_norm
            residuals.append(relative)
            residual_action_counts.append(matvec_count)
            if cfg.log:
                print(
                    f"[BiCGStab] iter={completed_updates}, "
                    f"actions={matvec_count}, rel_res={relative:.3e}"
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
            rho_prev = rho
        else:
            status = "max_iter"

        converged = status == "converged"
        final_residual = float(residuals[-1]) if residuals else float("inf")
        return SolverResult(
            u=u,
            iterations=completed_updates,
            residual_history=residuals,
            converged=converged,
            matvec_count=matvec_count,
            rmatvec_count=0,
            residual_action_counts=residual_action_counts,
            status=status,
            true_final_residual=final_residual,
        )
