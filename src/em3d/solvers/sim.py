from __future__ import annotations

import numpy as np

from .base import SolverConfig, SolverResult


class SIM:
    def __init__(self, config: SolverConfig):
        config.require_gamma()
        self.cfg = config

    def solve(self, operator, rhs) -> SolverResult:
        backend = operator.backend
        xp = backend.xp
        config = self.cfg
        gamma = 1.0 / complex(config.mu)
        solution = xp.zeros_like(rhs)
        residuals: list[float] = []
        residual_action_counts: list[int] = []
        rhs_norm = float(backend.to_host(xp.linalg.norm(rhs)))
        if rhs_norm == 0.0:
            return SolverResult(
                u=solution,
                iterations=0,
                residual_history=[0.0],
                converged=True,
                matvec_count=0,
                residual_action_counts=[0],
                status="converged",
                true_final_residual=0.0,
            )

        relative_tolerance = max(
            float(config.rtol), float(config.atol) / rhs_norm
        )
        updates = 0
        matvec_count = 0
        status = "max_iter"
        converged = False

        # Residuals are evaluated at the initial state and after every update.
        # This makes the last recorded residual the true residual of the
        # returned iterate, including the max-iteration exit.
        while True:
            residual = operator.matvec(solution) - rhs
            matvec_count += 1
            relative = float(backend.to_host(xp.linalg.norm(residual))) / rhs_norm
            residuals.append(relative)
            residual_action_counts.append(matvec_count)
            if config.log:
                print(
                    f"[SIM] updates={updates}, actions={matvec_count}, "
                    f"rel_res={relative:.3e}"
                )
            if not np.isfinite(relative):
                status = "nonfinite"
                break
            if relative < relative_tolerance:
                status = "converged"
                converged = True
                break
            if (
                config.divergence_guard is not None
                and relative > float(config.divergence_guard)
            ):
                status = "divergence_guard"
                break
            if updates >= int(config.max_iter):
                status = "max_iter"
                break
            solution = solution - backend.complex_dtype(gamma) * residual
            updates += 1

        return SolverResult(
            u=solution,
            iterations=updates,
            residual_history=residuals,
            converged=converged,
            matvec_count=matvec_count,
            rmatvec_count=0,
            residual_action_counts=residual_action_counts,
            status=status,
            true_final_residual=float(residuals[-1]),
        )
