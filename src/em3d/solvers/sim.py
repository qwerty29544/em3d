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
        rhs_norm = float(backend.to_host(xp.linalg.norm(rhs)))
        if rhs_norm == 0.0:
            return SolverResult(
                u=solution,
                iterations=0,
                residual_history=[0.0],
                converged=True,
                matvec_count=0,
                status="converged",
            )

        relative_tolerance = max(
            float(config.rtol), float(config.atol) / rhs_norm
        )
        updates = 0
        status = "max_iter"
        converged = False
        for iteration in range(config.max_iter):
            residual = operator.matvec(solution) - rhs
            relative = float(backend.to_host(xp.linalg.norm(residual))) / rhs_norm
            residuals.append(relative)
            if config.log:
                print(f"[SIM] iter={iteration}, rel_res={relative:.3e}")
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
            solution = solution - backend.complex_dtype(gamma) * residual
            updates += 1

        return SolverResult(
            u=solution,
            iterations=updates,
            residual_history=residuals,
            converged=converged,
            matvec_count=len(residuals),
            status=status,
        )
