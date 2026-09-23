from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol


@dataclass
class SolverConfig:
    max_iter: int = 200
    rtol: float = 1e-6
    atol: float = 0.0
    divergence_guard: float | None = None
    max_operator_actions: int | None = None
    log: bool = False
    mu: Optional[complex] = None
    radius: Optional[float] = None

    def require_gamma(self) -> None:
        if self.mu is None or self.radius is None:
            raise ValueError(
                "SolverConfig: mu and radius must be set "
                "(call gamma0.find_params)"
            )
        if abs(self.mu) == 0.0:
            raise ValueError("SolverConfig: mu must be non-zero")


@dataclass
class SolverResult:
    u: Any
    iterations: int
    residual_history: List[float]
    converged: bool
    matvec_count: int = 0
    rmatvec_count: int = 0
    residual_action_counts: List[int] = field(default_factory=list)
    status: str = "unknown"
    true_final_residual: float | None = None

    @property
    def operator_action_count(self) -> int:
        """Total number of forward and adjoint operator applications."""

        return int(self.matvec_count + self.rmatvec_count)


class BaseSolver(Protocol):
    def solve(self, operator, rhs) -> SolverResult: ...
