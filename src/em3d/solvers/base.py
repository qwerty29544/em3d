from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol


@dataclass
class SolverConfig:
    max_iter: int = 200
    rtol: float = 1e-6
    atol: float = 0.0
    divergence_guard: float | None = None
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
    status: str = "unknown"


class BaseSolver(Protocol):
    def solve(self, operator, rhs) -> SolverResult: ...
