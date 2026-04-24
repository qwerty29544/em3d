"""Solver base classes: config, result, and Protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol


@dataclass
class SolverConfig:
    """Configuration shared by all iterative solvers.

    `mu` and `radius` are consumed only by :class:`SIM` (they encode the γ₀
    iteration parameter computed by :func:`em3d.gamma0.find_params`).
    :class:`BiCGStab` and :class:`TwoStep` silently ignore these fields.
    """

    max_iter: int = 200
    rtol: float = 1e-6
    log: bool = False
    mu: Optional[complex] = None     # γ₀ centre — SIM only
    radius: Optional[float] = None   # γ₀ radius — SIM only

    def require_gamma(self) -> None:
        if self.mu is None or self.radius is None:
            raise ValueError("SolverConfig: mu and radius must be set (call gamma0.find_params)")


@dataclass
class SolverResult:
    u: Any
    iterations: int
    residual_history: List[float]
    converged: bool


class BaseSolver(Protocol):
    """Iterative solver for problem (I + B·η) u = rhs."""

    def solve(self, operator, rhs) -> SolverResult: ...
