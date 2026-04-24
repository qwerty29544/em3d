"""Solver base classes: config, result, and Protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

import numpy as np


@dataclass
class SolverConfig:
    max_iter: int = 200
    rtol: float = 1e-6
    log: bool = False
    mu: Optional[complex] = None     # γ₀ centre for SIM
    radius: Optional[float] = None   # γ₀ radius for SIM

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
