"""em3d: volume-integral-equation solver for 3D electrodynamics on structured grids."""
from __future__ import annotations

from .backend import Backend
from .dtypes import Precision
from .grid import Grid
from .problem import Problem
from .operator import Operator
from .refraction import (
    apply_refraction,
    cylinder_refraction,
    ellipsis_refraction,
    step_refraction,
)
from .wave import flat_wave_vec
from . import gamma0
from .solvers import BaseSolver, BiCGStab, SIM, SolverConfig, SolverResult, TwoStep

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "Precision",
    "Grid",
    "Problem",
    "Operator",
    "apply_refraction",
    "cylinder_refraction",
    "ellipsis_refraction",
    "step_refraction",
    "flat_wave_vec",
    "gamma0",
    "BaseSolver",
    "BiCGStab",
    "SIM",
    "SolverConfig",
    "SolverResult",
    "TwoStep",
    "__version__",
]
