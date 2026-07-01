"""Packaged research experiment helpers for em3d."""
from __future__ import annotations

from .structured_lattice import (
    ExperimentLogger,
    InclusionSpec,
    MaterialSpec,
    StructuredLatticeCase,
    build_structured_lattice_problem,
    make_structured_lattice_case,
    run_structured_lattice_experiment,
)

__all__ = [
    "ExperimentLogger",
    "InclusionSpec",
    "MaterialSpec",
    "StructuredLatticeCase",
    "build_structured_lattice_problem",
    "make_structured_lattice_case",
    "run_structured_lattice_experiment",
]
