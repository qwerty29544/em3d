"""Scalar acoustic scattering tools."""
from __future__ import annotations

from . import farfield
from . import gamma0
from . import visualization
from .dense import B_scalar_matrix, H_scalar_matrix
from .farfield import farfield_amplitude, pattern_plane, scattering_pattern
from .kernel import kernel_on_doubled_grid, self_cell_coefficient
from .materials import eta_ellipsoid, eta_homogeneous, eta_slab, eta_sphere
from .operator import AcousticOperator
from .problem import AcousticProblem, make_acoustic_problem, plane_wave_scalar

__all__ = [
    "AcousticProblem",
    "AcousticOperator",
    "B_scalar_matrix",
    "H_scalar_matrix",
    "eta_ellipsoid",
    "eta_homogeneous",
    "eta_slab",
    "eta_sphere",
    "farfield",
    "farfield_amplitude",
    "gamma0",
    "kernel_on_doubled_grid",
    "make_acoustic_problem",
    "pattern_plane",
    "plane_wave_scalar",
    "scattering_pattern",
    "self_cell_coefficient",
    "visualization",
]
