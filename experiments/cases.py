"""Experiment case definitions for chapter 6 electrodynamic studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .materials import MaterialSpec


@dataclass(frozen=True)
class LayerSpec:
    """One z-layer of a full-domain layered box."""

    z_min: float
    z_max: float
    material: MaterialSpec


@dataclass(frozen=True)
class ExperimentCase:
    """Configuration for one electrodynamic VIE experiment."""

    name: str
    N: tuple[int, int, int]
    L: tuple[float, float, float]
    k0: float
    geometry: str
    eps_real: Any
    eps_imag: Any
    center: tuple[float, float, float]
    radius: tuple[float, float, float]
    wave_orient: tuple[float, float, float]
    wave_amplitude: tuple[float, float, float]
    material: MaterialSpec | None = None
    layers: tuple[LayerSpec, ...] = ()

    @property
    def dof(self) -> int:
        return 3 * self.N[0] * self.N[1] * self.N[2]

    def to_jsonable(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("eps_real", "eps_imag"):
            value = out[key]
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        return out


@dataclass(frozen=True)
class SolverRun:
    """Serializable solver result summary for tables and plots."""

    case_name: str
    solver_name: str
    N: tuple[int, int, int]
    dof: int
    converged: bool
    iterations: int
    final_residual: float
    elapsed_sec: float
    residual_history: list[float]

    def to_row(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "solver_name": self.solver_name,
            "N": "x".join(str(n) for n in self.N),
            "dof": self.dof,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_residual": self.final_residual,
            "elapsed_sec": self.elapsed_sec,
        }


def grid_shape(N: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(N, int):
        return (N, N, N)
    shape = tuple(int(n) for n in N)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"N must be a positive int or 3-tuple, got {N!r}")
    return shape


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _n_token(shape: tuple[int, int, int]) -> int | str:
    return shape[0] if shape[0] == shape[1] == shape[2] else "x".join(str(n) for n in shape)


def make_sphere_case(
    *,
    N: int | tuple[int, int, int],
    eps_r: complex,
    k0a: float,
    a: float,
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wave_orient: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wave_amplitude: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ExperimentCase:
    """Build the canonical isotropic sphere case used for Mie validation."""
    shape = grid_shape(N)
    eps = complex(eps_r)
    material = MaterialSpec.isotropic(eps)
    return ExperimentCase(
        name=f"sphere_eps{number_token(eps.real)}_k0a{number_token(k0a)}_N{_n_token(shape)}",
        N=shape,
        L=L,
        k0=float(k0a) / float(a),
        geometry="sphere",
        eps_real=float(eps.real),
        eps_imag=float(eps.imag),
        center=center,
        radius=(float(a), float(a), float(a)),
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        material=material,
    )


def make_anisotropic_ellipsoid_case(
    *,
    N: int | tuple[int, int, int],
    eps_real,
    eps_imag,
    k0: float,
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: tuple[float, float, float] = (0.35, 0.25, 0.2),
    wave_orient: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wave_amplitude: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ExperimentCase:
    """Build a full-tensor anisotropic ellipsoid case."""
    shape = grid_shape(N)
    if np.ndim(eps_real) == 0:
        eps_real_out = float(eps_real)
        eps_imag_out = float(eps_imag)
        material = MaterialSpec.isotropic(complex(eps_real_out, eps_imag_out))
    else:
        eps_real_out = np.array(eps_real, dtype=np.float64, copy=True)
        eps_imag_out = np.array(eps_imag, dtype=np.float64, copy=True)
        material = MaterialSpec.anisotropic_lossy(eps_real_out, eps_imag_out)
    return ExperimentCase(
        name=f"anisotropic_ellipsoid_N{_n_token(shape)}",
        N=shape,
        L=L,
        k0=float(k0),
        geometry="ellipsoid",
        eps_real=eps_real_out,
        eps_imag=eps_imag_out,
        center=center,
        radius=radius,
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        material=material,
    )


def make_uniaxial_crystal_ellipsoid_case(
    *,
    N: int | tuple[int, int, int],
    eps_o: float,
    eps_e: float,
    k0: float,
    radius: tuple[float, float, float] = (0.35, 0.25, 0.2),
    orientation=None,
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wave_orient: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wave_amplitude: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ExperimentCase:
    """Build a uniaxial crystal ellipsoid with ordinary and extraordinary eps."""
    shape = grid_shape(N)
    eps = np.diag([eps_o, eps_o, eps_e]).astype(np.float64)
    material = MaterialSpec.anisotropic(eps, orientation=orientation)
    return ExperimentCase(
        name=f"uniaxial_crystal_N{_n_token(shape)}",
        N=shape,
        L=L,
        k0=float(k0),
        geometry="ellipsoid",
        eps_real=eps,
        eps_imag=np.zeros((3, 3), dtype=np.float64),
        center=center,
        radius=radius,
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        material=material,
    )


def make_layered_box_case(
    *,
    N: int | tuple[int, int, int],
    k0: float,
    layers,
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wave_orient: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wave_amplitude: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ExperimentCase:
    """Build a full-domain layered rectangular parallelepiped."""
    shape = grid_shape(N)
    return ExperimentCase(
        name=f"layered_box_N{_n_token(shape)}_k{number_token(k0)}",
        N=shape,
        L=L,
        k0=float(k0),
        geometry="layered_box",
        eps_real=0.0,
        eps_imag=0.0,
        center=center,
        radius=(0.0, 0.0, 0.0),
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        layers=tuple(layers),
    )
