"""Scalar acoustic scattering problem definitions."""
from __future__ import annotations

from dataclasses import dataclass

from ..grid import Grid


def _as_direction_tuple(direction) -> tuple[float, float, float]:
    values = tuple(float(x) for x in direction)
    if len(values) != 3:
        raise ValueError(f"direction must have length 3, got {direction!r}")
    return values


def _normalized_direction(direction, be):
    xp = be.xp
    d = xp.asarray(_as_direction_tuple(direction), dtype=be.real_dtype)
    norm = float(xp.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    return d / norm


def plane_wave_scalar(
    grid: Grid,
    *,
    k: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    amplitude: complex = 1.0,
):
    """Return scalar plane wave ``amplitude * exp(i k direction dot x)``."""
    be = grid.backend
    xp = be.xp
    d = _normalized_direction(direction, be)
    X, Y, Z = grid.coords()
    phase = d[0] * X + d[1] * Y + d[2] * Z
    ik = be.complex_dtype(1j * float(k))
    wave = be.complex_dtype(amplitude) * xp.exp(ik * phase)
    return wave.astype(be.complex_dtype, copy=False)


@dataclass(frozen=True)
class AcousticProblem:
    """Grid, scalar eta, incident wave and wave number for acoustic scattering."""

    grid: Grid
    eta: object
    wave: object
    k0: float
    volume: float

    def __post_init__(self) -> None:
        be = self.grid.backend
        expected = self.grid.N
        if getattr(self.eta, "shape", None) != expected:
            raise ValueError(f"eta.shape {getattr(self.eta, 'shape', None)} != expected {expected}")
        if getattr(self.wave, "shape", None) != expected:
            raise ValueError(f"wave.shape {getattr(self.wave, 'shape', None)} != expected {expected}")
        if self.eta.dtype != be.complex_dtype:
            raise TypeError(f"eta.dtype {self.eta.dtype} != {be.complex_dtype}")
        if self.wave.dtype != be.complex_dtype:
            raise TypeError(f"wave.dtype {self.wave.dtype} != {be.complex_dtype}")
        if float(self.k0) <= 0.0:
            raise ValueError(f"k0 must be positive, got {self.k0}")
        if float(self.volume) <= 0.0:
            raise ValueError(f"volume must be positive, got {self.volume}")

    @property
    def backend(self):
        return self.grid.backend

    @property
    def chi(self):
        """Return acoustic contrast used by the integral operator."""
        return (self.eta - self.backend.complex_dtype(1.0)).astype(
            self.backend.complex_dtype,
            copy=False,
        )


def make_acoustic_problem(
    grid: Grid,
    eta,
    *,
    k0: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    amplitude: complex = 1.0,
) -> AcousticProblem:
    """Build an acoustic problem from eta and a scalar incident plane wave."""
    be = grid.backend
    eta_arr = be.asarray_of_kind(eta, "complex")
    wave = plane_wave_scalar(grid, k=float(k0), direction=direction, amplitude=amplitude)
    n_cells = int(grid.N[0] * grid.N[1] * grid.N[2])
    return AcousticProblem(
        grid=grid,
        eta=eta_arr,
        wave=wave,
        k0=float(k0),
        volume=grid.dv * n_cells,
    )
