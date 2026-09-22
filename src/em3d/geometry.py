from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import pi
from typing import TypeAlias

import numpy as np

from .grid import Grid


class SamplingMode(str, Enum):
    CELL_CENTER = "cell_center"
    VOLUME_FRACTION = "volume_fraction"


@dataclass(frozen=True)
class FullDomain:
    pass


@dataclass(frozen=True)
class AxisAlignedBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(float(value) <= 0.0 for value in self.size):
            raise ValueError(f"box sizes must be positive, got {self.size!r}")


@dataclass(frozen=True)
class Ellipsoid:
    center: tuple[float, float, float]
    radii: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(float(value) <= 0.0 for value in self.radii):
            raise ValueError(f"ellipsoid radii must be positive, got {self.radii!r}")


Geometry: TypeAlias = FullDomain | AxisAlignedBox | Ellipsoid


@dataclass(frozen=True)
class GeometryDiagnostics:
    cells_total: int
    cells_per_axis: tuple[int, int, int]
    represented_volume: float
    exact_volume: float
    relative_volume_error: float
    resolved: bool
    sampling_mode: SamplingMode


def _host_axis(grid: Grid, axis: int) -> np.ndarray:
    values = (grid.x, grid.y, grid.z)[axis]
    return np.asarray(grid.backend.to_host(values), dtype=np.float64)


def _canonical_tolerance(grid: Grid) -> float:
    scale = max(1.0, *(abs(float(value)) for value in grid.L))
    return 64.0 * np.finfo(np.float64).eps * scale


def center_mask(geometry: Geometry, grid: Grid) -> np.ndarray:
    """Return the canonical CPU mask of cell centres inside ``geometry``."""

    if isinstance(geometry, FullDomain):
        return np.ones(grid.N, dtype=bool)

    x, y, z = (_host_axis(grid, index) for index in range(3))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    tol = _canonical_tolerance(grid)

    if isinstance(geometry, AxisAlignedBox):
        result = np.ones(grid.N, dtype=bool)
        for coordinates, centre, size in zip(
            (X, Y, Z), geometry.center, geometry.size, strict=True
        ):
            result &= np.abs(coordinates - float(centre)) <= float(size) / 2.0 + tol
        return result

    if isinstance(geometry, Ellipsoid):
        value = np.zeros(grid.N, dtype=np.float64)
        for coordinates, centre, radius in zip(
            (X, Y, Z), geometry.center, geometry.radii, strict=True
        ):
            value += ((coordinates - float(centre)) / float(radius)) ** 2
        return value <= 1.0 + tol

    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def _axis_overlap_fractions(
    centres: np.ndarray,
    step: float,
    feature_center: float,
    feature_size: float,
) -> np.ndarray:
    half_cell = float(step) / 2.0
    cell_left = centres - half_cell
    cell_right = centres + half_cell
    feature_left = float(feature_center) - float(feature_size) / 2.0
    feature_right = float(feature_center) + float(feature_size) / 2.0
    overlap = np.maximum(
        0.0,
        np.minimum(cell_right, feature_right) - np.maximum(cell_left, feature_left),
    )
    return np.clip(overlap / float(step), 0.0, 1.0)


def volume_fractions(geometry: Geometry, grid: Grid) -> np.ndarray:
    """Fraction of every grid cell occupied by ``geometry``.

    Analytic fractions are currently implemented for the full domain and for
    axis-aligned boxes.  Ellipsoid fractions require a separate quadrature or
    sub-voxel rule and are deliberately not approximated silently.
    """

    if isinstance(geometry, FullDomain):
        return np.ones(grid.N, dtype=np.float64)

    if isinstance(geometry, AxisAlignedBox):
        steps = tuple(float(length) / int(count) for length, count in zip(grid.L, grid.N, strict=True))
        fractions = [
            _axis_overlap_fractions(
                _host_axis(grid, axis),
                steps[axis],
                geometry.center[axis],
                geometry.size[axis],
            )
            for axis in range(3)
        ]
        return (
            fractions[0][:, None, None]
            * fractions[1][None, :, None]
            * fractions[2][None, None, :]
        )

    if isinstance(geometry, Ellipsoid):
        raise NotImplementedError(
            "volume-fraction sampling for Ellipsoid is not implemented; "
            "use SamplingMode.CELL_CENTER or add an explicit quadrature rule"
        )

    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def exact_volume(geometry: Geometry, grid: Grid) -> float:
    if isinstance(geometry, FullDomain):
        return float(np.prod(grid.L))
    if isinstance(geometry, AxisAlignedBox):
        return float(np.prod(geometry.size))
    if isinstance(geometry, Ellipsoid):
        return float(4.0 * pi * np.prod(geometry.radii) / 3.0)
    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def sampling_weights(
    geometry: Geometry,
    grid: Grid,
    mode: SamplingMode = SamplingMode.CELL_CENTER,
) -> np.ndarray:
    mode = SamplingMode(mode)
    if mode is SamplingMode.CELL_CENTER:
        return center_mask(geometry, grid).astype(np.float64)
    return volume_fractions(geometry, grid)


def geometry_diagnostics(
    geometry: Geometry,
    grid: Grid,
    mode: SamplingMode = SamplingMode.CELL_CENTER,
) -> GeometryDiagnostics:
    mode = SamplingMode(mode)
    weights = sampling_weights(geometry, grid, mode)
    occupied = weights > 0.0
    indices = np.argwhere(occupied)
    if len(indices):
        cells_per_axis = tuple(
            int(len(np.unique(indices[:, axis]))) for axis in range(3)
        )
    else:
        cells_per_axis = (0, 0, 0)

    represented = float(np.sum(weights) * grid.dv)
    exact = exact_volume(geometry, grid)
    relative_error = (represented - exact) / exact if exact > 0.0 else 0.0
    resolved = min(cells_per_axis) >= 2
    return GeometryDiagnostics(
        cells_total=int(np.count_nonzero(occupied)),
        cells_per_axis=cells_per_axis,
        represented_volume=represented,
        exact_volume=exact,
        relative_volume_error=float(relative_error),
        resolved=bool(resolved),
        sampling_mode=mode,
    )


def _as_material_tensor(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        return complex(array) * np.eye(3, dtype=np.complex128)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must be a scalar or a (3, 3) tensor, got {array.shape}")
    return array


def sample_relative_permittivity(
    grid: Grid,
    *,
    background_eps_r,
    feature_eps_r,
    geometry: Geometry,
    mode: SamplingMode = SamplingMode.CELL_CENTER,
):
    """Sample a two-material relative-permittivity field on ``grid``."""

    background = _as_material_tensor(background_eps_r, name="background_eps_r")
    feature = _as_material_tensor(feature_eps_r, name="feature_eps_r")
    weights = sampling_weights(geometry, grid, mode)
    eps_host = (
        background[:, :, None, None, None]
        + (feature - background)[:, :, None, None, None]
        * weights[None, None, ...]
    )
    return grid.backend.array(eps_host, dtype=grid.backend.complex_dtype)


def sample_contrast_tensor(
    grid: Grid,
    *,
    background_eps_r,
    feature_eps_r,
    geometry: Geometry,
    mode: SamplingMode = SamplingMode.CELL_CENTER,
):
    """Sample ``chi = eps_r - I`` for the electrodynamic volume operator."""

    eps_r = sample_relative_permittivity(
        grid,
        background_eps_r=background_eps_r,
        feature_eps_r=feature_eps_r,
        geometry=geometry,
        mode=mode,
    )
    identity = np.eye(3, dtype=np.complex128)[:, :, None, None, None]
    identity_backend = grid.backend.array(identity, dtype=grid.backend.complex_dtype)
    return (eps_r - identity_backend).astype(grid.backend.complex_dtype, copy=False)
