from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from em3d.spectral import ArnoldiConfig


@dataclass(frozen=True)
class GridHierarchyConfig:
    coarse_sizes: tuple[int, ...]
    control_size: int
    fine_size: int

    def __post_init__(self) -> None:
        if not self.coarse_sizes or any(value <= 0 for value in self.coarse_sizes):
            raise ValueError("coarse_sizes must contain positive integers")
        if self.control_size <= 0 or self.fine_size <= 0:
            raise ValueError("control_size and fine_size must be positive")
        if len(set(self.coarse_sizes)) != len(self.coarse_sizes):
            raise ValueError("coarse_sizes must not contain duplicates")


@dataclass(frozen=True)
class IterationStudyConfig:
    max_iter: int = 500
    rtol: float = 1e-8
    divergence_guard: float = 1e6

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.rtol <= 0.0:
            raise ValueError("rtol must be positive")
        if self.divergence_guard <= 1.0:
            raise ValueError("divergence_guard must exceed one")


@dataclass(frozen=True)
class GeometryScanConfig:
    inclusion_sides: tuple[float, ...]
    ensembles: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.inclusion_sides or any(value <= 0.0 for value in self.inclusion_sides):
            raise ValueError("inclusion_sides must contain positive values")
        if len(set(self.inclusion_sides)) != len(self.inclusion_sides):
            raise ValueError("inclusion_sides must not contain duplicates")
        for levels in self.ensembles:
            if not levels:
                raise ValueError("each ensemble must contain at least one level")
            if tuple(sorted(set(levels))) != tuple(levels):
                raise ValueError(
                    "ensemble levels must be strictly increasing and unique"
                )


@dataclass(frozen=True)
class WaveNumberScanConfig:
    inclusion_side: float
    wave_numbers: tuple[float, ...]
    boundary_points: tuple[tuple[float, int], ...] = ()
    ppw_threshold: float = 2.0
    arnoldi_boundary_only: bool = True

    def __post_init__(self) -> None:
        if self.inclusion_side <= 0.0:
            raise ValueError("inclusion_side must be positive")
        if not self.wave_numbers or any(value <= 0.0 for value in self.wave_numbers):
            raise ValueError("wave_numbers must contain positive values")
        if self.ppw_threshold <= 0.0:
            raise ValueError("ppw_threshold must be positive")
        for wave_number, level in self.boundary_points:
            if wave_number <= 0.0 or level <= 0:
                raise ValueError("boundary points must contain positive values")


@dataclass(frozen=True)
class RuntimeConfig:
    device: Literal["cpu", "cuda", "auto"] = "auto"
    precision: Literal["single", "double"] = "double"
    random_seed: int = 20260908


@dataclass(frozen=True)
class SpectralStudyConfig:
    mode: Literal["quick", "publication"]
    grids: GridHierarchyConfig
    iteration: IterationStudyConfig = field(default_factory=IterationStudyConfig)
    arnoldi: ArnoldiConfig = field(default_factory=ArnoldiConfig)
    geometry: GeometryScanConfig = field(
        default_factory=lambda: GeometryScanConfig(
            inclusion_sides=(0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30),
            ensembles=(
                (4, 5),
                (4, 5, 6),
                (4, 5, 6, 7),
                (4, 5, 6, 7, 8),
            ),
        )
    )
    wave_number: WaveNumberScanConfig = field(
        default_factory=lambda: WaveNumberScanConfig(
            inclusion_side=0.40,
            wave_numbers=(4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
            boundary_points=((18.0, 12), (20.0, 12), (20.0, 14)),
        )
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eigenvalue_repeats: int = 1
    output_root: Path = Path("outputs/em_spectral_transfer")

    def __post_init__(self) -> None:
        if self.eigenvalue_repeats <= 0:
            raise ValueError("eigenvalue_repeats must be positive")
        allowed_levels = set(self.grids.coarse_sizes)
        for ensemble in self.geometry.ensembles:
            if not set(ensemble).issubset(allowed_levels):
                raise ValueError(
                    f"ensemble {ensemble!r} contains levels outside coarse_sizes"
                )

    @classmethod
    def quick(
        cls,
        *,
        output_root: str | Path = "outputs/em_spectral_transfer/quick",
        device: Literal["cpu", "cuda", "auto"] = "cpu",
    ) -> "SpectralStudyConfig":
        return cls(
            mode="quick",
            grids=GridHierarchyConfig(
                coarse_sizes=(2, 3),
                control_size=4,
                fine_size=8,
            ),
            iteration=IterationStudyConfig(max_iter=120, rtol=1e-7),
            arnoldi=ArnoldiConfig(
                krylov_dimension=20,
                milestones=(5, 10, 15, 20),
            ),
            geometry=GeometryScanConfig(
                inclusion_sides=(0.25, 0.50),
                ensembles=((2, 3),),
            ),
            wave_number=WaveNumberScanConfig(
                inclusion_side=0.50,
                wave_numbers=(0.5, 1.0),
                boundary_points=(),
            ),
            runtime=RuntimeConfig(device=device),
            eigenvalue_repeats=1,
            output_root=Path(output_root),
        )

    @classmethod
    def publication(
        cls,
        *,
        output_root: str | Path = "outputs/em_spectral_transfer/publication",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> "SpectralStudyConfig":
        return cls(
            mode="publication",
            grids=GridHierarchyConfig(
                coarse_sizes=(4, 5, 6, 7, 8),
                control_size=10,
                fine_size=64,
            ),
            iteration=IterationStudyConfig(),
            arnoldi=ArnoldiConfig(),
            geometry=GeometryScanConfig(
                inclusion_sides=(0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30),
                ensembles=(
                    (4, 5),
                    (4, 5, 6),
                    (4, 5, 6, 7),
                    (4, 5, 6, 7, 8),
                ),
            ),
            wave_number=WaveNumberScanConfig(
                inclusion_side=0.40,
                wave_numbers=(4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
                boundary_points=((18.0, 12), (20.0, 12), (20.0, 14)),
            ),
            runtime=RuntimeConfig(device=device),
            eigenvalue_repeats=3,
            output_root=Path(output_root),
        )
