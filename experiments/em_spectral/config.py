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


@dataclass(frozen=True)
class IterationStudyConfig:
    max_iter: int = 500
    rtol: float = 1e-8
    divergence_guard: float = 1e6


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
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eigenvalue_repeats: int = 1
    output_root: Path = Path("outputs/em_spectral_transfer")

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
            runtime=RuntimeConfig(device=device),
            eigenvalue_repeats=3,
            output_root=Path(output_root),
        )
