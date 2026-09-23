from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RuntimeConfig:
    device: Literal["cpu", "cuda", "auto"] = "auto"
    precision: Literal["single", "double"] = "double"
    progress: bool = True
    clear_cuda_cache_between_cases: bool = True


@dataclass(frozen=True)
class SolverStudyConfig:
    case_keys: tuple[str, ...] = ("anisotropic_ellipsoid", "local_inclusion")
    grid_size: int = 24
    coarse_size: int = 6
    coarse_size_overrides: tuple[tuple[str, int], ...] = ()
    solver_names: tuple[str, ...] = ("SIM", "BiCGStab", "TwoStep")
    max_iter: int = 300
    rtol: float = 1e-6
    divergence_guard: float = 1e6
    rcs_n_phi: int = 180
    rcs_planes: tuple[str, ...] = ("xy", "xz", "yz")
    save_solutions: bool = True

    def __post_init__(self) -> None:
        if not self.case_keys:
            raise ValueError("case_keys must not be empty")
        if self.grid_size <= 0 or self.coarse_size <= 0:
            raise ValueError("grid sizes must be positive")
        override_keys = [key for key, _ in self.coarse_size_overrides]
        if len(set(override_keys)) != len(override_keys):
            raise ValueError("coarse_size_overrides must not contain duplicate keys")
        if any(int(size) <= 0 for _, size in self.coarse_size_overrides):
            raise ValueError("coarse-size overrides must be positive")
        if not self.solver_names:
            raise ValueError("solver_names must not be empty")
        if self.max_iter <= 0 or self.rtol <= 0.0:
            raise ValueError("max_iter and rtol must be positive")
        if self.divergence_guard <= 1.0:
            raise ValueError("divergence_guard must exceed one")
        if self.rcs_n_phi < 8:
            raise ValueError("rcs_n_phi must be at least 8")
        if any(plane not in {"xy", "xz", "yz"} for plane in self.rcs_planes):
            raise ValueError("rcs_planes must contain xy, xz, or yz")

    def coarse_size_for(self, case_key: str) -> int:
        overrides = dict(self.coarse_size_overrides)
        return int(overrides.get(case_key, self.coarse_size))


@dataclass(frozen=True)
class MieStudyConfig:
    grid_sizes: tuple[int, ...] = (24, 32, 48)
    eps_values: tuple[complex, ...] = (1.5 + 0.0j, 2.25 + 0.0j, 4.0 + 0.0j)
    k0a_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 6.0)
    radius: float = 0.30
    domain_length: float = 1.0
    solver_names: tuple[str, ...] = ("BiCGStab",)
    sim_coarse_size: int = 6
    max_iter: int = 350
    rtol: float = 1e-6
    divergence_guard: float = 1e6
    rcs_n_phi: int = 180
    rcs_plane: str = "xz"
    field_reference_grids: tuple[int, ...] = (24, 32, 48)
    compare_farfield_backends_on_smallest_grid: bool = True
    save_solutions: bool = False

    def __post_init__(self) -> None:
        if not self.grid_sizes or any(value <= 0 for value in self.grid_sizes):
            raise ValueError("grid_sizes must contain positive integers")
        if len(set(self.grid_sizes)) != len(self.grid_sizes):
            raise ValueError("grid_sizes must not contain duplicates")
        if not self.eps_values:
            raise ValueError("eps_values must not be empty")
        if not self.k0a_values or any(value <= 0.0 for value in self.k0a_values):
            raise ValueError("k0a_values must contain positive values")
        if self.radius <= 0.0 or self.domain_length <= 2.0 * self.radius:
            raise ValueError("domain_length must strictly exceed the sphere diameter")
        if not self.solver_names:
            raise ValueError("solver_names must not be empty")
        if self.sim_coarse_size <= 0 or self.max_iter <= 0 or self.rtol <= 0.0:
            raise ValueError("SIM coarse size, max_iter, and rtol must be positive")
        if self.divergence_guard <= 1.0:
            raise ValueError("divergence_guard must exceed one")
        if self.rcs_n_phi < 8:
            raise ValueError("rcs_n_phi must be at least 8")
        if self.rcs_plane not in {"xy", "xz", "yz"}:
            raise ValueError("rcs_plane must be xy, xz, or yz")
        unknown = set(self.field_reference_grids) - set(self.grid_sizes)
        if unknown:
            raise ValueError(
                f"field_reference_grids must be drawn from grid_sizes, got {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ValidationStudyConfig:
    mode: Literal["quick", "publication"]
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    solver: SolverStudyConfig = field(default_factory=SolverStudyConfig)
    mie: MieStudyConfig = field(default_factory=MieStudyConfig)
    output_root: Path = Path("experiments/outputs/em_validation")

    @classmethod
    def quick(
        cls,
        *,
        output_root: str | Path = "experiments/outputs/em_validation_quick",
        device: Literal["cpu", "cuda", "auto"] = "cpu",
    ) -> "ValidationStudyConfig":
        return cls(
            mode="quick",
            runtime=RuntimeConfig(device=device, precision="double"),
            solver=SolverStudyConfig(
                case_keys=("anisotropic_ellipsoid",),
                grid_size=8,
                coarse_size=3,
                max_iter=80,
                rtol=1e-5,
                rcs_n_phi=36,
                rcs_planes=("xz",),
                save_solutions=True,
            ),
            mie=MieStudyConfig(
                grid_sizes=(6, 8),
                eps_values=(1.2 + 0.0j, 1.5 + 0.0j),
                k0a_values=(0.5, 1.0),
                radius=0.25,
                domain_length=1.0,
                solver_names=("SIM", "BiCGStab", "TwoStep"),
                sim_coarse_size=3,
                max_iter=80,
                rtol=1e-5,
                rcs_n_phi=36,
                rcs_plane="xz",
                field_reference_grids=(6,),
                compare_farfield_backends_on_smallest_grid=True,
                save_solutions=False,
            ),
            output_root=Path(output_root),
        )

    @classmethod
    def publication(
        cls,
        *,
        output_root: str | Path = "experiments/outputs/em_validation_publication",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> "ValidationStudyConfig":
        return cls(
            mode="publication",
            runtime=RuntimeConfig(device=device, precision="double"),
            solver=SolverStudyConfig(
                case_keys=(
                    "anisotropic_ellipsoid",
                    "local_inclusion_stable",
                    "local_inclusion_stress",
                ),
                coarse_size_overrides=(("local_inclusion_stress", 5),),
            ),
            mie=MieStudyConfig(),
            output_root=Path(output_root),
        )
