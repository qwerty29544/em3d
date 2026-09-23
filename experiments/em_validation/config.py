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


LargeRunProfile = Literal["smoke", "main64", "audit", "control96", "control128", "merge"]
JobTier = Literal["main", "audit", "control", "optional"]


@dataclass(frozen=True)
class SolverSuiteConfig:
    """Common solver protocol used by the large-grid Chapter 4 series."""

    solver_names: tuple[str, ...] = ("SIM", "BiCGStab", "TwoStep")
    reference_solver: str = "BiCGStab"
    true_rtol: float = 1e-6
    audit_rtol: float = 1e-8
    max_iter: int = 2000
    divergence_guard: float = 1e6
    max_operator_actions: tuple[tuple[str, int], ...] = (
        ("SIM", 1500),
        ("BiCGStab", 1500),
        ("TwoStep", 3000),
    )
    sim_parameter_strategy: Literal["coarse_ensemble", "single_coarse"] = (
        "coarse_ensemble"
    )
    sim_coarse_sizes: tuple[int, ...] = (5, 6, 7)
    require_true_residual: bool = True

    def __post_init__(self) -> None:
        allowed = {"SIM", "BiCGStab", "TwoStep"}
        if not self.solver_names or set(self.solver_names) - allowed:
            raise ValueError(
                "solver_names must be a non-empty subset of SIM, BiCGStab, TwoStep"
            )
        if self.reference_solver not in self.solver_names:
            raise ValueError("reference_solver must be present in solver_names")
        if self.true_rtol <= 0.0 or self.audit_rtol <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.divergence_guard <= 1.0:
            raise ValueError("divergence_guard must exceed one")
        action_map = dict(self.max_operator_actions)
        if set(action_map) != set(self.solver_names):
            raise ValueError(
                "max_operator_actions must contain exactly the configured solvers"
            )
        if any(value <= 0 for value in action_map.values()):
            raise ValueError("operator-action budgets must be positive")
        if not self.sim_coarse_sizes or any(v <= 0 for v in self.sim_coarse_sizes):
            raise ValueError("sim_coarse_sizes must contain positive integers")

    def action_budget(self, solver_name: str) -> int:
        return int(dict(self.max_operator_actions)[solver_name])


@dataclass(frozen=True)
class VisualizationConfig:
    field_planes: tuple[str, ...] = ("xy", "xz", "yz")
    rcs_planes: tuple[str, ...] = ("xy", "xz", "yz")
    rcs_coordinate_systems: tuple[str, ...] = ("cartesian", "polar")
    field_component: int = 0
    phase_mask_fraction: float = 1e-6
    save_formats: tuple[str, ...] = ("png", "pdf", "svg")
    create_subtree_archives: bool = True
    render_field_policy: Literal[
        "finest_common_qualified_grid", "every_qualified_grid"
    ] = "finest_common_qualified_grid"

    def __post_init__(self) -> None:
        allowed_planes = {"xy", "xz", "yz"}
        if not self.field_planes or set(self.field_planes) - allowed_planes:
            raise ValueError("field_planes must contain xy, xz, or yz")
        if not self.rcs_planes or set(self.rcs_planes) - allowed_planes:
            raise ValueError("rcs_planes must contain xy, xz, or yz")
        if set(self.rcs_coordinate_systems) - {"cartesian", "polar"}:
            raise ValueError("unsupported RCS coordinate system")
        if self.field_component not in {0, 1, 2}:
            raise ValueError("field_component must be 0, 1, or 2")
        if self.phase_mask_fraction <= 0.0:
            raise ValueError("phase_mask_fraction must be positive")


@dataclass(frozen=True)
class CudaMemoryPolicy:
    usable_fraction: float = 0.78
    reserve_gib: float = 2.0
    isolate_large_jobs: bool = True
    probe_grid_sizes: tuple[int, ...] = (96, 128)
    oom_behavior: Literal["skip_and_record", "raise"] = "skip_and_record"
    clear_pool_before_case: bool = True
    clear_fft_plan_cache_before_case: bool = True
    adjoint_storage: Literal["explicit", "derived"] = "derived"
    kernel_build_strategy: Literal["standard", "streamed"] = "streamed"

    def __post_init__(self) -> None:
        if not 0.0 < self.usable_fraction <= 1.0:
            raise ValueError("usable_fraction must lie in (0, 1]")
        if self.reserve_gib < 0.0:
            raise ValueError("reserve_gib must be non-negative")
        if any(value <= 0 for value in self.probe_grid_sizes):
            raise ValueError("probe_grid_sizes must be positive")


@dataclass(frozen=True)
class MieJobSpec:
    eps_r: complex
    k0a: float
    grid_size: int
    tier: JobTier = "main"
    true_rtol: float = 1e-6
    compute_full_field_metrics: bool = True
    render_field_slices: bool = False
    render_rcs: bool = True
    optional: bool = False
    variant_label: str = ""

    @property
    def key(self) -> str:
        eps = complex(self.eps_r)
        eps_token = f"{eps.real:g}".replace("-", "m").replace(".", "p")
        if abs(eps.imag) > 1e-15:
            eps_token += f"_i{eps.imag:g}".replace("-", "m").replace(".", "p")
        k_token = f"{float(self.k0a):g}".replace("-", "m").replace(".", "p")
        suffix = f"_{self.variant_label}" if self.variant_label else ""
        return f"mie_eps{eps_token}_k0a{k_token}_N{int(self.grid_size)}{suffix}"


@dataclass(frozen=True)
class StationaryGridJobSpec:
    case_key: str
    grid_size: int
    tier: JobTier = "main"
    true_rtol: float = 1e-6
    render_field_slices: bool = False
    render_rcs: bool = True
    optional: bool = False
    parameter_strategy: Literal[
        "inherit", "single_coarse", "coarse_ensemble"
    ] = "inherit"
    coarse_sizes: tuple[int, ...] = ()
    variant_label: str = ""

    def __post_init__(self) -> None:
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.true_rtol <= 0.0:
            raise ValueError("true_rtol must be positive")
        if any(value <= 0 for value in self.coarse_sizes):
            raise ValueError("coarse_sizes must contain positive integers")
        if self.parameter_strategy == "single_coarse" and len(self.coarse_sizes) != 1:
            raise ValueError("single_coarse requires exactly one coarse size")
        if self.parameter_strategy == "coarse_ensemble" and not self.coarse_sizes:
            raise ValueError("coarse_ensemble requires at least one coarse size")

    def resolved_parameter_strategy(
        self, solver_suite: SolverSuiteConfig
    ) -> tuple[str, tuple[int, ...]]:
        if self.parameter_strategy == "inherit":
            return (
                solver_suite.sim_parameter_strategy,
                tuple(int(value) for value in solver_suite.sim_coarse_sizes),
            )
        return self.parameter_strategy, tuple(int(value) for value in self.coarse_sizes)

    @property
    def key(self) -> str:
        suffix = f"_{self.variant_label}" if self.variant_label else ""
        return f"{self.case_key}{suffix}_N{int(self.grid_size)}"


@dataclass(frozen=True)
class LargeGridStudyConfig:
    profile: LargeRunProfile
    runtime: RuntimeConfig
    solver_suite: SolverSuiteConfig
    visualization: VisualizationConfig
    memory_policy: CudaMemoryPolicy
    mie_jobs: tuple[MieJobSpec, ...]
    stationary_jobs: tuple[StationaryGridJobSpec, ...] = ()
    radius: float = 0.30
    domain_length: float = 1.0
    rcs_n_phi: int = 720
    require_mie_nearfield_gate: bool = True
    compare_farfield_backends: bool = True
    farfield_backend_max_grid: int = 24
    output_root: Path = Path("experiments/outputs/em_validation_large")
    resume: bool = True

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.domain_length <= 2.0 * self.radius:
            raise ValueError("domain_length must exceed the sphere diameter")
        if self.rcs_n_phi < 36:
            raise ValueError("rcs_n_phi must be at least 36")
        if self.farfield_backend_max_grid <= 0:
            raise ValueError("farfield_backend_max_grid must be positive")
        keys = [job.key for job in self.mie_jobs]
        if len(keys) != len(set(keys)):
            raise ValueError("mie_jobs contain duplicate physical/grid jobs")
        stationary_keys = [job.key for job in self.stationary_jobs]
        if len(stationary_keys) != len(set(stationary_keys)):
            raise ValueError("stationary_jobs contain duplicates")
