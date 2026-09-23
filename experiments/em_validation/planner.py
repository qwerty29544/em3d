from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from .config import (
    CudaMemoryPolicy,
    LargeGridStudyConfig,
    LargeRunProfile,
    MieJobSpec,
    RuntimeConfig,
    SolverSuiteConfig,
    StationaryGridJobSpec,
    VisualizationConfig,
)


def _partition_jobs(jobs: Sequence, *, index: int, count: int):
    if count <= 0:
        raise ValueError("batch count must be positive")
    if index < 0 or index >= count:
        raise ValueError(f"batch index must satisfy 0 <= index < {count}")
    selected = tuple(job for position, job in enumerate(jobs) if position % count == index)
    if not selected:
        raise ValueError(
            f"batch {index + 1}/{count} is empty for {len(jobs)} configured jobs"
        )
    return selected


def _main_mie_jobs() -> tuple[MieJobSpec, ...]:
    jobs: list[MieJobSpec] = []
    for eps_r in (1.5 + 0.0j, 2.25 + 0.0j, 4.0 + 0.0j):
        for k0a in (0.5, 1.0, 2.0, 4.0, 6.0):
            for grid_size in (24, 32, 48, 64):
                jobs.append(
                    MieJobSpec(
                        eps_r=eps_r,
                        k0a=k0a,
                        grid_size=grid_size,
                        tier="main",
                        true_rtol=1e-6,
                        compute_full_field_metrics=True,
                        render_field_slices=grid_size == 64,
                    )
                )
    return tuple(jobs)


def _audit_mie_jobs() -> tuple[MieJobSpec, ...]:
    jobs: list[MieJobSpec] = []
    for eps_r, k0a in (
        (1.5 + 0.0j, 1.0),
        (1.5 + 0.0j, 4.0),
        (2.25 + 0.0j, 2.0),
        (4.0 + 0.0j, 1.0),
        (4.0 + 0.0j, 4.0),
    ):
        for grid_size in (48, 64):
            jobs.append(
                MieJobSpec(
                    eps_r=eps_r,
                    k0a=k0a,
                    grid_size=grid_size,
                    tier="audit",
                    true_rtol=1e-8,
                    compute_full_field_metrics=True,
                    render_field_slices=grid_size == 64,
                    variant_label="audit_rtol1e-8",
                )
            )
    return tuple(jobs)


def _control96_jobs() -> tuple[MieJobSpec, ...]:
    return tuple(
        MieJobSpec(
            eps_r=eps_r,
            k0a=k0a,
            grid_size=96,
            tier="control",
            true_rtol=1e-6,
            compute_full_field_metrics=False,
            render_field_slices=True,
        )
        for eps_r, k0a in (
            (1.5 + 0.0j, 0.5),
            (2.25 + 0.0j, 2.0),
            (2.25 + 0.0j, 6.0),
            (4.0 + 0.0j, 4.0),
            (4.0 + 0.0j, 6.0),
        )
    )


def _control128_jobs(*, include_optional: bool) -> tuple[MieJobSpec, ...]:
    mandatory = [
        (1.5 + 0.0j, 0.5, False),
        (2.25 + 0.0j, 6.0, False),
        (4.0 + 0.0j, 4.0, False),
    ]
    if include_optional:
        mandatory.append((4.0 + 0.0j, 6.0, True))
    return tuple(
        MieJobSpec(
            eps_r=eps_r,
            k0a=k0a,
            grid_size=128,
            tier="optional" if optional else "control",
            true_rtol=1e-6,
            compute_full_field_metrics=False,
            render_field_slices=True,
            optional=optional,
        )
        for eps_r, k0a, optional in mandatory
    )


def _stationary_jobs(profile: LargeRunProfile, *, include_optional: bool):
    if profile == "smoke":
        return (
            StationaryGridJobSpec(
                case_key="anisotropic_ellipsoid",
                grid_size=8,
                render_field_slices=True,
            ),
        )
    if profile == "main64":
        return (
            StationaryGridJobSpec(
                case_key="anisotropic_ellipsoid",
                grid_size=64,
                tier="main",
                render_field_slices=True,
                parameter_strategy="coarse_ensemble",
                coarse_sizes=(5, 6, 7),
                variant_label="ensemble_5_6_7",
            ),
            StationaryGridJobSpec(
                case_key="local_inclusion_stable",
                grid_size=64,
                tier="main",
                render_field_slices=True,
                parameter_strategy="coarse_ensemble",
                coarse_sizes=(5, 6, 7),
                variant_label="ensemble_5_6_7",
            ),
            # The two jobs below are deliberately the same physical problem.
            # They isolate the effect of the spectral parameter construction:
            # a single N_H=5 localization is unstable, while the 5+6+7
            # ensemble is expected to restore a safe parameter.
            StationaryGridJobSpec(
                case_key="local_inclusion_stress",
                grid_size=64,
                tier="main",
                render_field_slices=False,
                parameter_strategy="single_coarse",
                coarse_sizes=(5,),
                variant_label="single_N5",
            ),
            StationaryGridJobSpec(
                case_key="local_inclusion_stress",
                grid_size=64,
                tier="main",
                render_field_slices=True,
                parameter_strategy="coarse_ensemble",
                coarse_sizes=(5, 6, 7),
                variant_label="ensemble_5_6_7",
            ),
        )
    if profile == "control96":
        return tuple(
            StationaryGridJobSpec(
                case_key=case_key,
                grid_size=96,
                tier="control",
                render_field_slices=True,
                parameter_strategy="coarse_ensemble",
                coarse_sizes=(5, 6, 7),
                variant_label="ensemble_5_6_7",
            )
            for case_key in (
                "anisotropic_ellipsoid",
                "local_inclusion_stable",
                "local_inclusion_stress",
            )
        )
    if profile == "control128":
        keys = ["anisotropic_ellipsoid", "local_inclusion_stable"]
        if include_optional:
            keys.append("local_inclusion_stress")
        return tuple(
            StationaryGridJobSpec(
                case_key=case_key,
                grid_size=128,
                tier="optional" if case_key == "local_inclusion_stress" else "control",
                render_field_slices=True,
                optional=case_key == "local_inclusion_stress",
                parameter_strategy="coarse_ensemble",
                coarse_sizes=(5, 6, 7),
                variant_label="ensemble_5_6_7",
            )
            for case_key in keys
        )
    return ()


def build_large_grid_config(
    profile: LargeRunProfile,
    *,
    output_root: str | Path,
    device: str = "cuda",
    precision: str = "double",
    batch_index: int = 0,
    batch_count: int = 1,
    include_optional: bool = False,
    resume: bool = True,
    render_progress: bool = True,
) -> LargeGridStudyConfig:
    if profile == "smoke":
        mie_jobs = (
            MieJobSpec(
                eps_r=1.2 + 0.0j,
                k0a=0.5,
                grid_size=8,
                tier="main",
                true_rtol=1e-5,
                compute_full_field_metrics=True,
                render_field_slices=True,
            ),
        )
        solver_suite = SolverSuiteConfig(
            true_rtol=1e-5,
            audit_rtol=1e-6,
            max_iter=150,
            max_operator_actions=(
                ("SIM", 250),
                ("BiCGStab", 250),
                ("TwoStep", 500),
            ),
            sim_coarse_sizes=(3, 4),
        )
        rcs_n_phi = 36
    elif profile == "main64":
        mie_jobs = _main_mie_jobs()
        solver_suite = SolverSuiteConfig()
        rcs_n_phi = 720
    elif profile == "audit":
        mie_jobs = _audit_mie_jobs()
        solver_suite = SolverSuiteConfig(true_rtol=1e-8)
        rcs_n_phi = 720
    elif profile == "control96":
        mie_jobs = _control96_jobs()
        solver_suite = SolverSuiteConfig()
        rcs_n_phi = 720
    elif profile == "control128":
        mie_jobs = _control128_jobs(include_optional=include_optional)
        solver_suite = SolverSuiteConfig()
        rcs_n_phi = 720
    elif profile == "merge":
        mie_jobs = ()
        solver_suite = SolverSuiteConfig()
        rcs_n_phi = 720
    else:
        raise ValueError(f"unknown profile {profile!r}")

    if mie_jobs and batch_count > 1:
        mie_jobs = _partition_jobs(mie_jobs, index=batch_index, count=batch_count)
    stationary_jobs = _stationary_jobs(profile, include_optional=include_optional)
    if stationary_jobs and batch_count > 1:
        stationary_jobs = _partition_jobs(
            stationary_jobs, index=batch_index, count=batch_count
        )

    runtime = RuntimeConfig(
        device=device,  # type: ignore[arg-type]
        precision=precision,  # type: ignore[arg-type]
        progress=render_progress,
        clear_cuda_cache_between_cases=True,
    )
    memory_policy = (
        CudaMemoryPolicy(usable_fraction=0.85, reserve_gib=1.0)
        if profile == "control128"
        else CudaMemoryPolicy()
    )
    visualization = (
        VisualizationConfig(
            field_planes=("xz",),
            rcs_planes=("xz",),
            rcs_coordinate_systems=("cartesian", "polar"),
            save_formats=("png",),
            create_subtree_archives=False,
        )
        if profile == "smoke"
        else VisualizationConfig()
    )
    return LargeGridStudyConfig(
        profile=profile,
        runtime=runtime,
        solver_suite=solver_suite,
        visualization=visualization,
        memory_policy=memory_policy,
        mie_jobs=tuple(mie_jobs),
        stationary_jobs=tuple(stationary_jobs),
        radius=0.25 if profile == "smoke" else 0.30,
        domain_length=1.0,
        rcs_n_phi=rcs_n_phi,
        require_mie_nearfield_gate=True,
        output_root=Path(output_root),
        resume=resume,
    )


__all__ = ["build_large_grid_config"]
