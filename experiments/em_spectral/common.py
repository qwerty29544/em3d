from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.experiments.spectral_transfer import (
    ControlTransferRun,
    FineTransferRun,
    GridSpectrumRun,
    SamplingMetrics,
)
from em3d.solvers import SolverConfig

from .artifacts import ArtifactStore
from .config import SpectralStudyConfig


def make_backend(config: SpectralStudyConfig) -> Backend:
    precision = (
        Precision.SINGLE
        if config.runtime.precision == "single"
        else Precision.DOUBLE
    )
    if config.runtime.device == "cpu":
        return Backend.numpy(precision)
    if config.runtime.device == "cuda":
        return Backend.cupy(precision)
    return Backend.auto(precision)


def release_backend_memory(backend: Backend) -> None:
    """Synchronize and release cached CUDA blocks when a study changes grid."""

    if backend.device == "cuda":
        backend.xp.cuda.Stream.null.synchronize()
        backend.xp.get_default_memory_pool().free_all_blocks()


def cpu_backend(config: SpectralStudyConfig) -> Backend:
    precision = (
        Precision.SINGLE
        if config.runtime.precision == "single"
        else Precision.DOUBLE
    )
    return Backend.numpy(precision)


def hierarchy_levels(config: SpectralStudyConfig) -> tuple[int, ...]:
    return tuple(
        sorted(set(config.grids.coarse_sizes + (config.grids.control_size,)))
    )


def study_solver_config(config: SpectralStudyConfig) -> SolverConfig:
    return SolverConfig(
        max_iter=config.iteration.max_iter,
        rtol=config.iteration.rtol,
        divergence_guard=config.iteration.divergence_guard,
    )


def localization_row(
    run: GridSpectrumRun,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict:
    circle = run.localization.circle
    row = {
        "case": run.case_key,
        "N": run.grid_shape[0],
        "matrix_dimension": run.matrix_dimension,
        "status": run.localization.status.value,
        "origin_in_hull": run.localization.origin_in_hull,
        "origin_distance": run.localization.origin_distance,
        "mu_real": circle.mu.real if circle else np.nan,
        "mu_imag": circle.mu.imag if circle else np.nan,
        "radius": circle.radius if circle else np.nan,
        "q": circle.q if circle else np.nan,
        "margin": circle.margin if circle else np.nan,
        "feature_cells": run.geometry.cells_total,
        "feature_cells_x": run.geometry.cells_per_axis[0],
        "feature_cells_y": run.geometry.cells_per_axis[1],
        "feature_cells_z": run.geometry.cells_per_axis[2],
        "feature_cells_min_axis": min(run.geometry.cells_per_axis),
        "feature_volume": run.geometry.represented_volume,
        "feature_exact_volume": run.geometry.exact_volume,
        "feature_volume_error": run.geometry.relative_volume_error,
        "sampling_mode": run.geometry.sampling_mode.value,
        "build_seconds": run.matrix_build_seconds,
        "eig_seconds_median": run.median_eigenvalue_seconds,
        "spectrum_backend": run.backend_device,
    }
    if extra:
        row.update(extra)
    return row


def transfer_row(
    run: ControlTransferRun,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict:
    assessment = run.assessment
    row = {
        "case": run.coarse.case_key,
        "N_H": run.coarse.grid_shape[0],
        "N_c": run.target.grid_shape[0],
        "parameter_exists": assessment is not None,
        "coarse_status": run.coarse.localization.status.value,
    }
    if assessment is None:
        row.update(
            {
                "epsilon_directed": np.nan,
                "epsilon_hausdorff": np.nan,
                "q_H": np.nan,
                "Delta_H": np.nan,
                "eta": np.nan,
                "q_c_mu_H": np.nan,
                "q_bound_c": np.nan,
                "certified": False,
            }
        )
    else:
        row.update(
            {
                "epsilon_directed": assessment.directed_error,
                "epsilon_hausdorff": assessment.hausdorff_distance,
                "q_H": assessment.coarse_factor,
                "Delta_H": assessment.coarse_margin,
                "eta": assessment.normalized_error,
                "q_c_mu_H": assessment.target_factor,
                "q_bound_c": assessment.target_factor_bound,
                "certified": assessment.certified,
            }
        )
    if extra:
        row.update(extra)
    return row


def fine_row(
    run: FineTransferRun,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict:
    row = {
        "case": run.case_key,
        "parameter": run.parameter_label,
        "mu_real": run.circle.mu.real,
        "mu_imag": run.circle.mu.imag,
        "q_H": run.circle.q,
        "converged": run.solver_result.converged,
        "status": run.solver_result.status,
        "iterations": run.solver_result.iterations,
        "matvec_count": run.solver_result.matvec_count,
        "final_residual": run.convergence.final_residual,
        "minimum_residual": run.convergence.minimum_residual,
        "minimum_residual_iteration": run.convergence.minimum_residual_iteration,
        "asymptotic_ratio": run.convergence.asymptotic_ratio,
        "classification": run.convergence.classification.value,
        "observations_used": run.convergence.observations_used,
        "elapsed_seconds": run.elapsed_seconds,
    }
    if extra:
        row.update(extra)
    return row


def sampling_row(metrics: SamplingMetrics, *, prefix: str = "") -> dict:
    return {f"{prefix}{key}": value for key, value in asdict(metrics).items()}


def write_spectrum(
    store: ArtifactStore,
    relative_path: str | Path,
    run: GridSpectrumRun,
) -> None:
    circle = run.localization.circle
    store.write_npz(
        relative_path,
        spectrum=run.localization.spectrum,
        hull=run.localization.hull,
        mu=np.asarray(
            [circle.mu if circle is not None else np.nan + 1j * np.nan],
            dtype=np.complex128,
        ),
        radius=np.asarray(
            [circle.radius if circle is not None else np.nan], dtype=np.float64
        ),
        q=np.asarray(
            [circle.q if circle is not None else np.nan], dtype=np.float64
        ),
        margin=np.asarray(
            [circle.margin if circle is not None else np.nan], dtype=np.float64
        ),
        origin_in_hull=np.asarray([run.localization.origin_in_hull], dtype=bool),
        origin_distance=np.asarray(
            [run.localization.origin_distance], dtype=np.float64
        ),
    )


def rows_by_key(rows: Iterable[Mapping[str, object]], *keys: str) -> dict[tuple, dict]:
    result: dict[tuple, dict] = {}
    for row in rows:
        materialized = dict(row)
        key = tuple(materialized[name] for name in keys)
        result[key] = materialized
    return result
