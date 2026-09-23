from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from em3d.experiments.spectral_transfer import (
    FineTransferRun,
    GridSpectrumRun,
    assess_control_transfer,
    build_spectral_case,
    compute_grid_spectrum,
    local_inclusion_case,
    run_parameter_transfer,
)
from em3d.geometry import SamplingMode, sampling_weights
from em3d.operator import PreparedEMKernel
from em3d.spectral import CircleLocalization, assess_transfer

from .artifacts import ArtifactStore
from .common import (
    fine_row,
    hierarchy_levels,
    localization_row,
    make_backend,
    rows_by_key,
    study_solver_config,
    transfer_row,
    write_spectrum,
)
from .config import SpectralStudyConfig
from .geometry_resolution import GeometryResolutionStudyResult


@dataclass(frozen=True)
class VolumeAveragingSideResult:
    inclusion_side: float
    averaged_spectra: dict[int, GridSpectrumRun]
    fine_runs: tuple[FineTransferRun, ...]


@dataclass(frozen=True)
class VolumeAveragingStudyResult:
    config: SpectralStudyConfig
    sides: tuple[VolumeAveragingSideResult, ...]
    geometry_rows: tuple[dict, ...]
    spectral_rows: tuple[dict, ...]
    fine_rows: tuple[dict, ...]
    comparison_rows: tuple[dict, ...]
    overall_rows: tuple[dict, ...]
    by_level_rows: tuple[dict, ...]
    by_side_rows: tuple[dict, ...]


def _fraction_diagnostics(built_case) -> dict:
    fractions = sampling_weights(
        built_case.definition.feature_geometry,
        built_case.problem.grid,
        SamplingMode.VOLUME_FRACTION,
    )
    return {
        "avg_nonzero_cells": int(np.count_nonzero(fractions > 0.0)),
        "avg_partial_cells": int(
            np.count_nonzero((fractions > 0.0) & (fractions < 1.0))
        ),
        "avg_full_cells": int(np.count_nonzero(fractions >= 1.0 - 1e-14)),
        "avg_max_fraction": float(np.max(fractions)) if fractions.size else 0.0,
    }


def run_volume_averaging_study(
    config: SpectralStudyConfig,
    geometry_result: GeometryResolutionStudyResult,
    *,
    store: ArtifactStore | None = None,
    include_fine: bool = True,
    finalize: bool = True,
) -> VolumeAveragingStudyResult:
    if store is None:
        store = ArtifactStore(config.output_root)
        store.write_json("config.json", config)

    dense_backend = make_backend(config)
    work_backend = make_backend(config)
    levels = hierarchy_levels(config)
    center_by_side = {
        float(item.inclusion_side): item for item in geometry_result.sides
    }
    center_summary = rows_by_key(
        geometry_result.summary_rows, "inclusion_side", "N_H"
    )

    side_results: list[VolumeAveragingSideResult] = []
    geometry_rows: list[dict] = []
    spectral_rows: list[dict] = []
    fine_rows: list[dict] = []
    comparison_rows: list[dict] = []
    prepared_fine: PreparedEMKernel | None = None

    for side in config.geometry.inclusion_sides:
        side = float(side)
        if side not in center_by_side:
            raise KeyError(f"geometry result does not contain inclusion side {side}")
        center_side = center_by_side[side]
        definition = local_inclusion_case(side, k0=4.0)
        averaged_spectra: dict[int, GridSpectrumRun] = {}
        fraction_by_level: dict[int, dict] = {}

        for level in levels:
            built = build_spectral_case(
                definition,
                grid_shape=level,
                backend=dense_backend,
                sampling_mode=SamplingMode.VOLUME_FRACTION,
            )
            fractions = _fraction_diagnostics(built)
            fraction_by_level[level] = fractions
            geometry_rows.append(
                {
                    "experiment": "E6",
                    "inclusion_side": side,
                    "N_H": level,
                    "avg_volume": built.geometry.represented_volume,
                    "exact_volume": built.geometry.exact_volume,
                    "avg_volume_abs_error": abs(
                        built.geometry.represented_volume
                        - built.geometry.exact_volume
                    ),
                    "avg_volume_rel_error": built.geometry.relative_volume_error,
                    "avg_cells_min_axis": min(built.geometry.cells_per_axis),
                    **fractions,
                }
            )
            run = compute_grid_spectrum(
                built,
                eigenvalue_repeats=config.eigenvalue_repeats,
            )
            averaged_spectra[level] = run
            spectral_rows.append(
                localization_row(
                    run,
                    extra={
                        "experiment": "E6",
                        "inclusion_side": side,
                        **fractions,
                    },
                )
            )
            write_spectrum(
                store,
                f"raw/E6_avg_spectra/a_{side:g}/spectrum_avg_N{level}.npz",
                run,
            )

        avg_control = averaged_spectra[config.grids.control_size]
        center_control = center_side.spectra[config.grids.control_size]
        fine_runs: tuple[FineTransferRun, ...] = ()
        if include_fine:
            fine_case = build_spectral_case(
                definition,
                grid_shape=config.grids.fine_size,
                backend=work_backend,
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            if prepared_fine is None:
                prepared_fine = PreparedEMKernel.build(
                    fine_case.problem.grid, k=fine_case.problem.k0
                )
            parameters: dict[str, CircleLocalization] = {
                f"N{level}": averaged_spectra[level].localization.circle
                for level in levels
                if averaged_spectra[level].localization.circle is not None
            }
            fine_runs = run_parameter_transfer(
                fine_case,
                parameters,
                solver_config=study_solver_config(config),
                prepared_kernel=prepared_fine,
                retain_solution=False,
            )
            for run in fine_runs:
                level = int(run.parameter_label.removeprefix("N"))
                fine_rows.append(
                    fine_row(
                        run,
                        extra={
                            "experiment": "E6",
                            "inclusion_side": side,
                            "N_H": level,
                            "parameter_source": "volume_fraction_coarse",
                        },
                    )
                )
                store.write_json(
                    f"raw/E6_residual_histories/a_{side:g}_N{level}.json",
                    run.solver_result.residual_history,
                )

        fine_by_level = {
            int(item.parameter_label.removeprefix("N")): item
            for item in fine_runs
        }
        for level in levels:
            averaged = averaged_spectra[level]
            centered = center_side.spectra[level]
            circle = averaged.localization.circle
            avg_family = assess_control_transfer(averaged, avg_control)
            avg_to_center = None
            if circle is not None:
                avg_to_center = assess_transfer(
                    averaged.localization, center_control.localization
                )
            center_row = center_summary[(side, level)]
            avg_fine = fine_by_level.get(level)
            center_circle = centered.localization.circle
            row = {
                "experiment": "E6",
                "inclusion_side": side,
                "N_H": level,
                "center_mu_real": (
                    center_circle.mu.real if center_circle is not None else np.nan
                ),
                "center_mu_imag": (
                    center_circle.mu.imag if center_circle is not None else np.nan
                ),
                "center_q_H": (
                    center_circle.q if center_circle is not None else np.nan
                ),
                "avg_mu_real": circle.mu.real if circle is not None else np.nan,
                "avg_mu_imag": circle.mu.imag if circle is not None else np.nan,
                "avg_q_H": circle.q if circle is not None else np.nan,
                "mu_shift_abs": (
                    abs(circle.mu - center_circle.mu)
                    if circle is not None and center_circle is not None
                    else np.nan
                ),
                **fraction_by_level[level],
                "avg_volume_rel_error": averaged.geometry.relative_volume_error,
                "epsilon_avg_family": (
                    avg_family.assessment.directed_error
                    if avg_family.assessment is not None
                    else np.nan
                ),
                "eta_avg_family": (
                    avg_family.assessment.normalized_error
                    if avg_family.assessment is not None
                    else np.nan
                ),
                "q_avgc_mu_avgH": (
                    avg_family.assessment.target_factor
                    if avg_family.assessment is not None
                    else np.nan
                ),
                "guaranteed_avg_family": bool(
                    avg_family.assessment.certified
                    if avg_family.assessment is not None
                    else False
                ),
                "epsilon_center_target": (
                    avg_to_center.directed_error
                    if avg_to_center is not None
                    else np.nan
                ),
                "eta_center_target": (
                    avg_to_center.normalized_error
                    if avg_to_center is not None
                    else np.nan
                ),
                "q_center_c_mu_avgH": (
                    avg_to_center.target_factor
                    if avg_to_center is not None
                    else np.nan
                ),
                "q_bound_center_target": (
                    avg_to_center.target_factor_bound
                    if avg_to_center is not None
                    else np.nan
                ),
                "guaranteed_center_target": bool(
                    avg_to_center.certified if avg_to_center is not None else False
                ),
                "center_converged": bool(center_row["fine_converged"]),
                "center_status": center_row["fine_status"],
                "center_classification": center_row["fine_classification"],
                "center_ratio": center_row["observed_residual_ratio"],
                "center_matvec_count": center_row["fine_matvec_count"],
                "center_q_c": center_row["q_c_mu_H"],
                "center_eta_c_H": center_row["eta"],
                "center_guaranteed_c": center_row["certified_control"],
                "coarse_cells_min_axis": center_row["coarse_cells_min_axis"],
                "coarse_cells_total": center_row["coarse_cells_total"],
                "coarse_volume_error": center_row["coarse_volume_error"],
                "avg_converged": bool(
                    avg_fine.solver_result.converged
                    if avg_fine is not None
                    else False
                ),
                "avg_status": (
                    avg_fine.solver_result.status
                    if avg_fine is not None
                    else "not_run"
                ),
                "avg_classification": (
                    avg_fine.convergence.classification.value
                    if avg_fine is not None
                    else "not_run"
                ),
                "avg_ratio": (
                    avg_fine.convergence.asymptotic_ratio
                    if avg_fine is not None
                    else np.nan
                ),
                "avg_matvec_count": (
                    avg_fine.solver_result.matvec_count
                    if avg_fine is not None
                    else 0
                ),
                "avg_min_residual": (
                    avg_fine.convergence.minimum_residual
                    if avg_fine is not None
                    else np.nan
                ),
                "avg_final_residual": (
                    avg_fine.convergence.final_residual
                    if avg_fine is not None
                    else np.nan
                ),
            }
            row["ratio_improvement"] = row["center_ratio"] - row["avg_ratio"]
            row["recovered"] = bool(
                not row["center_converged"] and row["avg_converged"]
            )
            row["degraded"] = bool(
                row["center_converged"] and not row["avg_converged"]
            )
            row["both_converged"] = bool(
                row["center_converged"] and row["avg_converged"]
            )
            row["both_failed"] = bool(
                not row["center_converged"] and not row["avg_converged"]
            )
            row["status_code"] = (
                2
                if row["recovered"]
                else 1
                if row["both_converged"]
                else -1
                if row["degraded"]
                else 0
            )
            comparison_rows.append(row)

        side_results.append(
            VolumeAveragingSideResult(
                inclusion_side=side,
                averaged_spectra=averaged_spectra,
                fine_runs=fine_runs,
            )
        )

    frame = pd.DataFrame(comparison_rows)
    if len(frame):
        overall = pd.DataFrame(
            [
                {
                    "experiments": len(frame),
                    "center_converged": int(frame["center_converged"].sum()),
                    "avg_converged": int(frame["avg_converged"].sum()),
                    "recovered": int(frame["recovered"].sum()),
                    "degraded": int(frame["degraded"].sum()),
                    "both_converged": int(frame["both_converged"].sum()),
                    "both_failed": int(frame["both_failed"].sum()),
                    "median_ratio_improvement": float(
                        np.nanmedian(frame["ratio_improvement"])
                    ),
                }
            ]
        )
        by_level = (
            frame.groupby("N_H")
            .agg(
                experiments=("inclusion_side", "size"),
                center_converged=("center_converged", "sum"),
                avg_converged=("avg_converged", "sum"),
                recovered=("recovered", "sum"),
                degraded=("degraded", "sum"),
                median_ratio_improvement=("ratio_improvement", "median"),
                median_mu_shift=("mu_shift_abs", "median"),
            )
            .reset_index()
        )
        by_side = (
            frame.groupby("inclusion_side")
            .agg(
                experiments=("N_H", "size"),
                center_converged=("center_converged", "sum"),
                avg_converged=("avg_converged", "sum"),
                recovered=("recovered", "sum"),
                degraded=("degraded", "sum"),
                median_ratio_improvement=("ratio_improvement", "median"),
            )
            .reset_index()
        )
    else:
        overall = by_level = by_side = pd.DataFrame()

    overall_rows = tuple(overall.to_dict(orient="records"))
    by_level_rows = tuple(by_level.to_dict(orient="records"))
    by_side_rows = tuple(by_side.to_dict(orient="records"))
    store.write_rows("tables/E6_geometry_validation.csv", geometry_rows)
    store.write_rows("tables/E6_avg_spectral.csv", spectral_rows)
    if fine_rows:
        store.write_rows("tables/E6_avg_fine.csv", fine_rows)
    store.write_rows("tables/E6_comparison.csv", comparison_rows)
    store.write_rows("tables/E6_overall_summary.csv", overall_rows)
    store.write_rows("tables/E6_by_NH.csv", by_level_rows)
    store.write_rows("tables/E6_by_side.csv", by_side_rows)
    if finalize:
        store.finalize(config=config)

    return VolumeAveragingStudyResult(
        config=config,
        sides=tuple(side_results),
        geometry_rows=tuple(geometry_rows),
        spectral_rows=tuple(spectral_rows),
        fine_rows=tuple(fine_rows),
        comparison_rows=tuple(comparison_rows),
        overall_rows=overall_rows,
        by_level_rows=by_level_rows,
        by_side_rows=by_side_rows,
    )
