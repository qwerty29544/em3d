from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from em3d.experiments.spectral_transfer import (
    EnsembleParameterRun,
    FineTransferRun,
    build_ensemble_parameter,
    build_spectral_case,
    local_inclusion_case,
    run_parameter_transfer,
)
from em3d.geometry import SamplingMode
from em3d.operator import PreparedEMKernel

from .artifacts import ArtifactStore
from .common import fine_row, make_backend, rows_by_key, study_solver_config
from .config import SpectralStudyConfig
from .geometry_resolution import GeometryResolutionStudyResult


@dataclass(frozen=True)
class EnsembleSideResult:
    inclusion_side: float
    ensembles: tuple[EnsembleParameterRun, ...]
    fine_runs: tuple[FineTransferRun, ...]


@dataclass(frozen=True)
class EnsembleTransferStudyResult:
    config: SpectralStudyConfig
    sides: tuple[EnsembleSideResult, ...]
    spectral_rows: tuple[dict, ...]
    fine_rows: tuple[dict, ...]
    comparison_rows: tuple[dict, ...]
    diagnostic_rows: tuple[dict, ...]
    by_ensemble_rows: tuple[dict, ...]
    overall_rows: tuple[dict, ...]


def _ensemble_name(levels: tuple[int, ...]) -> str:
    return "+".join(str(level) for level in levels)


def _classification_is_contracting(classification: str) -> bool:
    return classification in {"converged", "contracting_unresolved"}


def run_ensemble_transfer_study(
    config: SpectralStudyConfig,
    geometry_result: GeometryResolutionStudyResult,
    *,
    store: ArtifactStore | None = None,
    include_fine: bool = True,
    finalize: bool = True,
) -> EnsembleTransferStudyResult:
    """Run E5 using the spectra already computed by E4.

    The function deliberately preserves both the strict convergence result
    (the requested residual tolerance was attained) and the weaker asymptotic
    classification.  This prevents a slowly contracting run from being
    silently promoted to a fully converged calculation.
    """

    if store is None:
        store = ArtifactStore(config.output_root)
        store.write_json("config.json", config)

    work_backend = make_backend(config)
    geometry_by_side = {
        float(result.inclusion_side): result for result in geometry_result.sides
    }
    geometry_summary = rows_by_key(
        geometry_result.summary_rows, "inclusion_side", "N_H"
    )

    side_results: list[EnsembleSideResult] = []
    spectral_rows: list[dict] = []
    fine_rows: list[dict] = []
    comparison_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    prepared_fine: PreparedEMKernel | None = None

    for side_value in config.geometry.inclusion_sides:
        side = float(side_value)
        if side not in geometry_by_side:
            raise KeyError(f"geometry result does not contain inclusion side {side}")
        source = geometry_by_side[side]
        control = source.spectra[config.grids.control_size]

        ensemble_parameters: list[EnsembleParameterRun] = []
        circles = {}
        for levels in config.geometry.ensembles:
            parameter = build_ensemble_parameter(
                source.spectra,
                levels,
                control=control,
            )
            ensemble_parameters.append(parameter)
            label = _ensemble_name(parameter.levels)
            circle = parameter.circle
            assessment = parameter.control_assessment
            largest_level = max(parameter.levels)
            largest_single = geometry_summary[(side, largest_level)]
            row = {
                "experiment": "E5",
                "inclusion_side": side,
                "ensemble": label,
                "levels": list(parameter.levels),
                "largest_level": largest_level,
                "parameter_exists": circle is not None,
                "mu_real": circle.mu.real if circle is not None else np.nan,
                "mu_imag": circle.mu.imag if circle is not None else np.nan,
                "radius": circle.radius if circle is not None else np.nan,
                "q_ensemble": circle.q if circle is not None else np.nan,
                "Delta_ensemble": circle.margin if circle is not None else np.nan,
                "epsilon_control": (
                    assessment.directed_error if assessment is not None else np.nan
                ),
                "eta_control": (
                    assessment.normalized_error if assessment is not None else np.nan
                ),
                "q_control_mu_ensemble": (
                    assessment.target_factor if assessment is not None else np.nan
                ),
                "q_bound_control": (
                    assessment.target_factor_bound if assessment is not None else np.nan
                ),
                "certified_control": bool(
                    assessment.certified if assessment is not None else False
                ),
                "delta_latest": parameter.delta_latest,
                "delta_max": parameter.delta_max,
                "q_inflated_latest": parameter.q_inflated_latest,
                "q_inflated_max": parameter.q_inflated_max,
                "inflated_latest_safe": parameter.inflated_latest_safe,
                "inflated_max_safe": parameter.inflated_max_safe,
                "maximum_resolution": parameter.maximum_resolution,
                "geometry_two_cell_rule": parameter.geometry_two_cell_rule,
                "adequate_geometry_levels": list(parameter.adequate_geometry_levels),
                "resolution_by_level": parameter.resolution_by_level,
                "largest_single_q_H": largest_single["q_H"],
                "largest_single_q_control": largest_single["q_c_mu_H"],
                "largest_single_certified_control": largest_single[
                    "certified_control"
                ],
                "largest_single_converged": largest_single["fine_converged"],
                "largest_single_classification": largest_single[
                    "fine_classification"
                ],
                "largest_single_ratio": largest_single[
                    "observed_residual_ratio"
                ],
            }
            spectral_rows.append(row)
            store.write_npz(
                f"raw/E5_ensemble_hulls/a_{side:g}/ensemble_{label}.npz",
                hull=parameter.ensemble.localization.hull,
                levels=np.asarray(parameter.levels, dtype=np.int64),
                mu=np.asarray(
                    [circle.mu if circle is not None else np.nan + 1j * np.nan],
                    dtype=np.complex128,
                ),
                radius=np.asarray(
                    [circle.radius if circle is not None else np.nan],
                    dtype=np.float64,
                ),
            )
            if circle is not None:
                circles[f"ENS_{label}"] = circle

        side_fine_runs: tuple[FineTransferRun, ...] = ()
        if include_fine and circles:
            definition = local_inclusion_case(side, k0=4.0)
            fine_case = build_spectral_case(
                definition,
                grid_shape=config.grids.fine_size,
                backend=work_backend,
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            if prepared_fine is None:
                prepared_fine = PreparedEMKernel.build(
                    fine_case.problem.grid,
                    k=fine_case.problem.k0,
                )
            side_fine_runs = run_parameter_transfer(
                fine_case,
                circles,
                solver_config=study_solver_config(config),
                prepared_kernel=prepared_fine,
                retain_solution=False,
            )
            for run in side_fine_runs:
                label = run.parameter_label.removeprefix("ENS_")
                fine_rows.append(
                    fine_row(
                        run,
                        extra={
                            "experiment": "E5",
                            "inclusion_side": side,
                            "ensemble": label,
                        },
                    )
                )
                store.write_json(
                    f"raw/E5_residual_histories/a_{side:g}_ensemble_{label}.json",
                    run.solver_result.residual_history,
                )

        fine_by_label = {
            run.parameter_label.removeprefix("ENS_"): run
            for run in side_fine_runs
        }
        spectral_by_label = {
            row["ensemble"]: row
            for row in spectral_rows
            if row["inclusion_side"] == side
        }
        for parameter in ensemble_parameters:
            label = _ensemble_name(parameter.levels)
            spec = spectral_by_label[label]
            fine = fine_by_label.get(label)
            largest_level = max(parameter.levels)
            baseline = geometry_summary[(side, largest_level)]
            ensemble_classification = (
                fine.convergence.classification.value if fine is not None else "not_run"
            )
            ensemble_converged = bool(
                fine.solver_result.converged if fine is not None else False
            )
            ensemble_contracting = bool(
                _classification_is_contracting(ensemble_classification)
            )
            single_contracting = bool(
                _classification_is_contracting(
                    str(baseline["fine_classification"])
                )
            )
            row = {
                **spec,
                "ensemble_converged": ensemble_converged,
                "ensemble_contracting": ensemble_contracting,
                "ensemble_status": (
                    fine.solver_result.status if fine is not None else "not_run"
                ),
                "ensemble_classification": ensemble_classification,
                "ensemble_ratio": (
                    fine.convergence.asymptotic_ratio if fine is not None else np.nan
                ),
                "ensemble_final_residual": (
                    fine.convergence.final_residual if fine is not None else np.nan
                ),
                "ensemble_min_residual": (
                    fine.convergence.minimum_residual if fine is not None else np.nan
                ),
                "ensemble_matvec_count": (
                    fine.solver_result.matvec_count if fine is not None else 0
                ),
                "single_contracting": single_contracting,
            }
            row["ratio_improvement_vs_largest_single"] = (
                baseline["observed_residual_ratio"] - row["ensemble_ratio"]
            )
            row["recovered_strict"] = bool(
                not baseline["fine_converged"] and ensemble_converged
            )
            row["degraded_strict"] = bool(
                baseline["fine_converged"] and not ensemble_converged
            )
            row["recovered_asymptotic"] = bool(
                not single_contracting and ensemble_contracting
            )
            row["degraded_asymptotic"] = bool(
                single_contracting and not ensemble_contracting
            )
            row["geometry_rule_matches_asymptotic"] = bool(
                parameter.geometry_two_cell_rule == ensemble_contracting
            )
            row["inflated_latest_rule_matches_asymptotic"] = bool(
                parameter.inflated_latest_safe == ensemble_contracting
            )
            row["inflated_max_rule_matches_asymptotic"] = bool(
                parameter.inflated_max_safe == ensemble_contracting
            )
            comparison_rows.append(row)
            diagnostic_rows.extend(
                (
                    {
                        "experiment": "E5",
                        "inclusion_side": side,
                        "ensemble": label,
                        "rule": "geometry_two_cells",
                        "prediction_stable": parameter.geometry_two_cell_rule,
                        "observed_contracting": ensemble_contracting,
                        "match": row["geometry_rule_matches_asymptotic"],
                    },
                    {
                        "experiment": "E5",
                        "inclusion_side": side,
                        "ensemble": label,
                        "rule": "inflated_latest",
                        "prediction_stable": parameter.inflated_latest_safe,
                        "observed_contracting": ensemble_contracting,
                        "match": row[
                            "inflated_latest_rule_matches_asymptotic"
                        ],
                    },
                    {
                        "experiment": "E5",
                        "inclusion_side": side,
                        "ensemble": label,
                        "rule": "inflated_max",
                        "prediction_stable": parameter.inflated_max_safe,
                        "observed_contracting": ensemble_contracting,
                        "match": row["inflated_max_rule_matches_asymptotic"],
                    },
                )
            )

        side_results.append(
            EnsembleSideResult(
                inclusion_side=side,
                ensembles=tuple(ensemble_parameters),
                fine_runs=side_fine_runs,
            )
        )

    frame = pd.DataFrame(comparison_rows)
    if len(frame):
        by_ensemble = (
            frame.groupby("ensemble")
            .agg(
                experiments=("inclusion_side", "size"),
                ensemble_converged=("ensemble_converged", "sum"),
                ensemble_contracting=("ensemble_contracting", "sum"),
                recovered_strict=("recovered_strict", "sum"),
                degraded_strict=("degraded_strict", "sum"),
                recovered_asymptotic=("recovered_asymptotic", "sum"),
                degraded_asymptotic=("degraded_asymptotic", "sum"),
                median_ratio=("ensemble_ratio", "median"),
                median_ratio_improvement=(
                    "ratio_improvement_vs_largest_single",
                    "median",
                ),
                median_q_ensemble=("q_ensemble", "median"),
                median_q_inflated_max=("q_inflated_max", "median"),
            )
            .reset_index()
        )
        overall = pd.DataFrame(
            [
                {
                    "experiments": len(frame),
                    "ensemble_converged": int(frame["ensemble_converged"].sum()),
                    "ensemble_contracting": int(
                        frame["ensemble_contracting"].sum()
                    ),
                    "recovered_strict": int(frame["recovered_strict"].sum()),
                    "degraded_strict": int(frame["degraded_strict"].sum()),
                    "recovered_asymptotic": int(
                        frame["recovered_asymptotic"].sum()
                    ),
                    "degraded_asymptotic": int(
                        frame["degraded_asymptotic"].sum()
                    ),
                    "geometry_rule_accuracy": float(
                        frame["geometry_rule_matches_asymptotic"].mean()
                    ),
                    "inflated_latest_rule_accuracy": float(
                        frame[
                            "inflated_latest_rule_matches_asymptotic"
                        ].mean()
                    ),
                    "inflated_max_rule_accuracy": float(
                        frame["inflated_max_rule_matches_asymptotic"].mean()
                    ),
                }
            ]
        )
    else:
        by_ensemble = pd.DataFrame()
        overall = pd.DataFrame()

    by_ensemble_rows = tuple(by_ensemble.to_dict(orient="records"))
    overall_rows = tuple(overall.to_dict(orient="records"))
    store.write_rows("tables/E5_ensemble_spectral.csv", spectral_rows)
    if fine_rows:
        store.write_rows("tables/E5_ensemble_fine.csv", fine_rows)
    store.write_rows("tables/E5_comparison.csv", comparison_rows)
    store.write_rows("tables/E5_rule_diagnostics.csv", diagnostic_rows)
    store.write_rows("tables/E5_by_ensemble.csv", by_ensemble_rows)
    store.write_rows("tables/E5_overall_summary.csv", overall_rows)
    if finalize:
        store.finalize(config=config)

    return EnsembleTransferStudyResult(
        config=config,
        sides=tuple(side_results),
        spectral_rows=tuple(spectral_rows),
        fine_rows=tuple(fine_rows),
        comparison_rows=tuple(comparison_rows),
        diagnostic_rows=tuple(diagnostic_rows),
        by_ensemble_rows=by_ensemble_rows,
        overall_rows=overall_rows,
    )
