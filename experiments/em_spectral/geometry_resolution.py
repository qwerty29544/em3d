from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from em3d.experiments.spectral_transfer import (
    ControlTransferRun,
    FineTransferRun,
    GridSpectrumRun,
    assess_control_transfer,
    build_spectral_case,
    compute_grid_spectrum,
    local_inclusion_case,
    run_parameter_transfer,
)
from em3d.geometry import SamplingMode
from em3d.operator import PreparedEMKernel
from em3d.spectral import CircleLocalization

from .artifacts import ArtifactStore
from .common import (
    fine_row,
    hierarchy_levels,
    localization_row,
    make_backend,
    study_solver_config,
    transfer_row,
    write_spectrum,
)
from .config import SpectralStudyConfig


@dataclass(frozen=True)
class GeometrySideResult:
    inclusion_side: float
    spectra: dict[int, GridSpectrumRun]
    control_transfers: tuple[ControlTransferRun, ...]
    fine_runs: tuple[FineTransferRun, ...]


@dataclass(frozen=True)
class GeometryResolutionStudyResult:
    config: SpectralStudyConfig
    sides: tuple[GeometrySideResult, ...]
    spectral_rows: tuple[dict, ...]
    transfer_rows: tuple[dict, ...]
    fine_rows: tuple[dict, ...]
    summary_rows: tuple[dict, ...]
    by_resolution_rows: tuple[dict, ...]
    by_side_rows: tuple[dict, ...]


def run_geometry_resolution_study(
    config: SpectralStudyConfig,
    *,
    store: ArtifactStore | None = None,
    include_fine: bool = True,
    finalize: bool = True,
) -> GeometryResolutionStudyResult:
    owns_store = store is None
    if store is None:
        store = ArtifactStore(config.output_root)
        store.write_json("config.json", config)

    dense_backend = make_backend(config)
    work_backend = make_backend(config)
    levels = hierarchy_levels(config)
    side_results: list[GeometrySideResult] = []
    spectral_rows: list[dict] = []
    transfer_rows: list[dict] = []
    fine_rows: list[dict] = []
    summary_rows: list[dict] = []
    prepared_fine: PreparedEMKernel | None = None

    for side in config.geometry.inclusion_sides:
        definition = local_inclusion_case(float(side), k0=4.0)
        spectra: dict[int, GridSpectrumRun] = {}
        for level in levels:
            built = build_spectral_case(
                definition,
                grid_shape=level,
                backend=dense_backend,
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            run = compute_grid_spectrum(
                built,
                eigenvalue_repeats=config.eigenvalue_repeats,
            )
            spectra[level] = run
            spectral_rows.append(
                localization_row(
                    run,
                    extra={"experiment": "E4", "inclusion_side": float(side)},
                )
            )
            write_spectrum(
                store,
                f"raw/E4_spectra/a_{float(side):g}/spectrum_N{level}.npz",
                run,
            )

        control = spectra[config.grids.control_size]
        transfers = tuple(
            assess_control_transfer(spectra[level], control) for level in levels
        )
        transfer_rows.extend(
            transfer_row(
                item,
                extra={"experiment": "E4", "inclusion_side": float(side)},
            )
            for item in transfers
        )

        fine_runs: tuple[FineTransferRun, ...] = ()
        fine_geometry = None
        if include_fine:
            fine_case = build_spectral_case(
                definition,
                grid_shape=config.grids.fine_size,
                backend=work_backend,
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            fine_geometry = fine_case.geometry
            if prepared_fine is None:
                prepared_fine = PreparedEMKernel.build(
                    fine_case.problem.grid, k=fine_case.problem.k0
                )
            parameters: dict[str, CircleLocalization] = {
                f"N{level}": spectra[level].localization.circle
                for level in levels
                if spectra[level].localization.circle is not None
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
                row = fine_row(
                    run,
                    extra={
                        "experiment": "E4",
                        "inclusion_side": float(side),
                        "N_H": level,
                        "fine_cells_min_axis": min(fine_geometry.cells_per_axis),
                        "fine_cells_total": fine_geometry.cells_total,
                        "fine_volume_error": fine_geometry.relative_volume_error,
                    },
                )
                fine_rows.append(row)
                store.write_json(
                    f"raw/E4_residual_histories/a_{float(side):g}_N{level}.json",
                    run.solver_result.residual_history,
                )

        transfer_by_level = {
            item.coarse.grid_shape[0]: item for item in transfers
        }
        fine_by_level = {
            int(item.parameter_label.removeprefix("N")): item
            for item in fine_runs
        }
        for level in levels:
            spectrum = spectra[level]
            transfer = transfer_by_level[level]
            assessment = transfer.assessment
            fine = fine_by_level.get(level)
            row = {
                "experiment": "E4",
                "inclusion_side": float(side),
                "N_H": level,
                "N_c": config.grids.control_size,
                "N_f": config.grids.fine_size,
                "coarse_cells_min_axis": min(spectrum.geometry.cells_per_axis),
                "coarse_cells_total": spectrum.geometry.cells_total,
                "coarse_volume_error": spectrum.geometry.relative_volume_error,
                "control_cells_min_axis": min(control.geometry.cells_per_axis),
                "control_cells_total": control.geometry.cells_total,
                "parameter_exists": spectrum.localization.circle is not None,
                "q_H": (
                    spectrum.localization.circle.q
                    if spectrum.localization.circle is not None
                    else float("nan")
                ),
                "Delta_H": (
                    spectrum.localization.circle.margin
                    if spectrum.localization.circle is not None
                    else float("nan")
                ),
                "epsilon_directed": (
                    assessment.directed_error if assessment is not None else float("nan")
                ),
                "eta": (
                    assessment.normalized_error if assessment is not None else float("nan")
                ),
                "q_c_mu_H": (
                    assessment.target_factor if assessment is not None else float("nan")
                ),
                "q_bound_c": (
                    assessment.target_factor_bound if assessment is not None else float("nan")
                ),
                "certified_control": bool(
                    assessment.certified if assessment is not None else False
                ),
                "fine_converged": bool(
                    fine.solver_result.converged if fine is not None else False
                ),
                "fine_status": (
                    fine.solver_result.status if fine is not None else "not_run"
                ),
                "fine_classification": (
                    fine.convergence.classification.value
                    if fine is not None
                    else "not_run"
                ),
                "fine_iterations": (
                    fine.solver_result.iterations if fine is not None else 0
                ),
                "fine_matvec_count": (
                    fine.solver_result.matvec_count if fine is not None else 0
                ),
                "fine_final_residual": (
                    fine.convergence.final_residual if fine is not None else float("nan")
                ),
                "observed_residual_ratio": (
                    fine.convergence.asymptotic_ratio if fine is not None else float("nan")
                ),
                "fine_cells_min_axis": (
                    min(fine_geometry.cells_per_axis)
                    if fine_geometry is not None
                    else 0
                ),
                "fine_cells_total": (
                    fine_geometry.cells_total if fine_geometry is not None else 0
                ),
                "fine_volume_error": (
                    fine_geometry.relative_volume_error
                    if fine_geometry is not None
                    else float("nan")
                ),
            }
            row["control_safe_but_fine_failed"] = bool(
                row["certified_control"] and include_fine and not row["fine_converged"]
            )
            row["observed_stability_margin"] = (
                1.0 - row["observed_residual_ratio"]
                if pd.notna(row["observed_residual_ratio"])
                else float("nan")
            )
            summary_rows.append(row)

        side_results.append(
            GeometrySideResult(
                inclusion_side=float(side),
                spectra=spectra,
                control_transfers=transfers,
                fine_runs=fine_runs,
            )
        )

    summary_frame = pd.DataFrame(summary_rows)
    if len(summary_frame):
        by_resolution = (
            summary_frame.groupby("coarse_cells_min_axis", dropna=False)
            .agg(
                experiments=("N_H", "size"),
                converged=("fine_converged", "sum"),
                mean_ratio=("observed_residual_ratio", "mean"),
                median_ratio=("observed_residual_ratio", "median"),
            )
            .reset_index()
        )
        by_resolution["diverged_or_unresolved"] = (
            by_resolution["experiments"] - by_resolution["converged"]
        )
        by_side = (
            summary_frame.groupby("inclusion_side")
            .agg(
                experiments=("N_H", "size"),
                converged=("fine_converged", "sum"),
                control_safe_but_fine_failed=(
                    "control_safe_but_fine_failed",
                    "sum",
                ),
                median_ratio=("observed_residual_ratio", "median"),
            )
            .reset_index()
        )
    else:
        by_resolution = pd.DataFrame()
        by_side = pd.DataFrame()

    by_resolution_rows = tuple(by_resolution.to_dict(orient="records"))
    by_side_rows = tuple(by_side.to_dict(orient="records"))
    store.write_rows("tables/E4_spectral.csv", spectral_rows)
    store.write_rows("tables/E4_control_transfer.csv", transfer_rows)
    if fine_rows:
        store.write_rows("tables/E4_fine.csv", fine_rows)
    store.write_rows("tables/E4_summary.csv", summary_rows)
    store.write_rows("tables/E4_by_resolution.csv", by_resolution_rows)
    store.write_rows("tables/E4_by_side.csv", by_side_rows)
    if finalize:
        store.finalize(config=config)

    return GeometryResolutionStudyResult(
        config=config,
        sides=tuple(side_results),
        spectral_rows=tuple(spectral_rows),
        transfer_rows=tuple(transfer_rows),
        fine_rows=tuple(fine_rows),
        summary_rows=tuple(summary_rows),
        by_resolution_rows=by_resolution_rows,
        by_side_rows=by_side_rows,
    )
