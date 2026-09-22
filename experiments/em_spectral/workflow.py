from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.experiments.spectral_transfer import (
    BuiltSpectralCase,
    ControlTransferRun,
    FineTransferRun,
    GridSpectrumRun,
    SpectralCaseDefinition,
    assess_control_transfer,
    build_spectral_case,
    check_operator_consistency,
    compute_grid_spectrum,
    default_article_cases,
    diagnose_iteration_spectrum,
    run_parameter_transfer,
)
from em3d.operator import PreparedEMKernel
from em3d.solvers import SolverConfig
from em3d.spectral import ArnoldiResult, CircleLocalization

from .artifacts import ArtifactStore
from .config import SpectralStudyConfig


@dataclass(frozen=True)
class CaseStudyResult:
    definition: SpectralCaseDefinition
    operator_consistency_error: float
    spectra: dict[int, GridSpectrumRun]
    control_transfers: tuple[ControlTransferRun, ...]
    fine_runs: tuple[FineTransferRun, ...]
    arnoldi_runs: dict[str, ArnoldiResult]


@dataclass(frozen=True)
class SpectralStudyResult:
    config: SpectralStudyConfig
    cases: tuple[CaseStudyResult, ...]
    output_root: str


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


def _localization_row(run: GridSpectrumRun) -> dict:
    circle = run.localization.circle
    return {
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
        "feature_volume_error": run.geometry.relative_volume_error,
        "build_seconds": run.matrix_build_seconds,
        "eig_seconds_median": run.median_eigenvalue_seconds,
    }


def _transfer_row(run: ControlTransferRun) -> dict:
    assessment = run.assessment
    row = {
        "case": run.coarse.case_key,
        "N_H": run.coarse.grid_shape[0],
        "N_c": run.target.grid_shape[0],
        "parameter_exists": assessment is not None,
        "coarse_status": run.coarse.localization.status.value,
    }
    if assessment is None:
        row.update({
            "epsilon_directed": np.nan,
            "epsilon_hausdorff": np.nan,
            "q_H": np.nan,
            "Delta_H": np.nan,
            "eta": np.nan,
            "q_c_mu_H": np.nan,
            "q_bound_c": np.nan,
            "certified": False,
        })
    else:
        row.update({
            "epsilon_directed": assessment.directed_error,
            "epsilon_hausdorff": assessment.hausdorff_distance,
            "q_H": assessment.coarse_factor,
            "Delta_H": assessment.coarse_margin,
            "eta": assessment.normalized_error,
            "q_c_mu_H": assessment.target_factor,
            "q_bound_c": assessment.target_factor_bound,
            "certified": assessment.certified,
        })
    return row


def _fine_row(run: FineTransferRun) -> dict:
    return {
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
        "asymptotic_ratio": run.convergence.asymptotic_ratio,
        "classification": run.convergence.classification.value,
        "elapsed_seconds": run.elapsed_seconds,
    }


def run_spectral_transfer_study(
    config: SpectralStudyConfig,
    *,
    cases: Iterable[SpectralCaseDefinition] | None = None,
    include_fine: bool = True,
    include_arnoldi: bool = True,
) -> SpectralStudyResult:
    definitions = tuple(default_article_cases() if cases is None else cases)
    store = ArtifactStore(config.output_root)
    store.write_json("config.json", config)

    cpu_backend = Backend.numpy(Precision.DOUBLE)
    work_backend = make_backend(config)
    results: list[CaseStudyResult] = []
    localization_rows: list[dict] = []
    transfer_rows: list[dict] = []
    fine_rows: list[dict] = []
    arnoldi_rows: list[dict] = []
    consistency_rows: list[dict] = []

    for definition in definitions:
        spectra: dict[int, GridSpectrumRun] = {}
        levels = tuple(
            sorted(
                set(config.grids.coarse_sizes + (config.grids.control_size,))
            )
        )
        for level in levels:
            built = build_spectral_case(
                definition,
                grid_shape=level,
                backend=cpu_backend,
            )
            run = compute_grid_spectrum(
                built,
                eigenvalue_repeats=config.eigenvalue_repeats,
            )
            spectra[level] = run
            localization_rows.append(_localization_row(run))
            store.write_npz(
                f"raw/spectra/{definition.key}_N{level}.npz",
                spectrum=run.localization.spectrum,
                hull=run.localization.hull,
            )

        smallest = build_spectral_case(
            definition,
            grid_shape=config.grids.coarse_sizes[0],
            backend=cpu_backend,
        )
        consistency_error = check_operator_consistency(
            smallest,
            seed=config.runtime.random_seed,
        )
        consistency_rows.append(
            {
                "case": definition.key,
                "N": config.grids.coarse_sizes[0],
                "relative_dense_fft_error": consistency_error,
                "seed": config.runtime.random_seed,
            }
        )

        control = spectra[config.grids.control_size]
        transfers = tuple(
            assess_control_transfer(spectra[level], control)
            for level in config.grids.coarse_sizes
        )
        transfer_rows.extend(_transfer_row(item) for item in transfers)

        fine_runs: tuple[FineTransferRun, ...] = ()
        arnoldi_runs: dict[str, ArnoldiResult] = {}
        if include_fine:
            fine_case = build_spectral_case(
                definition,
                grid_shape=config.grids.fine_size,
                backend=work_backend,
            )
            prepared = PreparedEMKernel.build(
                fine_case.problem.grid,
                k=fine_case.problem.k0,
            )
            parameters: dict[str, CircleLocalization] = {}
            for level in levels:
                circle = spectra[level].localization.circle
                if circle is not None:
                    parameters[f"N{level}"] = circle
            fine_runs = run_parameter_transfer(
                fine_case,
                parameters,
                solver_config=SolverConfig(
                    max_iter=config.iteration.max_iter,
                    rtol=config.iteration.rtol,
                    divergence_guard=config.iteration.divergence_guard,
                ),
                prepared_kernel=prepared,
            )
            fine_rows.extend(_fine_row(item) for item in fine_runs)
            for run in fine_runs:
                store.write_json(
                    f"raw/residual_histories/{definition.key}_{run.parameter_label}.json",
                    run.solver_result.residual_history,
                )

            if include_arnoldi:
                for label, circle in parameters.items():
                    diagnostic = diagnose_iteration_spectrum(
                        fine_case,
                        circle,
                        config=config.arnoldi,
                        prepared_kernel=prepared,
                    )
                    arnoldi_runs[label] = diagnostic
                    arnoldi_rows.append(
                        {
                            "case": definition.key,
                            "parameter": label,
                            "spectral_radius": diagnostic.spectral_radius,
                            "dominant_real": diagnostic.dominant_value.real,
                            "dominant_imag": diagnostic.dominant_value.imag,
                            "dominant_residual": diagnostic.dominant_residual,
                            "orthogonality_error": diagnostic.orthogonality_error,
                            "arnoldi_dimension": diagnostic.actual_dimension,
                            "elapsed_seconds": diagnostic.elapsed_seconds,
                        }
                    )
                    if diagnostic.hessenberg is not None:
                        store.write_npz(
                            f"raw/arnoldi/{definition.key}_{label}.npz",
                            hessenberg=diagnostic.hessenberg,
                        )

        results.append(
            CaseStudyResult(
                definition=definition,
                operator_consistency_error=consistency_error,
                spectra=spectra,
                control_transfers=transfers,
                fine_runs=fine_runs,
                arnoldi_runs=arnoldi_runs,
            )
        )

    store.write_rows("tables/operator_consistency.csv", consistency_rows)
    store.write_rows("tables/spectral_localizations.csv", localization_rows)
    store.write_rows("tables/control_transfers.csv", transfer_rows)
    if fine_rows:
        store.write_rows("tables/fine_transfers.csv", fine_rows)
    if arnoldi_rows:
        store.write_rows("tables/arnoldi.csv", arnoldi_rows)
    store.finalize(config=config)
    return SpectralStudyResult(
        config=config,
        cases=tuple(results),
        output_root=str(config.output_root),
    )
