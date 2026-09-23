from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from em3d.experiments.spectral_transfer import (
    FineTransferRun,
    GridSpectrumRun,
    ParameterPhase,
    SamplingMetrics,
    assess_control_transfer,
    assess_parameter_phase,
    build_spectral_case,
    compute_grid_spectrum,
    diagnose_iteration_spectrum_multistart,
    local_inclusion_case,
    material_index_scale,
    run_parameter_transfer,
    sampling_metrics,
)
from em3d.geometry import SamplingMode
from em3d.operator import PreparedEMKernel
from em3d.spectral import ArnoldiMultiStartResult, recover_operator_eigenvalue

from .artifacts import ArtifactStore
from .common import (
    hierarchy_levels,
    localization_row,
    make_backend,
    release_backend_memory,
    sampling_row,
    study_solver_config,
    transfer_row,
    write_spectrum,
)
from .config import SpectralStudyConfig


@dataclass(frozen=True)
class WaveNumberResult:
    wave_number: float
    spectra: dict[int, GridSpectrumRun]
    fine_runs: tuple[FineTransferRun, ...]
    arnoldi_runs: dict[int, ArnoldiMultiStartResult]
    spectral_errors: dict[int, str]


@dataclass(frozen=True)
class WaveNumberPhaseStudyResult:
    config: SpectralStudyConfig
    wave_numbers: tuple[WaveNumberResult, ...]
    spectral_rows: tuple[dict, ...]
    fine_rows: tuple[dict, ...]
    phase_rows: tuple[dict, ...]
    by_wave_number_rows: tuple[dict, ...]
    rule_diagnostic_rows: tuple[dict, ...]
    arnoldi_rows: tuple[dict, ...]
    boundary_rows: tuple[dict, ...]


_DETAILED_PHASE_CODE = {
    ParameterPhase.NO_PARAMETER: 0,
    ParameterPhase.UNSTABLE: 1,
    ParameterPhase.CONVERGED: 2,
    ParameterPhase.CONTRACTING_UNRESOLVED: 3,
    ParameterPhase.UNRESOLVED: 4,
    ParameterPhase.NONFINITE: 5,
}


def _is_contracting(phase: ParameterPhase) -> bool:
    return phase in {
        ParameterPhase.CONVERGED,
        ParameterPhase.CONTRACTING_UNRESOLVED,
    }


def _legacy_article_code(
    phase: ParameterPhase,
    *,
    arnoldi_radius: float = np.nan,
) -> tuple[int, str]:
    """Return the historical three-state article classification explicitly.

    This projection is kept separate from the detailed phase.  In particular,
    ``contracting_unresolved`` is not relabelled as ``converged`` in the raw
    results even though the historical phase diagram grouped both as stable.
    """

    if phase is ParameterPhase.NO_PARAMETER:
        return 0, "no_parameter"
    if np.isfinite(arnoldi_radius):
        return (
            (2, "arnoldi_radius_below_one")
            if arnoldi_radius < 1.0
            else (1, "arnoldi_radius_not_below_one")
        )
    if _is_contracting(phase):
        return 2, "solver_converged_or_contracting"
    return 1, "solver_unstable_or_unresolved"


def _strict_phase_code(phase: ParameterPhase) -> int:
    if phase is ParameterPhase.NO_PARAMETER:
        return 0
    if phase is ParameterPhase.CONVERGED:
        return 2
    if phase in {ParameterPhase.UNSTABLE, ParameterPhase.NONFINITE}:
        return 1
    return 3


def _rule_diagnostic(rows: pd.DataFrame, rule: str) -> dict:
    prediction = rows[rule].astype(bool).to_numpy()
    truth = rows["empirically_contracting"].astype(bool).to_numpy()
    tp = int(np.sum(prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    return {
        "experiment": "E3",
        "rule": rule,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": float((tp + tn) / len(rows)) if len(rows) else np.nan,
        "experiments": int(len(rows)),
    }


def _boundary_levels(config: SpectralStudyConfig, k0: float) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(level)
                for wave_number, level in config.wave_number.boundary_points
                if np.isclose(float(wave_number), float(k0), rtol=0.0, atol=1e-12)
            }
        )
    )


def run_wave_number_phase_study(
    config: SpectralStudyConfig,
    *,
    store: ArtifactStore | None = None,
    include_fine: bool = True,
    include_arnoldi: bool = True,
    finalize: bool = True,
) -> WaveNumberPhaseStudyResult:
    """Run E3/E3b without collapsing unresolved modes in the raw data."""

    if store is None:
        store = ArtifactStore(config.output_root)
        store.write_json("config.json", config)

    dense_backend = make_backend(config)
    work_backend = make_backend(config)
    standard_levels = hierarchy_levels(config)
    boundary_points = {
        (float(wave_number), int(level))
        for wave_number, level in config.wave_number.boundary_points
    }

    wave_results: list[WaveNumberResult] = []
    spectral_rows: list[dict] = []
    fine_rows: list[dict] = []
    phase_rows: list[dict] = []
    arnoldi_rows: list[dict] = []
    boundary_rows: list[dict] = []

    for k0_value in config.wave_number.wave_numbers:
        k0 = float(k0_value)
        definition = local_inclusion_case(
            config.wave_number.inclusion_side,
            k0=k0,
            key_prefix="wave_scan",
            title_prefix="Волновая спектральная серия",
        )
        levels = tuple(
            sorted(set(standard_levels + _boundary_levels(config, k0)))
        )
        spectra: dict[int, GridSpectrumRun] = {}
        spectral_errors: dict[int, str] = {}
        sampling_by_level: dict[int, SamplingMetrics] = {}

        for level in levels:
            metrics = sampling_metrics(
                definition,
                grid_shape=level,
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            sampling_by_level[level] = metrics
            try:
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
            except Exception as exc:  # retained as a scientific resource-limit result
                spectral_errors[level] = f"{type(exc).__name__}: {exc}"
                spectral_rows.append(
                    {
                        "experiment": "E3b" if (k0, level) in boundary_points else "E3",
                        "k0": k0,
                        "N_H": level,
                        "computation_status": "spectral_error",
                        "error_message": spectral_errors[level],
                        **sampling_row(metrics, prefix="coarse_"),
                    }
                )
                continue

            spectra[level] = run
            experiment = "E3b" if (k0, level) in boundary_points else "E3"
            spectral_rows.append(
                localization_row(
                    run,
                    extra={
                        "experiment": experiment,
                        "k0": k0,
                        "N_H": level,
                        "computation_status": (
                            "parameter_exists"
                            if run.localization.circle is not None
                            else "no_parameter"
                        ),
                        "error_message": "",
                        **sampling_row(metrics, prefix="coarse_"),
                    },
                )
            )
            write_spectrum(
                store,
                f"raw/{experiment}_spectra/k0_{k0:g}/spectrum_N{level}.npz",
                run,
            )

        control = spectra.get(config.grids.control_size)
        control_assessment_by_level = {}
        if control is not None:
            for level, spectrum in spectra.items():
                if level <= config.grids.control_size:
                    control_assessment_by_level[level] = assess_control_transfer(
                        spectrum, control
                    )

        fine_runs: tuple[FineTransferRun, ...] = ()
        arnoldi_runs: dict[int, ArnoldiMultiStartResult] = {}
        fine_case = None
        prepared = None
        fine_sampling = sampling_metrics(
            definition,
            grid_shape=config.grids.fine_size,
            sampling_mode=SamplingMode.CELL_CENTER,
        )
        if include_fine:
            fine_case = build_spectral_case(
                definition,
                grid_shape=config.grids.fine_size,
                backend=work_backend,
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            prepared = PreparedEMKernel.build(
                fine_case.problem.grid,
                k=fine_case.problem.k0,
            )
            parameters = {
                f"N{level}": spectrum.localization.circle
                for level, spectrum in spectra.items()
                if spectrum.localization.circle is not None
            }
            if parameters:
                fine_runs = run_parameter_transfer(
                    fine_case,
                    parameters,
                    solver_config=study_solver_config(config),
                    prepared_kernel=prepared,
                    retain_solution=False,
                )
            for run in fine_runs:
                level = int(run.parameter_label.removeprefix("N"))
                fine_rows.append(
                    {
                        "experiment": "E3b" if (k0, level) in boundary_points else "E3",
                        "k0": k0,
                        "N_H": level,
                        "N_f": config.grids.fine_size,
                        "parameter": run.parameter_label,
                        "mu_real": run.circle.mu.real,
                        "mu_imag": run.circle.mu.imag,
                        "q_H": run.circle.q,
                        "reached_rtol": run.solver_result.converged,
                        "run_status": run.solver_result.status,
                        "iterations": run.solver_result.iterations,
                        "matvec_count": run.solver_result.matvec_count,
                        "minimum_residual": run.convergence.minimum_residual,
                        "final_residual": run.convergence.final_residual,
                        "observed_residual_ratio": run.convergence.asymptotic_ratio,
                        "classification": run.convergence.classification.value,
                        "elapsed_seconds": run.elapsed_seconds,
                        **sampling_row(fine_sampling, prefix="fine_"),
                    }
                )
                store.write_json(
                    f"raw/E3_residual_histories/k0_{k0:g}_N{level}.json",
                    run.solver_result.residual_history,
                )

            if include_arnoldi and fine_case is not None and prepared is not None:
                target_levels = (
                    _boundary_levels(config, k0)
                    if config.wave_number.arnoldi_boundary_only
                    else tuple(sorted(spectra))
                )
                seeds = tuple(
                    int(config.arnoldi.seed + offset) for offset in range(3)
                )
                for level in target_levels:
                    spectrum = spectra.get(level)
                    if spectrum is None or spectrum.localization.circle is None:
                        continue
                    result = diagnose_iteration_spectrum_multistart(
                        fine_case,
                        spectrum.localization.circle,
                        config=config.arnoldi,
                        seeds=seeds,
                        prepared_kernel=prepared,
                    )
                    arnoldi_runs[level] = result
                    dominant = result.dominant_run
                    recovered = recover_operator_eigenvalue(
                        dominant.dominant_value,
                        spectrum.localization.circle.mu,
                    )
                    for index, run in enumerate(result.runs):
                        arnoldi_rows.append(
                            {
                                "experiment": "E3b" if (k0, level) in boundary_points else "E3",
                                "k0": k0,
                                "N_H": level,
                                "start_index": index,
                                "seed": run.seed,
                                "spectral_radius": run.spectral_radius,
                                "dominant_real": run.dominant_value.real,
                                "dominant_imag": run.dominant_value.imag,
                                "dominant_residual": run.dominant_residual,
                                "orthogonality_error": run.orthogonality_error,
                                "arnoldi_dimension": run.actual_dimension,
                                "elapsed_seconds": run.elapsed_seconds,
                                "selected_dominant_run": run is dominant,
                                "recovered_lambda_real": (
                                    recovered.real if run is dominant else np.nan
                                ),
                                "recovered_lambda_imag": (
                                    recovered.imag if run is dominant else np.nan
                                ),
                            }
                        )
                    if dominant.hessenberg is not None:
                        store.write_npz(
                            f"raw/E3_arnoldi/k0_{k0:g}_N{level}.npz",
                            hessenberg=dominant.hessenberg,
                            seeds=np.asarray(seeds, dtype=np.int64),
                            radii=np.asarray(
                                [run.spectral_radius for run in result.runs],
                                dtype=np.float64,
                            ),
                        )

        fine_by_level = {
            int(run.parameter_label.removeprefix("N")): run for run in fine_runs
        }
        for level in levels:
            metrics = sampling_by_level[level]
            spectrum = spectra.get(level)
            fine = fine_by_level.get(level)
            if spectrum is None:
                phase = ParameterPhase.UNRESOLVED
                circle_exists = False
                reached_rtol = False
                ratio = np.nan
            else:
                assessment = assess_parameter_phase(spectrum.localization, fine)
                phase = assessment.phase
                circle_exists = assessment.circle_exists
                reached_rtol = assessment.reached_rtol
                ratio = assessment.asymptotic_ratio

            control_run = control_assessment_by_level.get(level)
            control_metrics = transfer_row(control_run) if control_run is not None else {}
            arnoldi = arnoldi_runs.get(level)
            arnoldi_radius = (
                arnoldi.dominant_run.spectral_radius if arnoldi is not None else np.nan
            )
            arnoldi_residual = (
                arnoldi.dominant_run.dominant_residual if arnoldi is not None else np.nan
            )
            legacy_code, legacy_basis = _legacy_article_code(
                phase, arnoldi_radius=arnoldi_radius
            )
            row = {
                "experiment": "E3b" if (k0, level) in boundary_points else "E3",
                "k0": k0,
                "N_H": level,
                "N_c": config.grids.control_size,
                "N_f": config.grids.fine_size,
                "spectral_computation_ok": spectrum is not None,
                "spectral_error": spectral_errors.get(level, ""),
                "origin_in_hull": (
                    spectrum.localization.origin_in_hull
                    if spectrum is not None
                    else np.nan
                ),
                "origin_distance": (
                    spectrum.localization.origin_distance
                    if spectrum is not None
                    else np.nan
                ),
                "parameter_exists": circle_exists,
                "parameter_status": (
                    spectrum.localization.status.value
                    if spectrum is not None
                    else "spectral_error"
                ),
                "phase_detailed": phase.value,
                "phase_code_detailed": _DETAILED_PHASE_CODE[phase],
                "phase_code_strict": _strict_phase_code(phase),
                "article_phase_code": legacy_code,
                "article_phase_basis": legacy_basis,
                "reached_rtol": reached_rtol,
                "empirically_contracting": _is_contracting(phase),
                "observed_residual_ratio": ratio,
                "run_status": (
                    fine.solver_result.status if fine is not None else "not_run"
                ),
                "matvec_count": (
                    fine.solver_result.matvec_count if fine is not None else 0
                ),
                "minimum_residual": (
                    fine.convergence.minimum_residual if fine is not None else np.nan
                ),
                "final_residual": (
                    fine.convergence.final_residual if fine is not None else np.nan
                ),
                "arnoldi_spectral_radius": arnoldi_radius,
                "arnoldi_dominant_residual": arnoldi_residual,
                "arnoldi_starts": len(arnoldi.runs) if arnoldi is not None else 0,
                **sampling_row(metrics, prefix="coarse_"),
                **sampling_row(fine_sampling, prefix="fine_"),
            }
            circle = (
                spectrum.localization.circle if spectrum is not None else None
            )
            if control_metrics:
                row.update(
                    {
                        "q_H": control_metrics["q_H"],
                        "Delta_H": control_metrics["Delta_H"],
                        "epsilon_c_H": control_metrics["epsilon_directed"],
                        "eta_c_H": control_metrics["eta"],
                        "q_c_mu_H": control_metrics["q_c_mu_H"],
                        "q_bound_c": control_metrics["q_bound_c"],
                        "certified_control": control_metrics["certified"],
                        "control_comparison": "coarse_to_control",
                    }
                )
            else:
                row.update(
                    {
                        "q_H": circle.q if circle is not None else np.nan,
                        "Delta_H": circle.margin if circle is not None else np.nan,
                        "epsilon_c_H": np.nan,
                        "eta_c_H": np.nan,
                        "q_c_mu_H": np.nan,
                        "q_bound_c": np.nan,
                        "certified_control": False,
                        "control_comparison": (
                            "not_applicable_finer_than_control"
                            if level > config.grids.control_size
                            else "unavailable"
                        ),
                    }
                )
            row["combined_free_rule"] = bool(
                metrics.geometry_two_cells_ok and metrics.nyquist_free_ok
            )
            row["combined_material_rule"] = bool(
                metrics.geometry_two_cells_ok and metrics.nyquist_material_ok
            )
            phase_rows.append(row)
            if (k0, level) in boundary_points:
                boundary_rows.append(dict(row))

        wave_results.append(
            WaveNumberResult(
                wave_number=k0,
                spectra=spectra,
                fine_runs=fine_runs,
                arnoldi_runs=arnoldi_runs,
                spectral_errors=spectral_errors,
            )
        )
        del prepared, fine_case
        release_backend_memory(work_backend)

    phase_frame = pd.DataFrame(phase_rows)
    by_wave_number_rows: tuple[dict, ...]
    rule_rows: tuple[dict, ...]
    if len(phase_frame):
        summaries: list[dict] = []
        for k0, part in phase_frame.groupby("k0", sort=True):
            part = part.sort_values("N_H")
            with_parameter = part[part["parameter_exists"]]
            contracting = part[part["empirically_contracting"]]
            converged = part[part["reached_rtol"]]
            definition = local_inclusion_case(
                config.wave_number.inclusion_side,
                k0=float(k0),
                key_prefix="wave_scan",
            )
            scale = material_index_scale(definition)

            def first_value(frame: pd.DataFrame, name: str):
                return frame.iloc[0][name] if len(frame) else np.nan

            summaries.append(
                {
                    "experiment": "E3",
                    "k0": float(k0),
                    "min_N_parameter_exists": first_value(with_parameter, "N_H"),
                    "min_N_contracting": first_value(contracting, "N_H"),
                    "min_N_reached_rtol": first_value(converged, "N_H"),
                    "ppw_free_at_parameter": first_value(
                        with_parameter, "coarse_ppw_free"
                    ),
                    "ppw_material_at_parameter": first_value(
                        with_parameter, "coarse_ppw_material"
                    ),
                    "ppw_free_at_contracting": first_value(
                        contracting, "coarse_ppw_free"
                    ),
                    "ppw_material_at_contracting": first_value(
                        contracting, "coarse_ppw_material"
                    ),
                    "N_nyquist_free": int(
                        ceil(float(k0) * definition.domain_lengths[0] / np.pi)
                    ),
                    "N_nyquist_material": int(
                        ceil(
                            float(k0)
                            * definition.domain_lengths[0]
                            * scale
                            / np.pi
                        )
                    ),
                    "no_parameter_count": int(
                        (part["phase_detailed"] == ParameterPhase.NO_PARAMETER.value).sum()
                    ),
                    "unstable_count": int(
                        part["phase_detailed"].isin(
                            [
                                ParameterPhase.UNSTABLE.value,
                                ParameterPhase.NONFINITE.value,
                            ]
                        ).sum()
                    ),
                    "contracting_unresolved_count": int(
                        (
                            part["phase_detailed"]
                            == ParameterPhase.CONTRACTING_UNRESOLVED.value
                        ).sum()
                    ),
                    "unresolved_count": int(
                        (part["phase_detailed"] == ParameterPhase.UNRESOLVED.value).sum()
                    ),
                    "converged_count": int(part["reached_rtol"].sum()),
                    "fine_ppw_free": float(part["fine_ppw_free"].iloc[0]),
                    "fine_ppw_material": float(part["fine_ppw_material"].iloc[0]),
                }
            )
        by_wave_number = pd.DataFrame(summaries)
        evaluable = phase_frame[
            phase_frame["spectral_computation_ok"]
            & (phase_frame["phase_detailed"] != ParameterPhase.UNRESOLVED.value)
        ]
        rule_diagnostics = pd.DataFrame(
            [
                _rule_diagnostic(evaluable, "combined_free_rule"),
                _rule_diagnostic(evaluable, "combined_material_rule"),
            ]
        )
    else:
        by_wave_number = pd.DataFrame()
        rule_diagnostics = pd.DataFrame()

    by_wave_number_rows = tuple(by_wave_number.to_dict(orient="records"))
    rule_rows = tuple(rule_diagnostics.to_dict(orient="records"))
    store.write_rows("tables/E3_spectral_phase.csv", spectral_rows)
    if fine_rows:
        store.write_rows("tables/E3_fine_phase.csv", fine_rows)
    store.write_rows("tables/E3_summary_phase.csv", phase_rows)
    store.write_rows("tables/E3_by_k0_phase.csv", by_wave_number_rows)
    store.write_rows("tables/E3_sampling_rule_diagnostics.csv", rule_rows)
    if arnoldi_rows:
        store.write_rows("tables/E3b_arnoldi.csv", arnoldi_rows)
    if boundary_rows:
        store.write_rows("tables/E3b_upper_boundary.csv", boundary_rows)
    if finalize:
        store.finalize(config=config)

    return WaveNumberPhaseStudyResult(
        config=config,
        wave_numbers=tuple(wave_results),
        spectral_rows=tuple(spectral_rows),
        fine_rows=tuple(fine_rows),
        phase_rows=tuple(phase_rows),
        by_wave_number_rows=by_wave_number_rows,
        rule_diagnostic_rows=rule_rows,
        arnoldi_rows=tuple(arnoldi_rows),
        boundary_rows=tuple(boundary_rows),
    )
