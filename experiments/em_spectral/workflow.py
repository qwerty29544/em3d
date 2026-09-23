from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from em3d.experiments.spectral_transfer import (
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
from em3d.spectral import ArnoldiResult, CircleLocalization

from .artifacts import ArtifactStore
from .common import (
    cpu_backend,
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


def run_spectral_transfer_study(
    config: SpectralStudyConfig,
    *,
    cases: Iterable[SpectralCaseDefinition] | None = None,
    include_fine: bool = True,
    include_arnoldi: bool = True,
    store: ArtifactStore | None = None,
    finalize: bool = True,
    write_config: bool = True,
) -> SpectralStudyResult:
    definitions = tuple(default_article_cases() if cases is None else cases)
    owns_store = store is None
    if store is None:
        store = ArtifactStore(config.output_root)
    if write_config:
        store.write_json("config.json", config)

    dense_backend = make_backend(config)
    consistency_backend = cpu_backend(config)
    work_backend = make_backend(config)
    results: list[CaseStudyResult] = []
    localization_rows: list[dict] = []
    transfer_rows: list[dict] = []
    fine_rows: list[dict] = []
    arnoldi_rows: list[dict] = []
    consistency_rows: list[dict] = []
    levels = hierarchy_levels(config)

    for definition in definitions:
        spectra: dict[int, GridSpectrumRun] = {}
        for level in levels:
            built = build_spectral_case(
                definition,
                grid_shape=level,
                backend=dense_backend,
            )
            run = compute_grid_spectrum(
                built,
                eigenvalue_repeats=config.eigenvalue_repeats,
            )
            spectra[level] = run
            localization_rows.append(localization_row(run))
            write_spectrum(
                store,
                f"raw/spectra/{definition.key}_N{level}.npz",
                run,
            )

        smallest = build_spectral_case(
            definition,
            grid_shape=config.grids.coarse_sizes[0],
            backend=consistency_backend,
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
        transfer_rows.extend(transfer_row(item) for item in transfers)

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
                solver_config=study_solver_config(config),
                prepared_kernel=prepared,
                retain_solution=False,
            )
            fine_rows.extend(fine_row(item) for item in fine_runs)
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
    if finalize:
        store.finalize(config=config)
    return SpectralStudyResult(
        config=config,
        cases=tuple(results),
        output_root=str(store.root),
    )


__all__ = [
    "CaseStudyResult",
    "SpectralStudyResult",
    "make_backend",
    "run_spectral_transfer_study",
]
