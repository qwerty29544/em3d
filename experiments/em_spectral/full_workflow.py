from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from em3d.experiments.spectral_transfer import (
    SpectralCaseDefinition,
    default_article_cases,
)

from .artifacts import ArtifactStore
from .config import SpectralStudyConfig
from .ensemble_transfer import (
    EnsembleTransferStudyResult,
    run_ensemble_transfer_study,
)
from .geometry_resolution import (
    GeometryResolutionStudyResult,
    run_geometry_resolution_study,
)
from .volume_averaging import (
    VolumeAveragingStudyResult,
    run_volume_averaging_study,
)
from .wave_number_phase import (
    WaveNumberPhaseStudyResult,
    run_wave_number_phase_study,
)
from .workflow import SpectralStudyResult, run_spectral_transfer_study


@dataclass(frozen=True)
class SpectralExperimentSuiteResult:
    config: SpectralStudyConfig
    core: SpectralStudyResult | None
    geometry: GeometryResolutionStudyResult | None
    volume_averaging: VolumeAveragingStudyResult | None
    ensemble: EnsembleTransferStudyResult | None
    wave_number: WaveNumberPhaseStudyResult | None
    output_root: str


def run_spectral_experiment_suite(
    config: SpectralStudyConfig,
    *,
    cases: Iterable[SpectralCaseDefinition] | None = None,
    include_core: bool = True,
    include_geometry: bool = True,
    include_volume_averaging: bool = True,
    include_ensemble: bool = True,
    include_wave_number: bool = True,
    include_fine: bool = True,
    include_core_arnoldi: bool = True,
    include_boundary_arnoldi: bool = True,
    render_figures: bool = False,
) -> SpectralExperimentSuiteResult:
    """Run the complete E0--E6 spectral experiment suite.

    One artifact store and one manifest are used for the whole run.  E6 and E5
    consume E4 results instead of recomputing the same centre-sampled spectra.
    """

    if (include_volume_averaging or include_ensemble) and not include_geometry:
        raise ValueError(
            "volume averaging and ensemble studies require include_geometry=True"
        )

    store = ArtifactStore(config.output_root)
    store.write_json("config.json", config)
    definitions = tuple(default_article_cases() if cases is None else cases)

    core = None
    geometry = None
    volume = None
    ensemble = None
    wave = None

    if include_core:
        core = run_spectral_transfer_study(
            config,
            cases=definitions,
            include_fine=include_fine,
            include_arnoldi=include_core_arnoldi,
            store=store,
            finalize=False,
            write_config=False,
        )

    if include_geometry:
        geometry = run_geometry_resolution_study(
            config,
            store=store,
            include_fine=include_fine,
            finalize=False,
        )

    if include_volume_averaging:
        assert geometry is not None
        volume = run_volume_averaging_study(
            config,
            geometry,
            store=store,
            include_fine=include_fine,
            finalize=False,
        )

    if include_ensemble:
        assert geometry is not None
        ensemble = run_ensemble_transfer_study(
            config,
            geometry,
            store=store,
            include_fine=include_fine,
            finalize=False,
        )

    if include_wave_number:
        wave = run_wave_number_phase_study(
            config,
            store=store,
            include_fine=include_fine,
            include_arnoldi=include_boundary_arnoldi,
            finalize=False,
        )

    protocol = [
        {
            "experiment": "E0",
            "purpose": "dense/FFT operator consistency",
            "executed": bool(include_core),
        },
        {
            "experiment": "E1",
            "purpose": "coarse-to-control spectral transfer",
            "executed": bool(include_core),
        },
        {
            "experiment": "E2/E2A/E2B",
            "purpose": "fine transfer and Arnoldi diagnostics",
            "executed": bool(include_core and include_fine),
        },
        {
            "experiment": "E4",
            "purpose": "geometric resolution",
            "executed": bool(include_geometry),
        },
        {
            "experiment": "E6",
            "purpose": "volume-averaged coarse material model",
            "executed": bool(include_volume_averaging),
        },
        {
            "experiment": "E5",
            "purpose": "ensemble spectral localization",
            "executed": bool(include_ensemble),
        },
        {
            "experiment": "E3/E3b",
            "purpose": "wave-number phase map and upper boundary",
            "executed": bool(include_wave_number),
        },
    ]
    store.write_rows("tables/FINAL_experimental_protocol.csv", protocol)

    if render_figures:
        import matplotlib.pyplot as plt

        from .plots import (
            plot_ensemble_ratio,
            plot_geometry_resolution_cells,
            plot_geometry_resolution_ratio,
            plot_residual_histories,
            plot_spectral_localizations,
            plot_transfer_bounds,
            plot_volume_averaging_improvement,
            plot_volume_averaging_ratios,
            plot_wave_material_ppw,
            plot_wave_phase_article,
            plot_wave_phase_detailed,
            plot_wave_required_grid,
        )

        figures = []
        if core is not None:
            for case in core.cases:
                figure, _ = plot_spectral_localizations(case)
                store.save_figure(
                    f"figures/core/{case.definition.key}_spectral_hulls",
                    figure,
                )
                figures.append(figure)
                figure, _ = plot_transfer_bounds(case)
                store.save_figure(
                    f"figures/core/{case.definition.key}_control_transfer",
                    figure,
                )
                figures.append(figure)
                if case.fine_runs:
                    figure, _ = plot_residual_histories(case)
                    store.save_figure(
                        f"figures/core/{case.definition.key}_fine_residuals",
                        figure,
                    )
                    figures.append(figure)
        if geometry is not None:
            figure, _ = plot_geometry_resolution_ratio(geometry)
            store.save_figure("figures/E4_geometry_ratio", figure)
            figures.append(figure)
            figure, _ = plot_geometry_resolution_cells(geometry)
            store.save_figure("figures/E4_geometry_cells", figure)
            figures.append(figure)
        if volume is not None:
            figure, _ = plot_volume_averaging_improvement(volume)
            store.save_figure("figures/E6_volume_averaging_improvement", figure)
            figures.append(figure)
            figure, _ = plot_volume_averaging_ratios(volume)
            store.save_figure("figures/E6_volume_averaging_ratios", figure)
            figures.append(figure)
        if ensemble is not None:
            figure, _ = plot_ensemble_ratio(ensemble)
            store.save_figure("figures/E5_ensemble_ratio", figure)
            figures.append(figure)
        if wave is not None:
            figure, _ = plot_wave_phase_detailed(wave)
            store.save_figure("figures/E3_phase_detailed", figure)
            figures.append(figure)
            figure, _ = plot_wave_phase_article(wave)
            store.save_figure("figures/E3_phase_article", figure)
            figures.append(figure)
            figure, _ = plot_wave_material_ppw(wave)
            store.save_figure("figures/E3_material_ppw", figure)
            figures.append(figure)
            figure, _ = plot_wave_required_grid(wave)
            store.save_figure("figures/E3_required_grid", figure)
            figures.append(figure)
        for figure in figures:
            plt.close(figure)

    store.finalize(config=config)

    return SpectralExperimentSuiteResult(
        config=config,
        core=core,
        geometry=geometry,
        volume_averaging=volume,
        ensemble=ensemble,
        wave_number=wave,
        output_root=str(store.root),
    )
