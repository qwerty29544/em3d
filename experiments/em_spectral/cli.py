from __future__ import annotations

import argparse

from em3d.experiments.spectral_transfer import default_article_cases

from .artifacts import ArtifactStore
from .config import SpectralStudyConfig
from .ensemble_transfer import run_ensemble_transfer_study
from .full_workflow import run_spectral_experiment_suite
from .geometry_resolution import run_geometry_resolution_study
from .volume_averaging import run_volume_averaging_study
from .wave_number_phase import run_wave_number_phase_study
from .workflow import run_spectral_transfer_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible spectral-transfer experiments E0--E6"
    )
    parser.add_argument("--mode", choices=("quick", "publication"), default="quick")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--study",
        choices=("core", "geometry", "averaging", "ensemble", "wave", "all"),
        default="core",
    )
    parser.add_argument("--no-fine", action="store_true")
    parser.add_argument("--no-arnoldi", action="store_true")
    parser.add_argument("--figures", action="store_true")
    return parser


def _config_from_args(args) -> SpectralStudyConfig:
    factory = (
        SpectralStudyConfig.quick
        if args.mode == "quick"
        else SpectralStudyConfig.publication
    )
    kwargs = {"device": args.device}
    if args.output_root is not None:
        kwargs["output_root"] = args.output_root
    return factory(**kwargs)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    include_fine = not args.no_fine
    include_arnoldi = not args.no_arnoldi

    if args.study == "core":
        run_spectral_transfer_study(
            config,
            cases=default_article_cases(),
            include_fine=include_fine,
            include_arnoldi=include_arnoldi,
        )
    elif args.study == "geometry":
        run_geometry_resolution_study(
            config,
            include_fine=include_fine,
        )
    elif args.study == "averaging":
        store = ArtifactStore(config.output_root)
        store.write_json("config.json", config)
        geometry = run_geometry_resolution_study(
            config,
            store=store,
            include_fine=include_fine,
            finalize=False,
        )
        run_volume_averaging_study(
            config,
            geometry,
            store=store,
            include_fine=include_fine,
        )
    elif args.study == "ensemble":
        store = ArtifactStore(config.output_root)
        store.write_json("config.json", config)
        geometry = run_geometry_resolution_study(
            config,
            store=store,
            include_fine=include_fine,
            finalize=False,
        )
        run_ensemble_transfer_study(
            config,
            geometry,
            store=store,
            include_fine=include_fine,
        )
    elif args.study == "wave":
        run_wave_number_phase_study(
            config,
            include_fine=include_fine,
            include_arnoldi=include_arnoldi,
        )
    else:
        run_spectral_experiment_suite(
            config,
            include_fine=include_fine,
            include_core_arnoldi=include_arnoldi,
            include_boundary_arnoldi=include_arnoldi,
            render_figures=args.figures,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
