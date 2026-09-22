from __future__ import annotations

import argparse

from em3d.experiments.spectral_transfer import default_article_cases

from .config import SpectralStudyConfig
from .workflow import run_spectral_transfer_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible spectral-transfer study")
    parser.add_argument("--mode", choices=("quick", "publication"), default="quick")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-fine", action="store_true")
    parser.add_argument("--no-arnoldi", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    factory = SpectralStudyConfig.quick if args.mode == "quick" else SpectralStudyConfig.publication
    kwargs = {"device": args.device}
    if args.output_root is not None:
        kwargs["output_root"] = args.output_root
    config = factory(**kwargs)
    run_spectral_transfer_study(
        config,
        cases=default_article_cases(),
        include_fine=not args.no_fine,
        include_arnoldi=not args.no_arnoldi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
