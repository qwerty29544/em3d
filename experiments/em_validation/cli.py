from __future__ import annotations

import argparse
from pathlib import Path

from .config import ValidationStudyConfig
from .workflow import run_validation_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible Chapter 4 solver, field, far-field, and Mie "
            "validation experiments."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "publication"),
        default="quick",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
    )
    parser.add_argument(
        "--study",
        choices=("all", "solver", "mie"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip PNG/PDF/SVG rendering",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "quick":
        config = ValidationStudyConfig.quick(
            output_root=args.output_root or "experiments/outputs/em_validation_quick",
            device=args.device or "cpu",
        )
    else:
        config = ValidationStudyConfig.publication(
            output_root=args.output_root
            or "experiments/outputs/em_validation_publication",
            device=args.device or "auto",
        )

    result = run_validation_suite(
        config,
        include_solver_study=args.study in {"all", "solver"},
        include_mie_study=args.study in {"all", "mie"},
        render_figures=not args.no_figures,
    )
    qualified_stationary = sum(
        execution.qualified
        for case in result.stationary_cases
        for execution in case.executions
    )
    qualified_mie = sum(
        execution.qualified
        for case in result.mie_cases
        for execution in case.executions
    )
    print(f"output_root={result.output_root}")
    print(
        f"stationary_cases={len(result.stationary_cases)}, "
        f"qualified_solver_runs={qualified_stationary}"
    )
    print(
        f"mie_cases={len(result.mie_cases)}, "
        f"qualified_mie_runs={qualified_mie}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
