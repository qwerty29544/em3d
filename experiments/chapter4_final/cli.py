from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experiments.em_validation.kaggle import archive_results
from experiments.em_validation.large_workflow import run_large_grid_suite
from experiments.em_validation.planner import build_large_grid_config

from .campaign import create_core_archive, merge_chapter4_campaign, run_spectral_stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final exhaustive Chapter 4 campaign")
    parser.add_argument(
        "--stage",
        choices=("spectral", "smoke", "main64", "audit", "control96", "control128", "merge"),
        required=True,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--job-key")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--archive-core", action="store_true")
    parser.add_argument("--allow-incomplete-campaign", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    return parser



def _without_figures(config):
    """Disable all expensive rendering while preserving numerical outputs."""

    return replace(
        config,
        mie_jobs=tuple(
            replace(job, render_field_slices=False, render_rcs=False)
            for job in config.mie_jobs
        ),
        stationary_jobs=tuple(
            replace(job, render_field_slices=False, render_rcs=False)
            for job in config.stationary_jobs
        ),
        visualization=replace(
            config.visualization,
            create_subtree_archives=False,
        ),
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_root)
    if args.stage == "spectral":
        manifest = run_spectral_stage(
            output_root=output,
            device=args.device,
            render_figures=not args.no_figures,
            quick=False,
        )
        print(manifest)
        return 0
    if args.stage == "smoke":
        spectral_manifest = run_spectral_stage(
            output_root=output / "spectral",
            device=args.device,
            render_figures=not args.no_figures,
            quick=True,
        )
        config = build_large_grid_config(
            "smoke",
            output_root=output / "validation",
            device=args.device,
            precision=args.precision,
            resume=not args.no_resume,
            render_progress=True,
        )
        if args.no_figures:
            config = _without_figures(config)
        validation = run_large_grid_suite(config)
        print(spectral_manifest)
        print(validation.manifest_path)
        return 0
    if args.stage == "merge":
        if not args.input:
            raise SystemExit("--input is required for stage=merge")
        result = merge_chapter4_campaign(
            args.input,
            output_root=output,
            create_core=args.archive_core,
            require_complete_campaign=not args.allow_incomplete_campaign,
        )
        print(result.manifest_path)
        if result.core_archive_path:
            print(result.core_archive_path)
        return 0

    config = build_large_grid_config(
        args.stage,
        output_root=output,
        device=args.device,
        precision=args.precision,
        batch_index=args.batch_index,
        batch_count=args.batch_count,
        include_optional=args.include_optional,
        resume=not args.no_resume,
        render_progress=True,
    )
    if args.no_figures:
        config = _without_figures(config)
    jobs = list(config.mie_jobs) + list(config.stationary_jobs)
    if args.list_jobs:
        for job in jobs:
            print(job.key)
        return 0
    if args.job_key:
        mie_jobs = tuple(job for job in config.mie_jobs if job.key == args.job_key)
        stationary_jobs = tuple(job for job in config.stationary_jobs if job.key == args.job_key)
        if not mie_jobs and not stationary_jobs:
            raise SystemExit(f"unknown job key {args.job_key!r}")
        config = replace(config, mie_jobs=mie_jobs, stationary_jobs=stationary_jobs)
    result = run_large_grid_suite(config)
    print(result.manifest_path)
    if args.archive_core:
        path, digest = create_core_archive(output)
        print(f"core_archive={path}")
        print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
