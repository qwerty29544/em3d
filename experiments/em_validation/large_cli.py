from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .kaggle import archive_results
from .large_workflow import run_large_grid_suite
from .merge import merge_large_grid_runs
from .planner import build_large_grid_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chapter 4 all-solver large-grid validation workflow"
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "main64", "audit", "control96", "control128", "merge"),
        default="smoke",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--job-key")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--merge-input", action="append", default=[])
    parser.add_argument("--archive", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.profile == "merge":
        if not args.merge_input:
            raise SystemExit("--merge-input is required for profile=merge")
        manifest = merge_large_grid_runs(
            args.merge_input,
            output_root=args.output_root,
        )
        print(manifest)
        return 0

    config = build_large_grid_config(
        args.profile,
        output_root=args.output_root,
        device=args.device,
        precision=args.precision,
        batch_index=args.batch_index,
        batch_count=args.batch_count,
        include_optional=args.include_optional,
        resume=not args.no_resume,
        render_progress=not args.quiet,
    )
    all_jobs = [job.key for job in config.mie_jobs] + [
        job.key for job in config.stationary_jobs
    ]
    if args.list_jobs:
        for key in all_jobs:
            print(key)
        return 0
    if args.job_key:
        mie_jobs = tuple(job for job in config.mie_jobs if job.key == args.job_key)
        stationary_jobs = tuple(
            job for job in config.stationary_jobs if job.key == args.job_key
        )
        if not mie_jobs and not stationary_jobs:
            raise SystemExit(
                f"unknown job key {args.job_key!r}; use --list-jobs to inspect the plan"
            )
        config = replace(
            config,
            mie_jobs=mie_jobs,
            stationary_jobs=stationary_jobs,
            output_root=Path(args.output_root),
        )
    result = run_large_grid_suite(config)
    print(result.manifest_path)
    if args.archive:
        archive, digest = archive_results(config.output_root)
        print(f"archive={archive}")
        print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
