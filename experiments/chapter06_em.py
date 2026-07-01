"""Compatibility facade for chapter 6 electrodynamic experiment helpers."""
from __future__ import annotations

from .cases import (
    ExperimentCase,
    LayerSpec,
    SolverRun,
    make_anisotropic_ellipsoid_case,
    make_layered_box_case,
    make_sphere_case,
    make_uniaxial_crystal_ellipsoid_case,
)
from .experiment_logging import ExperimentLogger
from .materials import MaterialSpec, material_eps, rotate_tensor
from .plots import (
    plot_fft_vs_dense_timing,
    plot_rcs_scan,
    plot_residual_histories,
    plot_three_field_slices,
)
from .scans import (
    FFT_DENSE_N_VALUES,
    N_SERIES_FULL,
    N_SERIES_QUICK,
    RCS_DEFAULT_RADIUS,
    RCS_K0A_VALUES,
    benchmark_fft_vs_dense,
    benchmark_matvec,
    build_problem,
    compute_mie_rcs_diagnostics,
    ensure_output_dirs,
    estimate_gamma0,
    make_anisotropic_gamma0_case,
    make_isotropic_gamma0_case,
    make_layered_gamma0_case,
    n_series_for_mode,
    run_quick_experiment,
    run_solver_comparison,
    run_solver,
    run_solver_suite,
    save_json,
    save_runs_csv,
    scan_gamma0,
    scan_mie_rcs_by_k0a,
    scan_sim_convergence_by_gamma0,
)
from .structured_lattice import (
    InclusionSpec,
    StructuredLatticeCase,
    build_structured_lattice_problem,
    make_structured_lattice_case,
    run_structured_lattice_experiment,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for quick/full chapter 6 experiments."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run chapter 6 EM experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output-root", default=str(Path("experiments") / "outputs" / "chapter06"))
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    run_quick_experiment(
        output_root=args.output_root,
        n_values=n_series_for_mode(args.mode),
        max_iter=args.max_iter,
        rtol=args.rtol,
        mode=args.mode,
    )
    return 0


__all__ = [
    "ExperimentCase",
    "ExperimentLogger",
    "FFT_DENSE_N_VALUES",
    "InclusionSpec",
    "LayerSpec",
    "MaterialSpec",
    "N_SERIES_FULL",
    "N_SERIES_QUICK",
    "RCS_DEFAULT_RADIUS",
    "RCS_K0A_VALUES",
    "SolverRun",
    "StructuredLatticeCase",
    "benchmark_fft_vs_dense",
    "benchmark_matvec",
    "build_problem",
    "build_structured_lattice_problem",
    "compute_mie_rcs_diagnostics",
    "ensure_output_dirs",
    "estimate_gamma0",
    "main",
    "make_anisotropic_ellipsoid_case",
    "make_anisotropic_gamma0_case",
    "make_isotropic_gamma0_case",
    "make_layered_box_case",
    "make_layered_gamma0_case",
    "make_sphere_case",
    "make_structured_lattice_case",
    "make_uniaxial_crystal_ellipsoid_case",
    "material_eps",
    "n_series_for_mode",
    "plot_fft_vs_dense_timing",
    "plot_rcs_scan",
    "plot_residual_histories",
    "plot_three_field_slices",
    "rotate_tensor",
    "run_quick_experiment",
    "run_solver",
    "run_solver_comparison",
    "run_solver_suite",
    "run_structured_lattice_experiment",
    "save_json",
    "save_runs_csv",
    "scan_gamma0",
    "scan_mie_rcs_by_k0a",
    "scan_sim_convergence_by_gamma0",
]


if __name__ == "__main__":
    raise SystemExit(main())
