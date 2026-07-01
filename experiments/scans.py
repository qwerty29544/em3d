"""Scans, runners, and persistence helpers for chapter 6 experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import em3d

from .cases import (
    ExperimentCase,
    LayerSpec,
    SolverRun,
    grid_shape,
    make_anisotropic_ellipsoid_case,
    make_layered_box_case,
    make_sphere_case,
)
from .experiment_logging import ExperimentLogger
from .materials import MaterialSpec, material_eps


N_SERIES_FULL = [8, 16, 24, 32, 40, 48, 56, 64]
N_SERIES_QUICK = [8, 16, 24]
FFT_DENSE_N_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RCS_K0A_VALUES = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0]
RCS_DEFAULT_RADIUS = 0.5


def n_series_for_mode(mode: str) -> list[int]:
    """Return the grid-size series for a named experiment mode."""
    if mode == "quick":
        return list(N_SERIES_QUICK)
    if mode == "full":
        return list(N_SERIES_FULL)
    raise ValueError(f"mode must be 'quick' or 'full', got {mode!r}")


def ensure_output_dirs(root: str | Path = Path("experiments") / "outputs" / "chapter06") -> dict[str, Path]:
    """Create and return the output directory tree used by chapter 6 experiments."""
    root = Path(root)
    paths = {
        "root": root,
        "raw": root / "raw",
        "tables": root / "tables",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _log(logger: ExperimentLogger | None, event: str, **payload: Any) -> None:
    if logger is not None:
        logger.event(event, **payload)


def _eta_matrix(eps_real, eps_imag) -> np.ndarray:
    if np.ndim(eps_real) == 0:
        return (float(eps_real) - 1.0 + 1j * float(eps_imag)) * np.eye(3, dtype=np.complex128)
    return np.asarray(eps_real, dtype=np.float64) - np.eye(3) + 1j * np.asarray(eps_imag, dtype=np.float64)


def _case_eps(case: ExperimentCase):
    if case.material is not None:
        return material_eps(case.material, k0=case.k0)
    return case.eps_real, case.eps_imag


def _layered_box_refraction(grid: em3d.Grid, layers: tuple[LayerSpec, ...], k0: float):
    be = grid.backend
    xp = be.xp
    _, _, Z = grid.coords()
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for idx, layer in enumerate(layers):
        eps_real, eps_imag = material_eps(layer.material, k0=k0)
        eta_mat = _eta_matrix(eps_real, eps_imag)
        if idx == len(layers) - 1:
            mask = (Z >= layer.z_min) & (Z <= layer.z_max)
        else:
            mask = (Z >= layer.z_min) & (Z < layer.z_max)
        for i in range(3):
            for j in range(3):
                out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def build_problem(
    case: ExperimentCase,
    *,
    precision: em3d.Precision = em3d.Precision.DOUBLE,
) -> tuple[em3d.Problem, em3d.Operator]:
    """Build a CPU Problem and Operator for one experiment case."""
    be = em3d.Backend.numpy(precision)
    grid = em3d.Grid(N=case.N, L=case.L, center=case.center, backend=be)
    if case.geometry in {"sphere", "ellipsoid"}:
        eps_real, eps_imag = _case_eps(case)
        eps_tensor = em3d.ellipsis_refraction(
            grid,
            eps_real=eps_real,
            eps_imag=eps_imag,
            center=case.center,
            radius=case.radius,
        )
    elif case.geometry == "layered_box":
        eps_tensor = _layered_box_refraction(grid, case.layers, case.k0)
    else:
        raise ValueError(f"geometry must be 'sphere', 'ellipsoid', or 'layered_box', got {case.geometry!r}")
    wave = em3d.flat_wave_vec(
        grid,
        k=case.k0,
        orient=case.wave_orient,
        amplitude=case.wave_amplitude,
    )
    problem = em3d.Problem(
        grid=grid,
        eps_tensor=eps_tensor,
        wave=wave,
        k0=case.k0,
        volume=grid.dv * int(np.prod(case.N)),
    )
    return problem, em3d.Operator(problem)


def estimate_gamma0(
    problem: em3d.Problem,
    *,
    coarse_N=(4, 4, 4),
    max_coarse_N: int = 7,
) -> em3d.gamma0.Gamma0Analysis:
    """Estimate gamma0 from a coarse dense spectrum, retrying degenerate grids."""
    shape = grid_shape(coarse_N)
    last_error: ValueError | None = None
    while max(shape) <= max_coarse_N:
        try:
            return em3d.gamma0.estimate_from_problem(problem, coarse_N=shape)
        except ValueError as exc:
            last_error = exc
            shape = tuple(n + 1 for n in shape)
    if last_error is not None:
        raise last_error
    raise ValueError(f"coarse_N={coarse_N!r} exceeds max_coarse_N={max_coarse_N}")


def _solver_instance(
    solver_name: str,
    *,
    gamma0_analysis: em3d.gamma0.Gamma0Analysis | None,
    max_iter: int,
    rtol: float,
):
    if solver_name == "SIM":
        if gamma0_analysis is None:
            raise ValueError("SIM requires gamma0_analysis")
        cfg = em3d.SolverConfig(
            max_iter=max_iter,
            rtol=rtol,
            **gamma0_analysis.as_solver_config_kwargs(),
        )
        return em3d.SIM(cfg)
    if solver_name == "BiCGStab":
        return em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol))
    if solver_name == "TwoStep":
        return em3d.TwoStep(em3d.SolverConfig(max_iter=max_iter, rtol=rtol))
    raise ValueError(f"solver_name must be SIM, BiCGStab, or TwoStep, got {solver_name!r}")


def run_solver(
    problem: em3d.Problem,
    operator: em3d.Operator,
    case: ExperimentCase,
    solver_name: str,
    *,
    gamma0_analysis: em3d.gamma0.Gamma0Analysis | None = None,
    max_iter: int = 500,
    rtol: float = 1e-8,
) -> SolverRun:
    """Run one solver and return a serializable summary."""
    solver = _solver_instance(
        solver_name,
        gamma0_analysis=gamma0_analysis,
        max_iter=max_iter,
        rtol=rtol,
    )
    start = time.perf_counter()
    result = solver.solve(operator, problem.wave)
    elapsed = time.perf_counter() - start
    history = [float(x) for x in result.residual_history]
    final_residual = history[-1] if history else float("inf")
    return SolverRun(
        case_name=case.name,
        solver_name=solver_name,
        N=case.N,
        dof=case.dof,
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_residual=float(final_residual),
        elapsed_sec=float(elapsed),
        residual_history=history,
    )


def run_solver_suite(
    problem: em3d.Problem,
    operator: em3d.Operator,
    case: ExperimentCase,
    solver_names: list[str],
    *,
    gamma0_analysis: em3d.gamma0.Gamma0Analysis | None = None,
    max_iter: int = 500,
    rtol: float = 1e-8,
) -> list[SolverRun]:
    """Run several solvers for the same problem."""
    return [
        run_solver(
            problem,
            operator,
            case,
            solver_name,
            gamma0_analysis=gamma0_analysis,
            max_iter=max_iter,
            rtol=rtol,
        )
        for solver_name in solver_names
    ]


def save_runs_csv(runs: list[SolverRun], path: str | Path) -> None:
    """Save solver summaries as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "solver_name",
        "N",
        "dof",
        "converged",
        "iterations",
        "final_residual",
        "elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(run.to_row())


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def benchmark_matvec(
    problem: em3d.Problem,
    operator: em3d.Operator,
    case: ExperimentCase,
    *,
    repeats: int = 5,
    operator_build_sec: float = 0.0,
) -> dict[str, Any]:
    """Measure average FFT matvec time for the problem wave."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        operator.matvec(problem.wave)
        durations.append(time.perf_counter() - start)
    return {
        "case_name": case.name,
        "N": "x".join(str(n) for n in case.N),
        "dof": case.dof,
        "operator_build_sec": float(operator_build_sec),
        "matvec_avg_sec": float(np.mean(durations)),
        "matvec_min_sec": float(np.min(durations)),
        "matvec_max_sec": float(np.max(durations)),
        "repeats": int(repeats),
    }


def _flatten_field(u) -> np.ndarray:
    arr = np.asarray(u)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"u must have shape (3,Nx,Ny,Nz), got {arr.shape}")
    return np.transpose(arr, (1, 2, 3, 0)).reshape(-1)


def _unflatten_field(v, N: tuple[int, int, int]) -> np.ndarray:
    return np.transpose(np.asarray(v).reshape(N[0], N[1], N[2], 3), (3, 0, 1, 2))


def benchmark_fft_vs_dense(
    case_factory,
    *,
    n_values=FFT_DENSE_N_VALUES,
    repeats: int = 3,
    logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Compare FFT-backed matvec with dense NumPy multiplication on small grids."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    n_values = list(n_values)
    _log(logger, "start", scan="fft_vs_dense", n_values=n_values, repeats=int(repeats))
    rows = []
    try:
        for N in n_values:
            case = case_factory(N)
            _log(logger, "case_built", scan="fft_vs_dense", case_name=case.name, N=case.N)
            problem, operator = build_problem(case)
            dense_B = operator.to_dense()
            u = np.asarray(problem.wave)
            eta_u = np.einsum("ab...,b...->a...", np.asarray(problem.eps_tensor), u)
            u_flat = _flatten_field(u)
            eta_flat = _flatten_field(eta_u)
            fft_durations = []
            dense_durations = []
            fft_result = None
            dense_result = None
            for _ in range(repeats):
                start = time.perf_counter()
                fft_result = np.asarray(operator.matvec(u))
                fft_durations.append(time.perf_counter() - start)
                start = time.perf_counter()
                dense_result = _unflatten_field(u_flat - dense_B @ eta_flat, case.N)
                dense_durations.append(time.perf_counter() - start)
            den = float(np.linalg.norm(dense_result))
            rel = float(np.linalg.norm(fft_result - dense_result) / den) if den > 0 else 0.0
            row = {
                "case_name": case.name,
                "N": "x".join(str(n) for n in case.N),
                "dof": case.dof,
                "relative_error": rel,
                "fft_avg_sec": float(np.mean(fft_durations)),
                "dense_avg_sec": float(np.mean(dense_durations)),
                "repeats": int(repeats),
            }
            rows.append(row)
            _log(logger, "benchmark_done", scan="fft_vs_dense", **row)
    except Exception as exc:
        _log(logger, "error", scan="fft_vs_dense", error=str(exc))
        raise
    _log(logger, "finish", scan="fft_vs_dense", rows=len(rows))
    return rows


def compute_mie_rcs_diagnostics(
    result_u,
    problem: em3d.Problem,
    *,
    a: float,
    eps_r: complex,
    n_phi: int = 180,
    plane: str = "xy",
) -> dict[str, Any]:
    """Compute normalized Mie RCS shape diagnostics for a sphere case."""
    return em3d.mie.compare_rcs_plane(
        np.asarray(result_u),
        problem,
        a=a,
        eps_r=eps_r,
        n_phi=n_phi,
        plane=plane,
        normalize="max",
    )


def scan_mie_rcs_by_k0a(
    *,
    N=32,
    a: float = RCS_DEFAULT_RADIUS,
    eps_r: float | complex = 2.0,
    k0a_values=RCS_K0A_VALUES,
    n_phi: int = 90,
    max_iter: int = 500,
    rtol: float = 1e-8,
    logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Run normalized RCS/Mie diagnostics for a sphere over size parameters."""
    k0a_values = list(k0a_values)
    _log(logger, "start", scan="mie_rcs_by_k0a", N=N, a=float(a), k0a_values=k0a_values)
    rows = []
    try:
        for k0a in k0a_values:
            case = make_sphere_case(N=N, eps_r=complex(eps_r), k0a=float(k0a), a=a)
            _log(logger, "case_built", scan="mie_rcs_by_k0a", case_name=case.name, k0a=float(k0a))
            problem, operator = build_problem(case)
            result = em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
            _log(
                logger,
                "solver_finished",
                scan="mie_rcs_by_k0a",
                case_name=case.name,
                converged=bool(result.converged),
                iterations=int(result.iterations),
            )
            diagnostics = compute_mie_rcs_diagnostics(
                result.u,
                problem,
                a=a,
                eps_r=complex(eps_r),
                n_phi=n_phi,
            )
            row = {
                "case_name": case.name,
                "N": "x".join(str(n) for n in case.N),
                "k0a": float(k0a),
                "k0": float(case.k0),
                "converged": bool(result.converged),
                "iterations": int(result.iterations),
                "shape_err": float(diagnostics["shape_err"]),
                "scale_ratio": float(diagnostics["scale_ratio"]),
                "abs_rel_err": float(diagnostics["abs_rel_err"]),
                "phi": diagnostics["phi"],
                "sigma_num_norm": diagnostics["sigma_num_norm"],
                "sigma_mie_norm": diagnostics["sigma_mie_norm"],
            }
            rows.append(row)
            _log(
                logger,
                "rcs_done",
                scan="mie_rcs_by_k0a",
                case_name=case.name,
                k0a=float(k0a),
                shape_err=row["shape_err"],
                scale_ratio=row["scale_ratio"],
                abs_rel_err=row["abs_rel_err"],
            )
    except Exception as exc:
        _log(logger, "error", scan="mie_rcs_by_k0a", error=str(exc))
        raise
    _log(logger, "finish", scan="mie_rcs_by_k0a", rows=len(rows))
    return rows


def _analysis_to_row(
    analysis,
    *,
    scenario: str,
    coarse_N: int,
    k0: float,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "coarse_N": int(analysis.coarse_N[0] if analysis.coarse_N else coarse_N),
        "k0": float(k0),
        "mu_real": float(np.real(analysis.mu)),
        "mu_imag": float(np.imag(analysis.mu)),
        "radius": float(analysis.radius),
        "rho": float(analysis.rho),
        "matrix_shape": list(analysis.matrix_shape) if analysis.matrix_shape else None,
        "status": status,
        "error": error,
    }


def scan_gamma0(
    case_factory,
    *,
    coarse_values=(2, 3, 4, 5, 6),
    k_values=range(1, 11),
    scenario="gamma0",
    logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Estimate gamma0 over coarse-grid and k0 series."""
    coarse_values = list(coarse_values)
    k_values = list(k_values)
    _log(logger, "start", scan="gamma0", scenario=scenario, coarse_values=coarse_values, k_values=k_values)
    rows = []
    for coarse_N in coarse_values:
        for k0 in k_values:
            try:
                case = case_factory(coarse_N=coarse_N, k0=float(k0))
                _log(logger, "case_built", scan="gamma0", scenario=scenario, case_name=case.name)
                problem, _ = build_problem(case)
                analysis = estimate_gamma0(problem, coarse_N=(int(coarse_N),) * 3)
                row = _analysis_to_row(analysis, scenario=scenario, coarse_N=coarse_N, k0=float(k0))
                rows.append(row)
                _log(logger, "gamma0_estimated", scan="gamma0", **row)
            except Exception as exc:
                row = {
                    "scenario": scenario,
                    "coarse_N": int(coarse_N),
                    "k0": float(k0),
                    "mu_real": np.nan,
                    "mu_imag": np.nan,
                    "radius": np.nan,
                    "rho": np.nan,
                    "matrix_shape": None,
                    "status": "error",
                    "error": str(exc),
                }
                rows.append(row)
                _log(logger, "error", scan="gamma0", **row)
    _log(logger, "finish", scan="gamma0", scenario=scenario, rows=len(rows))
    return rows


def make_isotropic_gamma0_case(*, coarse_N: int, k0: float):
    return make_anisotropic_ellipsoid_case(N=8, eps_real=2.0, eps_imag=0.0, k0=k0)


def make_anisotropic_gamma0_case(*, coarse_N: int, k0: float):
    return make_anisotropic_ellipsoid_case(
        N=8,
        eps_real=np.diag([2.0, 1.6, 1.3]),
        eps_imag=np.zeros((3, 3)),
        k0=k0,
    )


def make_layered_gamma0_case(*, coarse_N: int, k0: float):
    scale = float(k0) / 10.0
    layers = [
        LayerSpec(
            -0.5,
            -1.0 / 6.0,
            MaterialSpec.anisotropic_lossy(
                np.diag([1.5, 1.4, 1.3]) * (1 + scale),
                np.diag([0.01, 0.01, 0.01]) * scale,
            ),
        ),
        LayerSpec(
            -1.0 / 6.0,
            1.0 / 6.0,
            MaterialSpec.anisotropic_lossy(
                np.diag([2.0, 1.8, 1.6]) * (1 + scale),
                np.diag([0.02, 0.02, 0.02]) * scale,
            ),
        ),
        LayerSpec(
            1.0 / 6.0,
            0.5,
            MaterialSpec.anisotropic_lossy(
                np.diag([2.5, 2.2, 1.9]) * (1 + scale),
                np.diag([0.03, 0.03, 0.03]) * scale,
            ),
        ),
    ]
    return make_layered_box_case(N=8, k0=k0, layers=layers)


def scan_sim_convergence_by_gamma0(
    case: ExperimentCase,
    *,
    coarse_values=(2, 3, 4, 5, 6),
    max_iter: int = 500,
    rtol: float = 1e-6,
    logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Run SIM with gamma0 estimated from several coarse grids."""
    coarse_values = list(coarse_values)
    _log(logger, "start", scan="sim_convergence_by_gamma0", case_name=case.name, coarse_values=coarse_values)
    problem, operator = build_problem(case)
    rows = []
    try:
        for coarse_N in coarse_values:
            analysis = estimate_gamma0(problem, coarse_N=(int(coarse_N),) * 3)
            _log(
                logger,
                "gamma0_estimated",
                scan="sim_convergence_by_gamma0",
                case_name=case.name,
                coarse_N=int(analysis.coarse_N[0] if analysis.coarse_N else coarse_N),
                rho=float(analysis.rho),
                mu_real=float(np.real(analysis.mu)),
                mu_imag=float(np.imag(analysis.mu)),
            )
            run = run_solver(
                problem,
                operator,
                case,
                "SIM",
                gamma0_analysis=analysis,
                max_iter=max_iter,
                rtol=rtol,
            )
            row = run.to_row()
            row.update({
                "coarse_N": int(analysis.coarse_N[0] if analysis.coarse_N else coarse_N),
                "rho": float(analysis.rho),
                "mu_real": float(np.real(analysis.mu)),
                "mu_imag": float(np.imag(analysis.mu)),
                "radius": float(analysis.radius),
                "residual_history": run.residual_history,
            })
            rows.append(row)
            _log(logger, "solver_finished", scan="sim_convergence_by_gamma0", **run.to_row())
    except Exception as exc:
        _log(logger, "error", scan="sim_convergence_by_gamma0", case_name=case.name, error=str(exc))
        raise
    _log(logger, "finish", scan="sim_convergence_by_gamma0", case_name=case.name, rows=len(rows))
    return rows


def run_solver_comparison(
    case: ExperimentCase,
    *,
    sim_coarse_N: int = 6,
    max_iter: int = 500,
    rtol: float = 1e-6,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Compare SIM, BiCGStab, and TwoStep on the same case."""
    _log(logger, "start", scan="solver_comparison", case_name=case.name, sim_coarse_N=int(sim_coarse_N))
    problem, operator = build_problem(case)
    gamma0_analysis = estimate_gamma0(problem, coarse_N=(int(sim_coarse_N),) * 3)
    _log(
        logger,
        "gamma0_estimated",
        scan="solver_comparison",
        case_name=case.name,
        coarse_N=int(gamma0_analysis.coarse_N[0] if gamma0_analysis.coarse_N else sim_coarse_N),
        rho=float(gamma0_analysis.rho),
    )
    runs = run_solver_suite(
        problem,
        operator,
        case,
        ["SIM", "BiCGStab", "TwoStep"],
        gamma0_analysis=gamma0_analysis,
        max_iter=max_iter,
        rtol=rtol,
    )
    for run in runs:
        _log(logger, "solver_finished", scan="solver_comparison", **run.to_row())
    bicg = em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
    if bicg.converged:
        reference_u = bicg.u
        reference_solver = "BiCGStab"
    else:
        two = em3d.TwoStep(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
        reference_u = two.u
        reference_solver = "TwoStep"
    _log(logger, "finish", scan="solver_comparison", case_name=case.name, reference_solver=reference_solver)
    return {
        "case": case,
        "problem": problem,
        "operator": operator,
        "gamma0_analysis": gamma0_analysis,
        "runs": runs,
        "reference_u": reference_u,
        "reference_solver": reference_solver,
    }


def run_quick_experiment(
    *,
    output_root: str | Path = Path("experiments") / "outputs" / "chapter06",
    n_values: list[int] | None = None,
    solver_names: list[str] | None = None,
    max_iter: int = 200,
    rtol: float = 1e-6,
    rcs_n_phi: int = 90,
    mode: str = "quick",
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run a small CPU experiment and persist core artifacts."""
    paths = ensure_output_dirs(output_root)
    if logger is None:
        logger = ExperimentLogger(paths["root"], f"chapter06-{mode}")
    n_values = list(n_values) if n_values is not None else n_series_for_mode("quick")
    solver_names = list(solver_names) if solver_names is not None else ["SIM", "BiCGStab", "TwoStep"]
    _log(
        logger,
        "start",
        scan="quick_experiment",
        mode=mode,
        n_values=n_values,
        solver_names=solver_names,
        max_iter=int(max_iter),
        rtol=float(rtol),
    )

    all_runs: list[SolverRun] = []
    matvec_rows: list[dict[str, Any]] = []
    rcs_diagnostics: dict[str, Any] = {}

    for N in n_values:
        case = make_sphere_case(N=N, eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)
        _log(logger, "case_built", scan="quick_experiment", case_name=case.name, N=case.N)
        build_start = time.perf_counter()
        problem, operator = build_problem(case)
        operator_build_sec = time.perf_counter() - build_start
        gamma0_analysis = estimate_gamma0(problem, coarse_N=(2, 2, 2)) if "SIM" in solver_names else None
        new_runs = run_solver_suite(
            problem,
            operator,
            case,
            solver_names,
            gamma0_analysis=gamma0_analysis,
            max_iter=max_iter,
            rtol=rtol,
        )
        all_runs.extend(new_runs)
        for run in new_runs:
            _log(logger, "solver_finished", scan="quick_experiment", **run.to_row())
        matvec_rows.append(
            benchmark_matvec(
                problem,
                operator,
                case,
                repeats=2,
                operator_build_sec=operator_build_sec,
            )
        )
        if N == n_values[0]:
            bicg = em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
            rcs_diagnostics[case.name] = compute_mie_rcs_diagnostics(
                bicg.u,
                problem,
                a=0.3,
                eps_r=2.0 + 0.0j,
                n_phi=rcs_n_phi,
            )

    save_runs_csv(all_runs, paths["tables"] / "solver_runs.csv")
    save_json({"matvec": matvec_rows}, paths["tables"] / "matvec_timing.json")
    save_json({"rcs": rcs_diagnostics}, paths["raw"] / "rcs_diagnostics.json")
    summary = {
        "mode": mode,
        "n_values": n_values,
        "solver_names": solver_names,
        "num_solver_runs": len(all_runs),
        "output_root": str(paths["root"]),
    }
    save_json(summary, paths["raw"] / "summary.json")
    _log(logger, "finish", scan="quick_experiment", **summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for quick/full chapter 6 experiments."""
    import argparse

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
