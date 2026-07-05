"""Packaged scalar acoustic scattering experiments."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import em3d
import em3d.acoustics as acoustics


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class ExperimentLogger:
    """Append JSONL and text event logs under an experiment output root."""

    output_root: str | Path
    experiment_name: str
    raw_dir: Path = field(init=False)
    jsonl_path: Path = field(init=False)
    text_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.output_root)
        self.raw_dir = root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.experiment_name.replace("/", "-").replace("\\", "-")
        self.jsonl_path = self.raw_dir / f"{safe_name}.jsonl"
        self.text_path = self.raw_dir / f"{safe_name}.log"

    def event(self, event: str, **payload: Any) -> dict[str, Any]:
        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False, default=_json_default)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        payload_text = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, default=_json_default)}"
            for key, value in payload.items()
        )
        text_line = f"{row['time']} {row['event']}"
        if payload_text:
            text_line += f" {payload_text}"
        with self.text_path.open("a", encoding="utf-8") as f:
            f.write(text_line + "\n")
        return row


@dataclass(frozen=True)
class AcousticCase:
    """Configuration for a scalar acoustic scattering experiment."""

    name: str
    kind: str
    N: tuple[int, int, int]
    coarse_N: tuple[int, int, int]
    L: tuple[float, float, float]
    k0: float
    eta_inside: complex
    eta_background: complex
    direction: tuple[float, float, float]
    amplitude: complex
    solver_names: tuple[str, ...]
    radius: float | None = None
    slab_axis: int | None = None
    slab_width_fraction: float | None = None

    @property
    def dof(self) -> int:
        return int(self.N[0] * self.N[1] * self.N[2])

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SolverRun:
    case_name: str
    solver_name: str
    N: tuple[int, int, int]
    dof: int
    converged: bool
    iterations: int
    final_residual: float
    elapsed_sec: float
    residual_history: list[float]

    def to_row(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "solver_name": self.solver_name,
            "N": "x".join(str(n) for n in self.N),
            "dof": self.dof,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_residual": self.final_residual,
            "elapsed_sec": self.elapsed_sec,
        }


def _grid_shape(N: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(N, int):
        return (int(N), int(N), int(N))
    shape = tuple(int(n) for n in N)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"N must be a positive int or 3-tuple, got {N!r}")
    return shape


def make_sphere_case(
    *,
    N: int | tuple[int, int, int] = (64, 64, 64),
    coarse_N: int | tuple[int, int, int] = (6, 6, 6),
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    k0: float = 5.0,
    radius: float = 0.25,
    eta_inside: complex = 2.0 + 0.25j,
    eta_background: complex = 1.0 + 0.0j,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    amplitude: complex = 1.0,
    solver_names: tuple[str, ...] = ("SIM", "BiCGStab", "TwoStep"),
    name: str | None = None,
) -> AcousticCase:
    """Build a spherical acoustic inclusion case."""
    shape = _grid_shape(N)
    coarse_shape = _grid_shape(coarse_N)
    return AcousticCase(
        name=name or f"acoustic_sphere_N{shape[0]}",
        kind="sphere",
        N=shape,
        coarse_N=coarse_shape,
        L=tuple(float(x) for x in L),
        k0=float(k0),
        eta_inside=complex(eta_inside),
        eta_background=complex(eta_background),
        direction=tuple(float(x) for x in direction),
        amplitude=complex(amplitude),
        solver_names=tuple(solver_names),
        radius=float(radius),
    )


def make_layered_case(
    *,
    N: int | tuple[int, int, int] = (64, 64, 64),
    coarse_N: int | tuple[int, int, int] = (6, 6, 6),
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    k0: float = float(np.sqrt(15.0)),
    eta_inside: complex = 2.5 + 1.5j,
    eta_background: complex = 1.0 + 0.0j,
    slab_axis: int = 0,
    slab_width_fraction: float = 0.5,
    direction: tuple[float, float, float] = (1.0, 1.0, 1.0),
    amplitude: complex = 1.0,
    solver_names: tuple[str, ...] = ("SIM", "BiCGStab", "TwoStep"),
    name: str | None = None,
) -> AcousticCase:
    """Build a layered acoustic slab case."""
    shape = _grid_shape(N)
    coarse_shape = _grid_shape(coarse_N)
    return AcousticCase(
        name=name or f"acoustic_layered_N{shape[0]}",
        kind="slab",
        N=shape,
        coarse_N=coarse_shape,
        L=tuple(float(x) for x in L),
        k0=float(k0),
        eta_inside=complex(eta_inside),
        eta_background=complex(eta_background),
        direction=tuple(float(x) for x in direction),
        amplitude=complex(amplitude),
        solver_names=tuple(solver_names),
        slab_axis=int(slab_axis),
        slab_width_fraction=float(slab_width_fraction),
    )


def make_homogeneous_case(**kwargs) -> AcousticCase:
    """Build a homogeneous acoustic case."""
    case = make_sphere_case(**kwargs)
    return AcousticCase(
        name=case.name.replace("sphere", "homogeneous"),
        kind="homogeneous",
        N=case.N,
        coarse_N=case.coarse_N,
        L=case.L,
        k0=case.k0,
        eta_inside=case.eta_inside,
        eta_background=case.eta_inside,
        direction=case.direction,
        amplitude=case.amplitude,
        solver_names=case.solver_names,
    )


def build_acoustic_problem(
    case: AcousticCase,
    *,
    precision: em3d.Precision = em3d.Precision.DOUBLE,
):
    """Build a CPU acoustic problem and FFT operator for a packaged case."""
    be = em3d.Backend.numpy(precision)
    grid = em3d.Grid(N=case.N, L=case.L, center=(0.0, 0.0, 0.0), backend=be)
    if case.kind == "sphere":
        eta = acoustics.eta_sphere(
            grid,
            center=(0.0, 0.0, 0.0),
            radius=float(case.radius),
            eta_inside=case.eta_inside,
            eta_outside=case.eta_background,
        )
    elif case.kind == "slab":
        eta = acoustics.eta_slab(
            grid,
            eta_inside=case.eta_inside,
            eta_outside=case.eta_background,
            axis=int(case.slab_axis),
            width_fraction=float(case.slab_width_fraction),
        )
    elif case.kind == "homogeneous":
        eta = acoustics.eta_homogeneous(grid, case.eta_inside)
    else:
        raise ValueError(f"unknown acoustic case kind {case.kind!r}")
    problem = acoustics.make_acoustic_problem(
        grid,
        eta,
        k0=case.k0,
        direction=case.direction,
        amplitude=case.amplitude,
    )
    return problem, acoustics.AcousticOperator(problem)


def _ensure_output_dirs(root: str | Path) -> dict[str, Path]:
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


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _save_runs_csv(runs: list[_SolverRun], path: str | Path) -> None:
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
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(run.to_row())


def _solve(problem, operator, solver_name: str, analysis, *, max_iter: int, rtol: float):
    if solver_name == "SIM":
        cfg = em3d.SolverConfig(max_iter=max_iter, rtol=rtol, **analysis.as_solver_config_kwargs())
        return em3d.SIM(cfg).solve(operator, problem.wave)
    if solver_name == "BiCGStab":
        return em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
    if solver_name == "TwoStep":
        return em3d.TwoStep(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
    raise ValueError(f"unknown solver_name {solver_name!r}")


def _select_reference_run(runs: list[_SolverRun]) -> _SolverRun:
    converged = {run.solver_name: run for run in runs if run.converged}
    for name in ("BiCGStab", "TwoStep", "SIM"):
        if name in converged:
            return converged[name]
    return min(runs, key=lambda run: run.final_residual)


def run_acoustic_experiment(
    *,
    case: AcousticCase | None = None,
    output_root: str | Path = Path("experiments") / "outputs" / "acoustic-scattering",
    max_iter: int = 500,
    rtol: float = 1e-6,
    n_angles: int = 180,
    make_plots: bool = True,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run an acoustic scattering experiment and persist metrics and artifacts."""
    case = case if case is not None else make_sphere_case()
    paths = _ensure_output_dirs(output_root)
    logger = logger if logger is not None else ExperimentLogger(paths["root"], "acoustic_scattering")
    logger.event("start", case_name=case.name, N=case.N, coarse_N=case.coarse_N, solver_names=case.solver_names)

    problem, operator = build_acoustic_problem(case)
    analysis = acoustics.gamma0.estimate_from_problem(problem, coarse_N=case.coarse_N)
    logger.event(
        "gamma0_estimated",
        coarse_N=analysis.coarse_N,
        mu=analysis.mu,
        radius=analysis.radius,
        rho=analysis.rho,
    )

    runs: list[_SolverRun] = []
    results = {}
    for solver_name in case.solver_names:
        start = time.perf_counter()
        result = _solve(problem, operator, solver_name, analysis, max_iter=max_iter, rtol=rtol)
        elapsed = time.perf_counter() - start
        history = [float(x) for x in result.residual_history]
        run = _SolverRun(
            case_name=case.name,
            solver_name=solver_name,
            N=case.N,
            dof=case.dof,
            converged=bool(result.converged),
            iterations=int(result.iterations),
            final_residual=float(history[-1] if history else np.inf),
            elapsed_sec=float(elapsed),
            residual_history=history,
        )
        runs.append(run)
        results[solver_name] = result
        logger.event("solver_finished", **run.to_row())

    reference_run = _select_reference_run(runs)
    u = results[reference_run.solver_name].u
    _save_runs_csv(runs, paths["tables"] / "acoustic_solver_runs.csv")
    _save_json(
        {"runs": [{**run.to_row(), "residual_history": run.residual_history} for run in runs]},
        paths["raw"] / "acoustic_residual_histories.json",
    )

    phi, sigma = acoustics.pattern_plane(u, problem, plane="xy", n_angles=n_angles, normalize="max")
    if make_plots:
        acoustics.visualization.plot_scalar_slices(u, problem.grid, output_dir=paths["figures"], prefix="acoustic")
        acoustics.visualization.plot_pattern(
            phi,
            sigma,
            filename=paths["figures"] / "acoustic_pattern_xy.png",
            polar=False,
            title="Acoustic scattering pattern",
        )
        acoustics.visualization.plot_pattern(
            phi,
            sigma,
            filename=paths["figures"] / "acoustic_pattern_xy_polar.png",
            polar=True,
            title="Acoustic scattering pattern",
        )

    summary = {
        "case_name": case.name,
        "kind": case.kind,
        "N": "x".join(str(n) for n in case.N),
        "coarse_N": "x".join(str(n) for n in case.coarse_N),
        "k0": float(case.k0),
        "eta_inside": case.eta_inside,
        "eta_background": case.eta_background,
        "solver_names": list(case.solver_names),
        "reference_solver": reference_run.solver_name,
        "gamma0": {
            "mu_real": float(np.real(analysis.mu)),
            "mu_imag": float(np.imag(analysis.mu)),
            "radius": float(analysis.radius),
            "rho": float(analysis.rho),
        },
        "pattern": {
            "max": float(np.max(sigma)) if sigma.size else 0.0,
            "mean": float(np.mean(sigma)) if sigma.size else 0.0,
        },
        "output_root": str(paths["root"]),
    }
    _save_json(summary, paths["raw"] / "acoustic_summary.json")
    logger.event("finish", **summary)
    return summary


__all__ = [
    "AcousticCase",
    "ExperimentLogger",
    "build_acoustic_problem",
    "make_homogeneous_case",
    "make_layered_case",
    "make_sphere_case",
    "run_acoustic_experiment",
]
