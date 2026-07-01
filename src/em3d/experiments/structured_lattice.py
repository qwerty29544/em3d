"""Structured inclusion lattice experiment shipped with the em3d package."""
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


@dataclass(frozen=True)
class MaterialSpec:
    """Simple dielectric material model for packaged experiments."""

    kind: str
    eps_real: Any
    eps_imag: Any = 0.0

    @classmethod
    def isotropic(cls, eps_r: float | complex) -> "MaterialSpec":
        eps = complex(eps_r)
        return cls(kind="isotropic", eps_real=float(eps.real), eps_imag=float(eps.imag))

    @classmethod
    def anisotropic_lossy(cls, eps_real, eps_imag) -> "MaterialSpec":
        return cls(
            kind="anisotropic_lossy",
            eps_real=np.array(eps_real, dtype=np.float64, copy=True),
            eps_imag=np.array(eps_imag, dtype=np.float64, copy=True),
        )


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
class InclusionSpec:
    """One axis-aligned ellipsoidal dielectric inclusion."""

    center: tuple[float, float, float]
    radius: tuple[float, float, float]
    material: MaterialSpec


@dataclass(frozen=True)
class StructuredLatticeCase:
    """Configuration for a structured inclusion lattice experiment."""

    name: str
    N: tuple[int, int, int]
    coarse_N: tuple[int, int, int]
    L: tuple[float, float, float]
    k0: float
    lattice_shape: tuple[int, int, int]
    inclusions: tuple[InclusionSpec, ...]
    wave_orient: tuple[float, float, float]
    wave_amplitude: tuple[float, float, float]
    solver_names: tuple[str, ...]

    @property
    def dof(self) -> int:
        return 3 * self.N[0] * self.N[1] * self.N[2]

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


def _grid_shape(N: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(N, int):
        return (N, N, N)
    shape = tuple(int(n) for n in N)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"N must be a positive int or 3-tuple, got {N!r}")
    return shape


def _radius_tuple(radius: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(radius, (int, float)):
        value = float(radius)
        return (value, value, value)
    out = tuple(float(x) for x in radius)
    if len(out) != 3 or any(x <= 0.0 for x in out):
        raise ValueError(f"radius must be positive float or 3-tuple, got {radius!r}")
    return out


def _default_centers(
    lattice_shape: tuple[int, int, int],
    *,
    L: tuple[float, float, float],
    margin_fraction: float = 0.16,
) -> list[tuple[float, float, float]]:
    axes = []
    for count, length in zip(lattice_shape, L):
        if count <= 0:
            raise ValueError(f"lattice_shape values must be positive, got {lattice_shape!r}")
        half = 0.5 * float(length)
        margin = float(length) * margin_fraction
        if count == 1:
            axes.append(np.array([0.0], dtype=np.float64))
        else:
            axes.append(np.linspace(-half + margin, half - margin, count))
    return [(float(x), float(y), float(z)) for x in axes[0] for y in axes[1] for z in axes[2]]


def make_structured_lattice_case(
    *,
    N: int | tuple[int, int, int] = (100, 100, 100),
    coarse_N: int | tuple[int, int, int] = (9, 9, 9),
    lattice_shape: tuple[int, int, int] = (5, 5, 5),
    inclusion_radius: float | tuple[float, float, float] = (0.045, 0.045, 0.045),
    material: MaterialSpec | None = None,
    k0: float = 8.0,
    L: tuple[float, float, float] = (1.0, 1.0, 1.0),
    wave_orient: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wave_amplitude: tuple[float, float, float] = (1.0, 0.0, 0.0),
    solver_names: tuple[str, ...] = ("SIM", "BiCGStab", "TwoStep"),
    name: str | None = None,
) -> StructuredLatticeCase:
    """Build a regular lattice of ellipsoidal inclusions."""
    shape = _grid_shape(N)
    coarse_shape = _grid_shape(coarse_N)
    lattice = _grid_shape(lattice_shape)
    radius = _radius_tuple(inclusion_radius)
    material = material if material is not None else MaterialSpec.isotropic(2.5 + 0.02j)
    inclusions = tuple(
        InclusionSpec(center=center, radius=radius, material=material)
        for center in _default_centers(lattice, L=L)
    )
    case_name = name or f"structured_lattice_{lattice[0]}x{lattice[1]}x{lattice[2]}_N{shape[0]}"
    return StructuredLatticeCase(
        name=case_name,
        N=shape,
        coarse_N=coarse_shape,
        L=tuple(float(x) for x in L),
        k0=float(k0),
        lattice_shape=lattice,
        inclusions=inclusions,
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        solver_names=tuple(solver_names),
    )


def _material_eps(material: MaterialSpec) -> tuple:
    return material.eps_real, material.eps_imag


def _eta_matrix(eps_real, eps_imag) -> np.ndarray:
    if np.ndim(eps_real) == 0:
        return (float(eps_real) - 1.0 + 1j * float(eps_imag)) * np.eye(3, dtype=np.complex128)
    return np.asarray(eps_real, dtype=np.float64) - np.eye(3) + 1j * np.asarray(eps_imag, dtype=np.float64)


def _structured_lattice_refraction(grid: em3d.Grid, case: StructuredLatticeCase):
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for inclusion in case.inclusions:
        cx, cy, cz = inclusion.center
        rx, ry, rz = inclusion.radius
        metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
        mask = metric <= 1.0
        eps_real, eps_imag = _material_eps(inclusion.material)
        eta = _eta_matrix(eps_real, eps_imag)
        for i in range(3):
            for j in range(3):
                out[i, j] = xp.where(mask, eta[i, j], out[i, j])
    return out


def build_structured_lattice_problem(
    case: StructuredLatticeCase,
    *,
    precision: em3d.Precision = em3d.Precision.DOUBLE,
) -> tuple[em3d.Problem, em3d.Operator]:
    """Build a CPU Problem and Operator for a structured inclusion lattice."""
    be = em3d.Backend.numpy(precision)
    grid = em3d.Grid(N=case.N, L=case.L, center=(0.0, 0.0, 0.0), backend=be)
    eps_tensor = _structured_lattice_refraction(grid, case)
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


def _solver_result(problem, operator, solver_name: str, gamma0_analysis, *, max_iter: int, rtol: float):
    if solver_name == "SIM":
        cfg = em3d.SolverConfig(max_iter=max_iter, rtol=rtol, **gamma0_analysis.as_solver_config_kwargs())
        return em3d.SIM(cfg).solve(operator, problem.wave)
    if solver_name == "BiCGStab":
        return em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
    if solver_name == "TwoStep":
        return em3d.TwoStep(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
    raise ValueError(f"unknown solver_name {solver_name!r}")


def _select_reference_run(runs: list[_SolverRun], preferred: tuple[str, ...]) -> _SolverRun:
    converged = {run.solver_name: run for run in runs if run.converged}
    for name in preferred:
        if name in converged:
            return converged[name]
    return min(runs, key=lambda run: run.final_residual)


def _estimate_gamma0(problem: em3d.Problem, coarse_N: tuple[int, int, int]):
    shape = tuple(int(n) for n in coarse_N)
    max_coarse_N = max(shape)
    last_error: ValueError | None = None
    while max(shape) <= max_coarse_N:
        try:
            return em3d.gamma0.estimate_from_problem(problem, coarse_N=shape)
        except ValueError as exc:
            last_error = exc
            shape = tuple(n + 1 for n in shape)
    if last_error is not None:
        raise last_error
    raise ValueError(f"invalid coarse_N={coarse_N!r}")


def _run_to_history(run: _SolverRun) -> dict[str, Any]:
    row = run.to_row()
    row["residual_history"] = run.residual_history
    return row


def _plot_three_field_slices(u, grid, *, output_dir: Path, prefix: str) -> None:
    for plane in ("xy", "xz", "yz"):
        em3d.vis.plot_field_scalar_slice(
            u,
            grid,
            plane=plane,
            part="abs",
            title=f"|E| slice {plane}",
            filename=str(output_dir / f"{prefix}_scalar_abs_{plane}.png"),
        )
        em3d.vis.plot_field_vector_slice(
            u,
            grid,
            plane=plane,
            part="real",
            title=f"Re(E) vector slice {plane}",
            filename=str(output_dir / f"{prefix}_vector_real_{plane}.png"),
        )


def _plot_residual_histories(runs: list[_SolverRun], *, output_dir: Path) -> None:
    plt = em3d.vis._require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for run in runs:
        if run.residual_history:
            ax.semilogy(np.arange(len(run.residual_history)), run.residual_history, label=run.solver_name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("Structured lattice convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()
    fig.savefig(str(output_dir / "structured_lattice_residuals.png"), dpi=160, bbox_inches="tight")


def _plot_structured_lattice_rcs(u, problem: em3d.Problem, *, output_dir: Path, n_phi: int) -> dict[str, Any]:
    phi, sigma = em3d.farfield.rcs_plane(u, problem, n_phi=n_phi, plane="xy")
    em3d.vis.plot_rcs(
        phi,
        sigma,
        title="Structured lattice RCS, xy",
        filename=str(output_dir / "structured_lattice_rcs_xy.png"),
    )
    em3d.vis.plot_rcs_polar(
        phi,
        sigma,
        title="Structured lattice RCS, xy",
        filename=str(output_dir / "structured_lattice_rcs_xy_polar.png"),
    )
    return {
        "phi": phi,
        "sigma": sigma,
        "sigma_max": float(np.max(sigma)) if sigma.size else 0.0,
        "sigma_mean": float(np.mean(sigma)) if sigma.size else 0.0,
    }


def run_structured_lattice_experiment(
    *,
    case: StructuredLatticeCase | None = None,
    output_root: str | Path = Path("experiments") / "outputs" / "structured-lattice",
    max_iter: int = 500,
    rtol: float = 1e-6,
    rcs_n_phi: int = 120,
    make_plots: bool = True,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Run the structured lattice experiment and persist metrics and plots."""
    case = case if case is not None else make_structured_lattice_case()
    paths = _ensure_output_dirs(output_root)
    logger = logger if logger is not None else ExperimentLogger(paths["root"], "structured_lattice")
    logger.event(
        "start",
        case_name=case.name,
        N=case.N,
        coarse_N=case.coarse_N,
        solver_names=case.solver_names,
        inclusions=len(case.inclusions),
    )

    problem, operator = build_structured_lattice_problem(case)
    gamma0_analysis = _estimate_gamma0(problem, case.coarse_N)
    logger.event(
        "gamma0_estimated",
        coarse_N=gamma0_analysis.coarse_N,
        mu=gamma0_analysis.mu,
        radius=gamma0_analysis.radius,
        rho=gamma0_analysis.rho,
    )

    runs: list[_SolverRun] = []
    results_by_solver = {}
    for solver_name in case.solver_names:
        start = time.perf_counter()
        result = _solver_result(
            problem,
            operator,
            solver_name,
            gamma0_analysis,
            max_iter=max_iter,
            rtol=rtol,
        )
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
        results_by_solver[solver_name] = result
        logger.event("solver_finished", **run.to_row())

    reference_run = _select_reference_run(runs, ("BiCGStab", "TwoStep", "SIM"))
    reference_result = results_by_solver[reference_run.solver_name]
    reference_solver = reference_run.solver_name
    u = np.asarray(reference_result.u)

    _save_runs_csv(runs, paths["tables"] / "structured_lattice_solver_runs.csv")
    _save_json({"runs": [_run_to_history(run) for run in runs]}, paths["raw"] / "structured_lattice_residual_histories.json")

    rcs = {"sigma_max": 0.0, "sigma_mean": 0.0}
    if make_plots:
        _plot_three_field_slices(u, problem.grid, output_dir=paths["figures"], prefix="structured_lattice")
        _plot_residual_histories(runs, output_dir=paths["figures"])
        rcs = _plot_structured_lattice_rcs(u, problem, output_dir=paths["figures"], n_phi=rcs_n_phi)
    else:
        phi, sigma = em3d.farfield.rcs_plane(u, problem, n_phi=rcs_n_phi, plane="xy")
        rcs = {
            "phi": phi,
            "sigma": sigma,
            "sigma_max": float(np.max(sigma)) if sigma.size else 0.0,
            "sigma_mean": float(np.mean(sigma)) if sigma.size else 0.0,
        }

    summary = {
        "case_name": case.name,
        "N": "x".join(str(n) for n in case.N),
        "coarse_N": "x".join(str(n) for n in case.coarse_N),
        "lattice_shape": "x".join(str(n) for n in case.lattice_shape),
        "num_inclusions": len(case.inclusions),
        "k0": float(case.k0),
        "solver_names": list(case.solver_names),
        "num_solver_runs": len(runs),
        "reference_solver": reference_solver,
        "gamma0": {
            "mu_real": float(np.real(gamma0_analysis.mu)),
            "mu_imag": float(np.imag(gamma0_analysis.mu)),
            "radius": float(gamma0_analysis.radius),
            "rho": float(gamma0_analysis.rho),
        },
        "rcs": {
            "sigma_max": rcs["sigma_max"],
            "sigma_mean": rcs["sigma_mean"],
        },
        "output_root": str(paths["root"]),
    }
    _save_json(summary, paths["raw"] / "structured_lattice_summary.json")
    logger.event("finish", **summary)
    return summary


__all__ = [
    "ExperimentLogger",
    "InclusionSpec",
    "MaterialSpec",
    "StructuredLatticeCase",
    "build_structured_lattice_problem",
    "make_structured_lattice_case",
    "run_structured_lattice_experiment",
]
