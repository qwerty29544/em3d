# Chapter 06 EM Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-first, reproducible experimental harness and notebook for chapter 6 electrodynamic VIE experiments.

**Architecture:** Keep reusable computation outside the public package in `experiments/chapter06_em.py`, and keep `notebooks/chapter-06-em.ipynb` as a thin narrative layer. The helper module builds `em3d` problems, runs solvers, collects metrics, computes RCS diagnostics, saves artifacts, and exposes a quick/full run mode.

**Tech Stack:** Python 3.11+, NumPy, matplotlib through `em3d.vis`, local `em3d`, pytest, optional `nbformat` for notebook generation if available.

---

## File Structure

- Create: `experiments/__init__.py`
  - Makes `experiments` importable by tests and local scripts.
- Create: `experiments/chapter06_em.py`
  - Owns dataclasses, case builders, solver runners, gamma0 helpers, RCS diagnostics, benchmark helpers, artifact saving, and CLI.
- Create: `tests/test_chapter06_em.py`
  - Unit and smoke tests for the experimental helper module.
- Create: `notebooks/chapter-06-em.ipynb`
  - Narrative notebook with sections 6.1-6.10.
- Modify: `pyproject.toml`
  - Add pytest marker `slow` if notebook/full experiment tests need it.
- Modify: `README.md`
  - Add a short pointer to the chapter 6 notebook after the implementation works.

Do not add `experiments` to `tool.setuptools.packages.find`; it is intentionally not public package API.

---

## Task 1: Experiment Module Skeleton and Run Modes

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/chapter06_em.py`
- Create: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for run modes and output directories**

Add to `tests/test_chapter06_em.py`:

```python
from pathlib import Path

import pytest

from experiments import chapter06_em as c6


def test_n_series_for_mode_quick_and_full():
    assert c6.n_series_for_mode("quick") == [8, 16, 24]
    assert c6.n_series_for_mode("full") == [8, 16, 24, 32, 40, 48, 56, 64]


def test_n_series_for_mode_rejects_unknown():
    with pytest.raises(ValueError, match="mode"):
        c6.n_series_for_mode("interactive")


def test_ensure_output_dirs_creates_expected_tree(tmp_path):
    paths = c6.ensure_output_dirs(tmp_path / "chapter06")
    assert paths["root"] == tmp_path / "chapter06"
    assert paths["raw"].is_dir()
    assert paths["tables"].is_dir()
    assert paths["figures"].is_dir()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
```

Expected: import failure because `experiments.chapter06_em` does not exist.

- [ ] **Step 3: Create minimal importable module**

Create `experiments/__init__.py`:

```python
"""Research experiment helpers for the em3d repository."""
```

Create `experiments/chapter06_em.py`:

```python
"""CPU-first experiments for chapter 6 electrodynamic VIE studies."""
from __future__ import annotations

from pathlib import Path


N_SERIES_FULL = [8, 16, 24, 32, 40, 48, 56, 64]
N_SERIES_QUICK = [8, 16, 24]


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
```

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
```

Expected: `3 passed`.

- [ ] **Step 5: Commit checkpoint**

If the working tree is isolated for this task, run:

```powershell
git add experiments/__init__.py experiments/chapter06_em.py tests/test_chapter06_em.py
git commit -m "feat(experiments): add chapter 6 harness skeleton"
```

If unrelated staged changes already exist, skip the commit and record the checkpoint in the session summary.

---

## Task 2: Dataclasses and Case Builders

**Files:**
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for dataclasses and case builders**

Append to `tests/test_chapter06_em.py`:

```python
import numpy as np


def test_make_sphere_case_uses_cubic_grid_and_transverse_wave():
    case = c6.make_sphere_case(N=16, eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)
    assert case.name == "sphere_eps2_k0a1_N16"
    assert case.N == (16, 16, 16)
    assert case.L == (1.0, 1.0, 1.0)
    assert case.k0 == pytest.approx(1.0 / 0.3)
    assert case.geometry == "sphere"
    assert case.radius == (0.3, 0.3, 0.3)
    assert case.wave_orient == (0.0, 0.0, 1.0)
    assert case.wave_amplitude == (1.0, 0.0, 0.0)


def test_make_anisotropic_ellipsoid_case_preserves_tensor():
    eps_real = np.array([[2.0, 0.1, 0.0], [0.1, 1.6, 0.0], [0.0, 0.0, 1.3]])
    eps_imag = np.zeros((3, 3))
    case = c6.make_anisotropic_ellipsoid_case(N=8, eps_real=eps_real, eps_imag=eps_imag, k0=2.5)
    assert case.name == "anisotropic_ellipsoid_N8"
    assert case.N == (8, 8, 8)
    assert case.geometry == "ellipsoid"
    np.testing.assert_allclose(case.eps_real, eps_real)
    np.testing.assert_allclose(case.eps_imag, eps_imag)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_make_sphere_case_uses_cubic_grid_and_transverse_wave tests/test_chapter06_em.py::test_make_anisotropic_ellipsoid_case_preserves_tensor -q -p no:cacheprovider
```

Expected: failures because `make_sphere_case` and `make_anisotropic_ellipsoid_case` are undefined.

- [ ] **Step 3: Implement dataclasses and builders**

Add to `experiments/chapter06_em.py`:

```python
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentCase:
    """Configuration for one electrodynamic VIE experiment."""

    name: str
    N: tuple[int, int, int]
    L: tuple[float, float, float]
    k0: float
    geometry: str
    eps_real: Any
    eps_imag: Any
    center: tuple[float, float, float]
    radius: tuple[float, float, float]
    wave_orient: tuple[float, float, float]
    wave_amplitude: tuple[float, float, float]

    @property
    def dof(self) -> int:
        return 3 * self.N[0] * self.N[1] * self.N[2]

    def to_jsonable(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("eps_real", "eps_imag"):
            value = out[key]
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        return out


@dataclass(frozen=True)
class SolverRun:
    """Serializable solver result summary for tables and plots."""

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


def _cube_shape(N: int) -> tuple[int, int, int]:
    N = int(N)
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    return (N, N, N)


def make_sphere_case(N: int, *, eps_r: complex, k0a: float, a: float) -> ExperimentCase:
    """Build the isotropic sphere case used for Mie RCS comparison."""
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
    if k0a <= 0:
        raise ValueError(f"k0a must be positive, got {k0a}")
    N_tuple = _cube_shape(N)
    eps_r = complex(eps_r)
    eps_label = f"{eps_r.real:g}" if abs(eps_r.imag) < 1e-15 else f"{eps_r.real:g}_{eps_r.imag:g}j"
    return ExperimentCase(
        name=f"sphere_eps{eps_label}_k0a{k0a:g}_N{N}",
        N=N_tuple,
        L=(1.0, 1.0, 1.0),
        k0=float(k0a / a),
        geometry="sphere",
        eps_real=float(eps_r.real),
        eps_imag=float(eps_r.imag),
        center=(0.0, 0.0, 0.0),
        radius=(float(a), float(a), float(a)),
        wave_orient=(0.0, 0.0, 1.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )


def make_anisotropic_ellipsoid_case(N: int, *, eps_real, eps_imag, k0: float) -> ExperimentCase:
    """Build a representative anisotropic ellipsoid case."""
    N_tuple = _cube_shape(N)
    eps_real = np.asarray(eps_real, dtype=np.float64)
    eps_imag = np.asarray(eps_imag, dtype=np.float64)
    if eps_real.shape != (3, 3) or eps_imag.shape != (3, 3):
        raise ValueError("eps_real and eps_imag must both have shape (3, 3)")
    return ExperimentCase(
        name=f"anisotropic_ellipsoid_N{N}",
        N=N_tuple,
        L=(1.0, 1.0, 1.0),
        k0=float(k0),
        geometry="ellipsoid",
        eps_real=eps_real,
        eps_imag=eps_imag,
        center=(0.0, 0.0, 0.0),
        radius=(0.25, 0.18, 0.30),
        wave_orient=(0.0, 0.0, 1.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )
```

- [ ] **Step 4: Verify targeted tests pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_make_sphere_case_uses_cubic_grid_and_transverse_wave tests/test_chapter06_em.py::test_make_anisotropic_ellipsoid_case_preserves_tensor -q -p no:cacheprovider
```

Expected: `2 passed`.

- [ ] **Step 5: Run all helper tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
```

Expected: all tests in `tests/test_chapter06_em.py` pass.

---

## Task 3: Problem Construction

**Files:**
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for `build_problem`**

Append to `tests/test_chapter06_em.py`:

```python
import em3d


def test_build_problem_sphere_shapes():
    case = c6.make_sphere_case(N=8, eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)
    problem, operator = c6.build_problem(case)
    assert isinstance(problem, em3d.Problem)
    assert isinstance(operator, em3d.Operator)
    assert problem.grid.N == (8, 8, 8)
    assert problem.eps_tensor.shape == (3, 3, 8, 8, 8)
    assert problem.wave.shape == (3, 8, 8, 8)
    assert problem.k0 == pytest.approx(case.k0)


def test_build_problem_rejects_unknown_geometry():
    case = c6.ExperimentCase(
        name="bad",
        N=(8, 8, 8),
        L=(1.0, 1.0, 1.0),
        k0=1.0,
        geometry="cube",
        eps_real=2.0,
        eps_imag=0.0,
        center=(0.0, 0.0, 0.0),
        radius=(0.3, 0.3, 0.3),
        wave_orient=(0.0, 0.0, 1.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="geometry"):
        c6.build_problem(case)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_build_problem_sphere_shapes tests/test_chapter06_em.py::test_build_problem_rejects_unknown_geometry -q -p no:cacheprovider
```

Expected: failures because `build_problem` is undefined.

- [ ] **Step 3: Implement `build_problem`**

Add to `experiments/chapter06_em.py`:

```python
import em3d


def build_problem(
    case: ExperimentCase,
    *,
    precision: em3d.Precision = em3d.Precision.DOUBLE,
) -> tuple[em3d.Problem, em3d.Operator]:
    """Build a CPU Problem and Operator for one experiment case."""
    be = em3d.Backend.numpy(precision)
    grid = em3d.Grid(N=case.N, L=case.L, center=case.center, backend=be)
    if case.geometry in {"sphere", "ellipsoid"}:
        eps_tensor = em3d.ellipsis_refraction(
            grid,
            eps_real=case.eps_real,
            eps_imag=case.eps_imag,
            center=case.center,
            radius=case.radius,
        )
    else:
        raise ValueError(f"geometry must be 'sphere' or 'ellipsoid', got {case.geometry!r}")
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
```

- [ ] **Step 4: Verify targeted tests pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_build_problem_sphere_shapes tests/test_chapter06_em.py::test_build_problem_rejects_unknown_geometry -q -p no:cacheprovider
```

Expected: `2 passed`.

---

## Task 4: Gamma0 and Solver Runner

**Files:**
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for gamma0 and solvers**

Append to `tests/test_chapter06_em.py`:

```python
def test_estimate_gamma0_returns_analysis():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, _ = c6.build_problem(case)
    analysis = c6.estimate_gamma0(problem, coarse_N=(2, 2, 2))
    assert analysis.coarse_N == (2, 2, 2)
    assert analysis.matrix_shape == (24, 24)
    assert analysis.radius > 0.0


def test_run_solver_bicgstab_smoke():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    run = c6.run_solver(problem, operator, case, "BiCGStab", max_iter=50, rtol=1e-5)
    assert run.case_name == case.name
    assert run.solver_name == "BiCGStab"
    assert run.N == (8, 8, 8)
    assert run.dof == 3 * 8 * 8 * 8
    assert run.iterations >= 0
    assert run.final_residual >= 0.0
    assert run.elapsed_sec >= 0.0
    assert len(run.residual_history) >= 1


def test_run_solver_suite_returns_requested_solvers():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    analysis = c6.estimate_gamma0(problem, coarse_N=(2, 2, 2))
    runs = c6.run_solver_suite(
        problem,
        operator,
        case,
        ["SIM", "BiCGStab", "TwoStep"],
        gamma0_analysis=analysis,
        max_iter=50,
        rtol=1e-5,
    )
    assert [run.solver_name for run in runs] == ["SIM", "BiCGStab", "TwoStep"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_estimate_gamma0_returns_analysis tests/test_chapter06_em.py::test_run_solver_bicgstab_smoke tests/test_chapter06_em.py::test_run_solver_suite_returns_requested_solvers -q -p no:cacheprovider
```

Expected: failures because runner functions are undefined.

- [ ] **Step 3: Implement gamma0 and solver functions**

Add to `experiments/chapter06_em.py`:

```python
import time


def estimate_gamma0(problem: em3d.Problem, *, coarse_N=(4, 4, 4)) -> em3d.gamma0.Gamma0Analysis:
    """Estimate gamma0 from the dense spectrum of the coarse original operator."""
    return em3d.gamma0.estimate_from_problem(problem, coarse_N=coarse_N)


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
```

- [ ] **Step 4: Verify targeted tests pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_estimate_gamma0_returns_analysis tests/test_chapter06_em.py::test_run_solver_bicgstab_smoke tests/test_chapter06_em.py::test_run_solver_suite_returns_requested_solvers -q -p no:cacheprovider
```

Expected: `3 passed`.

---

## Task 5: RCS Diagnostics, Benchmarks, and Artifact Saving

**Files:**
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for diagnostics and persistence**

Append to `tests/test_chapter06_em.py`:

```python
def test_save_runs_csv_writes_solver_rows(tmp_path):
    run = c6.SolverRun(
        case_name="case",
        solver_name="BiCGStab",
        N=(8, 8, 8),
        dof=1536,
        converged=True,
        iterations=4,
        final_residual=1e-6,
        elapsed_sec=0.1,
        residual_history=[1.0, 1e-6],
    )
    path = tmp_path / "runs.csv"
    c6.save_runs_csv([run], path)
    text = path.read_text(encoding="utf-8")
    assert "case_name,solver_name,N,dof,converged,iterations,final_residual,elapsed_sec" in text
    assert "case,BiCGStab,8x8x8,1536,True,4,1e-06,0.1" in text


def test_save_json_roundtrip(tmp_path):
    path = tmp_path / "data.json"
    c6.save_json({"value": 1, "items": [1, 2]}, path)
    assert path.read_text(encoding="utf-8").startswith("{")


def test_benchmark_matvec_returns_timings():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    row = c6.benchmark_matvec(problem, operator, case, repeats=2)
    assert row["case_name"] == case.name
    assert row["N"] == "8x8x8"
    assert row["dof"] == case.dof
    assert row["matvec_avg_sec"] >= 0.0
    assert row["operator_build_sec"] == 0.0


def test_compute_mie_rcs_diagnostics_smoke():
    case = c6.make_sphere_case(N=8, eps_r=1.2 + 0.0j, k0a=0.5, a=0.25)
    problem, operator = c6.build_problem(case)
    run_result = em3d.BiCGStab(em3d.SolverConfig(max_iter=50, rtol=1e-5)).solve(operator, problem.wave)
    diagnostics = c6.compute_mie_rcs_diagnostics(
        run_result.u,
        problem,
        a=0.25,
        eps_r=1.2 + 0.0j,
        n_phi=36,
    )
    assert diagnostics["phi"].shape == (36,)
    assert diagnostics["sigma_num_norm"].shape == (36,)
    assert diagnostics["shape_err"] >= 0.0
    assert diagnostics["scale_ratio"] >= 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_save_runs_csv_writes_solver_rows tests/test_chapter06_em.py::test_save_json_roundtrip tests/test_chapter06_em.py::test_benchmark_matvec_returns_timings tests/test_chapter06_em.py::test_compute_mie_rcs_diagnostics_smoke -q -p no:cacheprovider
```

Expected: failures because these helper functions are undefined.

- [ ] **Step 3: Implement persistence, benchmark, and RCS helpers**

Add to `experiments/chapter06_em.py`:

```python
import csv
import json


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
```

- [ ] **Step 4: Verify targeted tests pass**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_save_runs_csv_writes_solver_rows tests/test_chapter06_em.py::test_save_json_roundtrip tests/test_chapter06_em.py::test_benchmark_matvec_returns_timings tests/test_chapter06_em.py::test_compute_mie_rcs_diagnostics_smoke -q -p no:cacheprovider
```

Expected: `4 passed`.

---

## Task 6: Quick Experiment Orchestrator and CLI

**Files:**
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing smoke test for quick mode**

Append to `tests/test_chapter06_em.py`:

```python
def test_run_quick_experiment_writes_artifacts(tmp_path):
    summary = c6.run_quick_experiment(
        output_root=tmp_path / "chapter06",
        n_values=[8],
        solver_names=["BiCGStab"],
        max_iter=50,
        rtol=1e-5,
        rcs_n_phi=24,
    )
    assert summary["mode"] == "quick"
    assert summary["n_values"] == [8]
    assert summary["num_solver_runs"] == 1
    assert (tmp_path / "chapter06" / "tables" / "solver_runs.csv").is_file()
    assert (tmp_path / "chapter06" / "raw" / "summary.json").is_file()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_run_quick_experiment_writes_artifacts -q -p no:cacheprovider
```

Expected: failure because `run_quick_experiment` is undefined.

- [ ] **Step 3: Implement quick orchestrator and CLI**

Add to `experiments/chapter06_em.py`:

```python
def run_quick_experiment(
    *,
    output_root: str | Path = Path("experiments") / "outputs" / "chapter06",
    n_values: list[int] | None = None,
    solver_names: list[str] | None = None,
    max_iter: int = 200,
    rtol: float = 1e-6,
    rcs_n_phi: int = 90,
) -> dict[str, Any]:
    """Run a small CPU experiment and persist core artifacts."""
    paths = ensure_output_dirs(output_root)
    n_values = list(n_values) if n_values is not None else n_series_for_mode("quick")
    solver_names = list(solver_names) if solver_names is not None else ["SIM", "BiCGStab", "TwoStep"]

    all_runs: list[SolverRun] = []
    matvec_rows: list[dict[str, Any]] = []
    rcs_diagnostics: dict[str, Any] = {}

    for N in n_values:
        case = make_sphere_case(N=N, eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)
        build_start = time.perf_counter()
        problem, operator = build_problem(case)
        operator_build_sec = time.perf_counter() - build_start
        gamma0_analysis = estimate_gamma0(problem, coarse_N=(2, 2, 2))
        all_runs.extend(
            run_solver_suite(
                problem,
                operator,
                case,
                solver_names,
                gamma0_analysis=gamma0_analysis,
                max_iter=max_iter,
                rtol=rtol,
            )
        )
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
        "mode": "quick",
        "n_values": n_values,
        "solver_names": solver_names,
        "num_solver_runs": len(all_runs),
        "output_root": str(paths["root"]),
    }
    save_json(summary, paths["raw"] / "summary.json")
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

    n_values = n_series_for_mode(args.mode)
    if args.mode == "quick":
        run_quick_experiment(
            output_root=args.output_root,
            n_values=n_values,
            max_iter=args.max_iter,
            rtol=args.rtol,
        )
        return 0

    run_quick_experiment(
        output_root=args.output_root,
        n_values=n_values,
        max_iter=args.max_iter,
        rtol=args.rtol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify smoke test passes**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_run_quick_experiment_writes_artifacts -q -p no:cacheprovider
```

Expected: `1 passed`.

- [ ] **Step 5: Verify CLI quick mode**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m experiments.chapter06_em --mode quick --output-root experiments/outputs/chapter06-smoke --max-iter 50 --rtol 1e-5
```

Expected: exit code 0 and files under `experiments/outputs/chapter06-smoke/`.

---

## Task 7: Notebook Generation

**Files:**
- Create: `notebooks/chapter-06-em.ipynb`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing test for notebook structure**

Append to `tests/test_chapter06_em.py`:

```python
def test_chapter06_notebook_exists_and_has_sections():
    import json

    path = Path("notebooks") / "chapter-06-em.ipynb"
    assert path.is_file()
    nb = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for section in ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10"]:
        assert section in markdown
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_chapter06_notebook_exists_and_has_sections -q -p no:cacheprovider
```

Expected: failure because the notebook file does not exist.

- [ ] **Step 3: Create notebook JSON**

Create `notebooks/chapter-06-em.ipynb` as a valid notebook v4 JSON with these cells:

1. Markdown title:

```markdown
# Глава 6. Численные эксперименты для объёмной электродинамической постановки

Notebook использует CPU-first harness `experiments.chapter06_em`.
```

2. Code setup:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import em3d
from experiments import chapter06_em as c6

RUN_MODE = "quick"
OUTPUT_ROOT = Path("../experiments/outputs/chapter06")
paths = c6.ensure_output_dirs(OUTPUT_ROOT)
N_VALUES = c6.n_series_for_mode(RUN_MODE)
N_VALUES
```

3. Markdown sections for `6.1` through `6.10`, each followed by one code cell that calls or prepares the relevant helper.

Use this Python script once to generate the file through `apply_patch` or a one-off local script created with `apply_patch` and then removed:

```python
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})

md("# Глава 6. Численные эксперименты для объёмной электродинамической постановки\n\nNotebook использует CPU-first harness `experiments.chapter06_em`.")
code("from pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nimport em3d\nfrom experiments import chapter06_em as c6\n\nRUN_MODE = \"quick\"\nOUTPUT_ROOT = Path(\"../experiments/outputs/chapter06\")\npaths = c6.ensure_output_dirs(OUTPUT_ROOT)\nN_VALUES = c6.n_series_for_mode(RUN_MODE)\nN_VALUES\n")
md("## 6.1. Обоснование использования электродинамической постановки\n\nИспользуется векторная постановка `(I - B eta) u = f` с dyadic Green kernel.")
code("sphere_case = c6.make_sphere_case(N=N_VALUES[0], eps_r=2.0 + 0.0j, k0a=1.0, a=0.3)\nsphere_case\n")
md("## 6.2. Постановка вычислительных экспериментов для анизотропных диэлектрических структур")
code("eps_real = np.array([[2.0, 0.1, 0.0], [0.1, 1.6, 0.0], [0.0, 0.0, 1.3]])\neps_imag = np.zeros((3, 3))\naniso_case = c6.make_anisotropic_ellipsoid_case(N=N_VALUES[0], eps_real=eps_real, eps_imag=eps_imag, k0=2.5)\naniso_case\n")
md("## 6.3. Дискретный спектр оператора и его влияние на сходимость стационарных итераций")
code("problem, operator = c6.build_problem(sphere_case)\ngamma0_analysis = c6.estimate_gamma0(problem, coarse_N=(2, 2, 2))\nfig, ax = em3d.vis.plot_gamma0_spectrum(gamma0_analysis, title=\"Coarse spectrum and gamma0 circle\")\ngamma0_analysis\n")
md("## 6.4. Применение обобщённого метода простой итерации")
code("sim_run = c6.run_solver(problem, operator, sphere_case, \"SIM\", gamma0_analysis=gamma0_analysis, max_iter=200, rtol=1e-6)\nsim_run\n")
md("## 6.5. Применение BiCGStab и сравнение с авторскими модификациями итерационных методов")
code("runs = c6.run_solver_suite(problem, operator, sphere_case, [\"SIM\", \"BiCGStab\", \"TwoStep\"], gamma0_analysis=gamma0_analysis, max_iter=200, rtol=1e-6)\nc6.save_runs_csv(runs, paths[\"tables\"] / \"solver_runs.csv\")\n[run.to_row() for run in runs]\n")
md("## 6.6. FFT-ускоренное матрично-векторное умножение для векторного оператора")
code("matvec_row = c6.benchmark_matvec(problem, operator, sphere_case, repeats=3)\nmatvec_row\n")
md("## 6.7. Расчёт распределения электрического поля внутри области неоднородности\n\n`mie_field_at` пока не используется как near-field oracle.")
code("bicg_result = em3d.BiCGStab(em3d.SolverConfig(max_iter=200, rtol=1e-6)).solve(operator, problem.wave)\nu = np.asarray(bicg_result.u)\nfig, ax = em3d.vis.plot_field_scalar_slice(u, problem.grid, plane=\"xy\", part=\"abs\", title=\"|E| slice\")\n")
md("## 6.8. Расчёт диаграммы направленности и эффективной поверхности рассеяния")
code("rcs_diag = c6.compute_mie_rcs_diagnostics(bicg_result.u, problem, a=0.3, eps_r=2.0 + 0.0j, n_phi=90)\nfig, ax = em3d.vis.plot_rcs_comparison(rcs_diag[\"phi\"], rcs_diag[\"sigma_num_norm\"], rcs_diag[\"sigma_mie_norm\"], title=\"Normalized RCS: em3d vs Mie\")\n{key: rcs_diag[key] for key in [\"shape_err\", \"scale_ratio\", \"abs_rel_err\"]}\n")
md("## 6.9. Сравнение вычислительной эффективности методов")
code("summary = c6.run_quick_experiment(output_root=OUTPUT_ROOT, n_values=N_VALUES[:1], max_iter=100, rtol=1e-6, rcs_n_phi=90)\nsummary\n")
md("## 6.10. Выводы по главе\n\nВыводы формируются после запуска `full` режима и анализа сохранённых таблиц.")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/chapter-06-em.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Verify notebook structure test passes**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_chapter06_notebook_exists_and_has_sections -q -p no:cacheprovider
```

Expected: `1 passed`.

---

## Task 8: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Run: full tests

- [ ] **Step 1: Add pytest marker if slow tests are introduced**

If any test is decorated with `@pytest.mark.slow`, update `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires CUDA-capable device and cupy",
    "slow: longer CPU experiments or notebook smoke tests",
]
```

If no `slow` marker is used, leave `pyproject.toml` unchanged.

- [ ] **Step 2: Add README pointer**

Add a short section to `README.md` after the visualization/gamma0 examples:

```markdown
### Эксперименты главы 6

Для диссертационных численных экспериментов добавлен CPU-first harness:

- `experiments/chapter06_em.py` — запуск cases, solver-suite, gamma0, RCS diagnostics, сохранение таблиц;
- `notebooks/chapter-06-em.ipynb` — narrative notebook по разделам 6.1–6.10.

Быстрый smoke-запуск:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m experiments.chapter06_em --mode quick --output-root experiments/outputs/chapter06-smoke --max-iter 50 --rtol 1e-5
```
```

- [ ] **Step 3: Run helper tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
```

Expected: all helper tests pass.

- [ ] **Step 4: Run full non-save test suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest -q -p no:cacheprovider -k "not saves_file"
```

Expected: all tests pass, preserving the existing skipped/xfail counts.

- [ ] **Step 5: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected: exit code 0. Git may print LF/CRLF warnings on Windows; those are not diff-check failures.

- [ ] **Step 6: Inspect working tree**

Run:

```powershell
git status --short
```

Expected: new/modified files are limited to the experiment harness, notebook, tests, README, and plan/spec files, plus any already-existing unrelated dirty files from earlier work.

---

## Self-Review Checklist

- Spec coverage:
  - CPU-only quick/full modes: Task 1 and Task 6.
  - `N_SERIES = [8, 16, 24, 32, 40, 48, 56, 64]`: Task 1.
  - Helper module outside package API: File Structure and Task 1.
  - Sphere Mie normalized RCS diagnostics: Task 5 and notebook Task 7.
  - Anisotropic ellipsoid case: Task 2 and notebook Task 7.
  - Gamma0 coarse spectrum: Task 4 and notebook Task 7.
  - Solver comparison: Task 4 and notebook Task 7.
  - FFT matvec timing: Task 5 and notebook Task 7.
  - Field visualizations without Mie near-field oracle: Task 7.
  - Saved CSV/JSON outputs: Task 5 and Task 6.
- Placeholder scan:
  - No unfinished placeholder markers are used.
  - Code snippets define all referenced helper functions.
- Type consistency:
  - `ExperimentCase`, `SolverRun`, `build_problem`, `run_solver`, and `run_solver_suite` signatures are consistent across tasks.
  - `gamma0_analysis` is required only for `SIM`.
  - Notebook calls match helper function signatures.
