# Chapter 06 EM Experiments Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Расширить CPU-first experimental harness и notebook главы 6 для материалов, анизотропных кристаллов, `gamma0` scans, `N=64` convergence studies, FFT-vs-dense matvec и RCS-серий по `k0a`.

**Architecture:** Новая логика разделяется по модулям `experiments/materials.py`, `experiments/cases.py`, `experiments/scans.py`, `experiments/plots.py`. Существующий `experiments/chapter06_em.py` остаётся фасадом и re-export слоем для обратной совместимости notebook/tests. Дорогие `N=64` расчёты доступны из notebook/CLI, но unit-тесты используют малые сетки.

**Tech Stack:** Python, NumPy, pytest, matplotlib через `em3d.vis`, локальный `em3d`, CPU backend.

---

## File Structure

- Create: `experiments/materials.py`
  - `MaterialSpec`, Drude model, tensor rotation, conversion to `(eps_real, eps_imag)`.
- Create: `experiments/cases.py`
  - `ExperimentCase`, `SolverRun`, case builders: sphere, ellipsoid, uniaxial crystal, layered box.
- Create: `experiments/scans.py`
  - `build_problem`, `estimate_gamma0`, solver runners, gamma0/convergence/matvec/RCS scans.
- Create: `experiments/plots.py`
  - grouped plotting helpers for three slices and scan curves.
- Modify: `experiments/chapter06_em.py`
  - becomes a compatibility facade importing the public helper API from new modules.
- Modify: `tests/test_chapter06_em.py`
  - keep existing coverage and add focused tests for new APIs.
- Modify: `notebooks/chapter-06-em.ipynb`
  - update narrative and code cells for sections 6.1-6.8.
- Modify: `README.md`
  - add note that chapter 6 harness is modular.

Do not add `experiments` to `tool.setuptools.packages.find`; it remains research code outside public `em3d` package API.

---

## Task 1: Materials Module

**Files:**
- Create: `experiments/materials.py`
- Modify: `tests/test_chapter06_em.py`
- Modify: `experiments/chapter06_em.py`

- [ ] **Step 1: Write failing tests for materials**

Append to `tests/test_chapter06_em.py`:

```python
def test_material_spec_isotropic_to_eps():
    material = c6.MaterialSpec.isotropic(2.5)
    eps_real, eps_imag = c6.material_eps(material, k0=3.0)
    assert eps_real == pytest.approx(2.5)
    assert eps_imag == pytest.approx(0.0)


def test_material_spec_anisotropic_lossy_to_eps():
    eps_real_in = np.diag([2.0, 1.5, 1.2])
    eps_imag_in = np.diag([0.1, 0.05, 0.02])
    material = c6.MaterialSpec.anisotropic_lossy(eps_real_in, eps_imag_in)
    eps_real, eps_imag = c6.material_eps(material, k0=3.0)
    np.testing.assert_allclose(eps_real, eps_real_in)
    np.testing.assert_allclose(eps_imag, eps_imag_in)


def test_material_spec_drude_matches_formula():
    material = c6.MaterialSpec.plasma_drude(eps_inf=1.0, omega_p=2.0, gamma=0.1)
    eps_real, eps_imag = c6.material_eps(material, k0=2.0)
    expected = 1.0 - 2.0**2 / (2.0**2 + 1j * 0.1 * 2.0)
    assert eps_real == pytest.approx(expected.real)
    assert eps_imag == pytest.approx(expected.imag)


def test_rotate_tensor_preserves_eigenvalues():
    eps = np.diag([2.0, 1.5, 1.2])
    theta = np.pi / 4.0
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = c6.rotate_tensor(eps, R)
    np.testing.assert_allclose(np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(eps), atol=1e-12)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_material_spec_isotropic_to_eps tests/test_chapter06_em.py::test_material_spec_anisotropic_lossy_to_eps tests/test_chapter06_em.py::test_material_spec_drude_matches_formula tests/test_chapter06_em.py::test_rotate_tensor_preserves_eigenvalues -q -p no:cacheprovider
```

Expected: failures because `MaterialSpec`, `material_eps`, or `rotate_tensor` are undefined.

- [ ] **Step 3: Implement `experiments/materials.py`**

Create:

```python
"""Material models for chapter 6 electrodynamic experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaterialSpec:
    kind: str
    eps_real: Any = None
    eps_imag: Any = 0.0
    eps_inf: float | None = None
    omega_p: float | None = None
    gamma: float | None = None
    orientation: Any = None

    @classmethod
    def isotropic(cls, eps_r: float | complex) -> "MaterialSpec":
        eps = complex(eps_r)
        return cls(kind="isotropic", eps_real=float(eps.real), eps_imag=float(eps.imag))

    @classmethod
    def anisotropic(cls, eps_real, *, orientation=None) -> "MaterialSpec":
        return cls(kind="anisotropic", eps_real=np.array(eps_real, dtype=np.float64, copy=True), eps_imag=np.zeros((3, 3)), orientation=orientation)

    @classmethod
    def anisotropic_lossy(cls, eps_real, eps_imag, *, orientation=None) -> "MaterialSpec":
        return cls(kind="anisotropic_lossy", eps_real=np.array(eps_real, dtype=np.float64, copy=True), eps_imag=np.array(eps_imag, dtype=np.float64, copy=True), orientation=orientation)

    @classmethod
    def plasma_drude(cls, *, eps_inf: float = 1.0, omega_p: float = 1.0, gamma: float = 0.0) -> "MaterialSpec":
        return cls(kind="plasma_drude", eps_inf=float(eps_inf), omega_p=float(omega_p), gamma=float(gamma))


def rotate_tensor(tensor, orientation):
    tensor = np.asarray(tensor, dtype=np.float64)
    R = np.asarray(orientation, dtype=np.float64)
    if tensor.shape != (3, 3):
        raise ValueError(f"tensor must have shape (3,3), got {tensor.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"orientation must have shape (3,3), got {R.shape}")
    return R @ tensor @ R.T


def _apply_orientation(eps_real, eps_imag, orientation):
    if orientation is None:
        return eps_real, eps_imag
    return rotate_tensor(eps_real, orientation), rotate_tensor(eps_imag, orientation)


def material_eps(material: MaterialSpec, *, k0: float) -> tuple:
    if material.kind == "plasma_drude":
        if k0 <= 0:
            raise ValueError(f"k0 must be positive for Drude model, got {k0}")
        eps = material.eps_inf - material.omega_p**2 / (k0**2 + 1j * material.gamma * k0)
        return float(np.real(eps)), float(np.imag(eps))
    if material.kind in {"isotropic", "anisotropic", "anisotropic_lossy"}:
        return _apply_orientation(material.eps_real, material.eps_imag, material.orientation)
    raise ValueError(f"unknown material kind {material.kind!r}")
```

- [ ] **Step 4: Re-export from facade**

Add to `experiments/chapter06_em.py`:

```python
from .materials import MaterialSpec, material_eps, rotate_tensor
```

- [ ] **Step 5: Run GREEN**

Run the RED command again. Expected: `4 passed`.

---

## Task 2: Cases Module and Layered Geometry

**Files:**
- Create: `experiments/cases.py`
- Modify: `experiments/chapter06_em.py`
- Modify: `experiments/scans.py` if created in Task 3, otherwise keep `build_problem` in facade until Task 3
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for crystal and layered cases**

Append:

```python
def test_make_uniaxial_crystal_ellipsoid_case_builds_tensor():
    case = c6.make_uniaxial_crystal_ellipsoid_case(N=8, eps_o=2.2, eps_e=1.4, k0=3.0)
    assert case.geometry == "ellipsoid"
    assert case.material.kind == "anisotropic"
    eps_real, eps_imag = c6.material_eps(case.material, k0=case.k0)
    np.testing.assert_allclose(eps_real, np.diag([2.2, 2.2, 1.4]))
    np.testing.assert_allclose(eps_imag, np.zeros((3, 3)))


def test_make_layered_box_case_builds_three_layers():
    layers = [
        c6.LayerSpec(z_min=-0.5, z_max=-1.0 / 6.0, material=c6.MaterialSpec.anisotropic_lossy(np.diag([1.5, 1.4, 1.3]), np.diag([0.01, 0.01, 0.01]))),
        c6.LayerSpec(z_min=-1.0 / 6.0, z_max=1.0 / 6.0, material=c6.MaterialSpec.anisotropic_lossy(np.diag([2.0, 1.8, 1.6]), np.diag([0.02, 0.02, 0.02]))),
        c6.LayerSpec(z_min=1.0 / 6.0, z_max=0.5, material=c6.MaterialSpec.anisotropic_lossy(np.diag([2.5, 2.2, 1.9]), np.diag([0.03, 0.03, 0.03]))),
    ]
    case = c6.make_layered_box_case(N=6, k0=4.0, layers=layers)
    problem, _ = c6.build_problem(case)
    assert case.geometry == "layered_box"
    assert problem.eps_tensor.shape == (3, 3, 6, 6, 6)
    assert np.count_nonzero(np.asarray(problem.eps_tensor[0, 0])) > 0
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_make_uniaxial_crystal_ellipsoid_case_builds_tensor tests/test_chapter06_em.py::test_make_layered_box_case_builds_three_layers -q -p no:cacheprovider
```

Expected: undefined symbols.

- [ ] **Step 3: Create `experiments/cases.py`**

Move or duplicate compatible definitions from `chapter06_em.py`:

```python
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .materials import MaterialSpec


@dataclass(frozen=True)
class LayerSpec:
    z_min: float
    z_max: float
    material: MaterialSpec


@dataclass(frozen=True)
class ExperimentCase:
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
    material: MaterialSpec | None = None
    layers: tuple[LayerSpec, ...] = ()

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
```

Implement `_grid_shape`, `_number_token`, existing `make_sphere_case`, existing `make_anisotropic_ellipsoid_case`, plus:

```python
def make_uniaxial_crystal_ellipsoid_case(*, N, eps_o: float, eps_e: float, k0: float, radius=(0.35, 0.25, 0.2), orientation=None, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), wave_orient=(0.0, 0.0, 1.0), wave_amplitude=(1.0, 0.0, 0.0)) -> ExperimentCase:
    eps = np.diag([eps_o, eps_o, eps_e])
    material = MaterialSpec.anisotropic(eps, orientation=orientation)
    return ExperimentCase(
        name=f"uniaxial_crystal_N{_grid_shape(N)[0]}",
        N=_grid_shape(N),
        L=L,
        k0=float(k0),
        geometry="ellipsoid",
        eps_real=eps,
        eps_imag=np.zeros((3, 3)),
        center=center,
        radius=radius,
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        material=material,
    )


def make_layered_box_case(*, N, k0: float, layers, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), wave_orient=(0.0, 0.0, 1.0), wave_amplitude=(1.0, 0.0, 0.0)) -> ExperimentCase:
    shape = _grid_shape(N)
    return ExperimentCase(
        name=f"layered_box_N{shape[0]}_k{_number_token(k0)}",
        N=shape,
        L=L,
        k0=float(k0),
        geometry="layered_box",
        eps_real=0.0,
        eps_imag=0.0,
        center=center,
        radius=(0.0, 0.0, 0.0),
        wave_orient=wave_orient,
        wave_amplitude=wave_amplitude,
        layers=tuple(layers),
    )
```

- [ ] **Step 4: Update facade imports**

In `experiments/chapter06_em.py`, import from `cases.py` and remove duplicate class/builder definitions only after tests pass:

```python
from .cases import (
    ExperimentCase,
    LayerSpec,
    SolverRun,
    make_anisotropic_ellipsoid_case,
    make_layered_box_case,
    make_sphere_case,
    make_uniaxial_crystal_ellipsoid_case,
)
```

- [ ] **Step 5: Extend `build_problem` for `layered_box`**

If `build_problem` still lives in `chapter06_em.py`, add:

```python
elif case.geometry == "layered_box":
    eps_tensor = _layered_box_refraction(grid, case.layers, case.k0)
```

Implement:

```python
def _layered_box_refraction(grid, layers, k0):
    be = grid.backend
    xp = be.xp
    _, _, Z = grid.coords()
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for layer in layers:
        eps_real, eps_imag = material_eps(layer.material, k0=k0)
        eta_mat = np.asarray(eps_real, dtype=np.float64) - np.eye(3) + 1j * np.asarray(eps_imag, dtype=np.float64)
        mask = (Z >= layer.z_min) & (Z < layer.z_max)
        for i in range(3):
            for j in range(3):
                out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out
```

Use `<=` for the last layer if gaps appear in tests.

- [ ] **Step 6: Run GREEN**

Run the RED command again. Expected: `2 passed`.

---

## Task 3: Scans Module and Compatibility Facade

**Files:**
- Create: `experiments/scans.py`
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for scan rows**

Append:

```python
def test_scan_gamma0_returns_rows_for_small_subset():
    def factory(coarse_N, k0):
        return c6.make_anisotropic_ellipsoid_case(
            N=8,
            eps_real=np.diag([2.0, 1.6, 1.3]),
            eps_imag=np.zeros((3, 3)),
            k0=k0,
        )

    rows = c6.scan_gamma0(factory, coarse_values=[2], k_values=[1, 2], scenario="test-aniso")
    assert len(rows) == 2
    assert rows[0]["scenario"] == "test-aniso"
    assert rows[0]["coarse_N"] >= 2
    assert rows[0]["k0"] in (1.0, 2.0)
    assert "rho" in rows[0]


def test_anisotropic_ellipsoid_gamma0_factory_uses_k0():
    case = c6.make_anisotropic_gamma0_case(coarse_N=4, k0=7.0)
    assert case.k0 == pytest.approx(7.0)
    assert case.geometry == "ellipsoid"
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_scan_gamma0_returns_rows_for_small_subset tests/test_chapter06_em.py::test_anisotropic_ellipsoid_gamma0_factory_uses_k0 -q -p no:cacheprovider
```

- [ ] **Step 3: Create `experiments/scans.py`**

Move compatible existing functions from `chapter06_em.py`: `build_problem`, `estimate_gamma0`, `_solver_instance`, `run_solver`, `run_solver_suite`, `save_runs_csv`, `save_json`, `benchmark_matvec`, `compute_mie_rcs_diagnostics`, `run_quick_experiment`, `main`.

Add:

```python
def _analysis_to_row(analysis, *, scenario: str, coarse_N: int, k0: float, status: str = "ok", error: str = ""):
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


def scan_gamma0(case_factory, *, coarse_values=(2, 3, 4, 5, 6), k_values=range(1, 11), scenario="gamma0"):
    rows = []
    for coarse_N in coarse_values:
        for k0 in k_values:
            case = case_factory(coarse_N=coarse_N, k0=float(k0))
            try:
                problem, _ = build_problem(case)
                analysis = estimate_gamma0(problem, coarse_N=(int(coarse_N),) * 3)
                rows.append(_analysis_to_row(analysis, scenario=scenario, coarse_N=coarse_N, k0=float(k0)))
            except Exception as exc:
                rows.append({
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
                })
    return rows
```

Add factories:

```python
def make_isotropic_gamma0_case(*, coarse_N: int, k0: float):
    return make_anisotropic_ellipsoid_case(N=8, eps_real=2.0, eps_imag=0.0, k0=k0)


def make_anisotropic_gamma0_case(*, coarse_N: int, k0: float):
    return make_anisotropic_ellipsoid_case(N=8, eps_real=np.diag([2.0, 1.6, 1.3]), eps_imag=np.zeros((3, 3)), k0=k0)


def make_layered_gamma0_case(*, coarse_N: int, k0: float):
    scale = float(k0) / 10.0
    layers = [
        LayerSpec(-0.5, -1.0 / 6.0, MaterialSpec.anisotropic_lossy(np.diag([1.5, 1.4, 1.3]) * (1 + scale), np.diag([0.01, 0.01, 0.01]) * scale)),
        LayerSpec(-1.0 / 6.0, 1.0 / 6.0, MaterialSpec.anisotropic_lossy(np.diag([2.0, 1.8, 1.6]) * (1 + scale), np.diag([0.02, 0.02, 0.02]) * scale)),
        LayerSpec(1.0 / 6.0, 0.5, MaterialSpec.anisotropic_lossy(np.diag([2.5, 2.2, 1.9]) * (1 + scale), np.diag([0.03, 0.03, 0.03]) * scale)),
    ]
    return make_layered_box_case(N=8, k0=k0, layers=layers)
```

- [ ] **Step 4: Re-export scans from facade**

In `chapter06_em.py`, import public functions from `scans.py`.

- [ ] **Step 5: Run GREEN and compatibility tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
```

Expected: all existing helper tests plus new tests pass.

---

## Task 4: Convergence and Solver Comparison Scans

**Files:**
- Modify: `experiments/scans.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests for convergence helpers**

Append:

```python
def test_scan_sim_convergence_by_gamma0_smoke():
    case = c6.make_anisotropic_ellipsoid_case(
        N=8,
        eps_real=np.diag([1.3, 1.2, 1.1]),
        eps_imag=np.zeros((3, 3)),
        k0=1.0,
    )
    rows = c6.scan_sim_convergence_by_gamma0(case, coarse_values=[2], max_iter=10, rtol=1e-4)
    assert len(rows) == 1
    assert rows[0]["solver_name"] == "SIM"
    assert rows[0]["case_name"] == case.name


def test_run_solver_comparison_returns_reference_solution():
    case = c6.make_anisotropic_ellipsoid_case(
        N=8,
        eps_real=np.diag([1.3, 1.2, 1.1]),
        eps_imag=np.zeros((3, 3)),
        k0=1.0,
    )
    result = c6.run_solver_comparison(case, sim_coarse_N=2, max_iter=10, rtol=1e-4)
    assert [run.solver_name for run in result["runs"]] == ["SIM", "BiCGStab", "TwoStep"]
    assert result["reference_u"].shape == (3, 8, 8, 8)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_scan_sim_convergence_by_gamma0_smoke tests/test_chapter06_em.py::test_run_solver_comparison_returns_reference_solution -q -p no:cacheprovider
```

- [ ] **Step 3: Implement scan functions**

In `experiments/scans.py`:

```python
def scan_sim_convergence_by_gamma0(case, *, coarse_values=(2, 3, 4, 5, 6), max_iter=500, rtol=1e-6):
    problem, operator = build_problem(case)
    rows = []
    for coarse_N in coarse_values:
        analysis = estimate_gamma0(problem, coarse_N=(int(coarse_N),) * 3)
        run = run_solver(problem, operator, case, "SIM", gamma0_analysis=analysis, max_iter=max_iter, rtol=rtol)
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
    return rows


def run_solver_comparison(case, *, sim_coarse_N=6, max_iter=500, rtol=1e-6):
    problem, operator = build_problem(case)
    gamma0_analysis = estimate_gamma0(problem, coarse_N=(int(sim_coarse_N),) * 3)
    runs = run_solver_suite(
        problem,
        operator,
        case,
        ["SIM", "BiCGStab", "TwoStep"],
        gamma0_analysis=gamma0_analysis,
        max_iter=max_iter,
        rtol=rtol,
    )
    bicg = em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
    if bicg.converged:
        reference_u = bicg.u
        reference_solver = "BiCGStab"
    else:
        two = em3d.TwoStep(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
        reference_u = two.u
        reference_solver = "TwoStep"
    return {
        "case": case,
        "problem": problem,
        "operator": operator,
        "gamma0_analysis": gamma0_analysis,
        "runs": runs,
        "reference_u": reference_u,
        "reference_solver": reference_solver,
    }
```

- [ ] **Step 4: Run GREEN**

Run the RED command again. Expected: `2 passed`.

---

## Task 5: FFT vs Dense Matvec

**Files:**
- Modify: `experiments/scans.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_benchmark_fft_vs_dense_matches_dense_on_small_grid():
    def factory(N):
        return c6.make_anisotropic_ellipsoid_case(
            N=N,
            eps_real=np.diag([1.2, 1.1, 1.05]),
            eps_imag=np.zeros((3, 3)),
            k0=1.0,
        )

    rows = c6.benchmark_fft_vs_dense(factory, n_values=[2], repeats=1)
    assert len(rows) == 1
    assert rows[0]["N"] == "2x2x2"
    assert rows[0]["relative_error"] < 1e-10
    assert rows[0]["fft_avg_sec"] >= 0.0
    assert rows[0]["dense_avg_sec"] >= 0.0
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_benchmark_fft_vs_dense_matches_dense_on_small_grid -q -p no:cacheprovider
```

- [ ] **Step 3: Implement `benchmark_fft_vs_dense`**

In `experiments/scans.py`:

```python
def _flatten_field(u):
    return np.asarray(u).reshape(-1)


def _unflatten_field(v, N):
    return np.asarray(v).reshape((3,) + tuple(N))


def benchmark_fft_vs_dense(case_factory, *, n_values=(2, 3, 4, 5, 6), repeats=3):
    rows = []
    for N in n_values:
        case = case_factory(N)
        problem, operator = build_problem(case)
        dense = operator.to_dense()
        u = np.asarray(problem.wave)
        fft_durations = []
        dense_durations = []
        fft_result = None
        dense_result = None
        for _ in range(repeats):
            start = time.perf_counter()
            fft_result = np.asarray(operator.matvec(u))
            fft_durations.append(time.perf_counter() - start)
            start = time.perf_counter()
            dense_result = _unflatten_field(dense @ _flatten_field(u), case.N)
            dense_durations.append(time.perf_counter() - start)
        den = float(np.linalg.norm(dense_result))
        rel = float(np.linalg.norm(fft_result - dense_result) / den) if den > 0 else 0.0
        rows.append({
            "case_name": case.name,
            "N": "x".join(str(n) for n in case.N),
            "dof": case.dof,
            "relative_error": rel,
            "fft_avg_sec": float(np.mean(fft_durations)),
            "dense_avg_sec": float(np.mean(dense_durations)),
            "repeats": int(repeats),
        })
    return rows
```

- [ ] **Step 4: Run GREEN**

Run the RED command again. Expected: `1 passed`.

---

## Task 6: RCS Scan by `k0a`

**Files:**
- Modify: `experiments/scans.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_scan_mie_rcs_by_k0a_smoke():
    rows = c6.scan_mie_rcs_by_k0a(N=8, a=0.25, eps_r=1.2, k0a_values=[0.25], n_phi=24, max_iter=50, rtol=1e-5)
    assert len(rows) == 1
    assert rows[0]["k0a"] == pytest.approx(0.25)
    assert rows[0]["shape_err"] >= 0.0
    assert rows[0]["scale_ratio"] >= 0.0
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_scan_mie_rcs_by_k0a_smoke -q -p no:cacheprovider
```

- [ ] **Step 3: Implement `scan_mie_rcs_by_k0a`**

In `experiments/scans.py`:

```python
def scan_mie_rcs_by_k0a(*, N=32, a=0.3, eps_r=2.0, k0a_values=(0.25, 0.5, 1.0, 1.5, 2.0), n_phi=90, max_iter=500, rtol=1e-8):
    rows = []
    for k0a in k0a_values:
        case = make_sphere_case(N=N, eps_r=complex(eps_r), k0a=float(k0a), a=a)
        problem, operator = build_problem(case)
        result = em3d.BiCGStab(em3d.SolverConfig(max_iter=max_iter, rtol=rtol)).solve(operator, problem.wave)
        diagnostics = compute_mie_rcs_diagnostics(result.u, problem, a=a, eps_r=complex(eps_r), n_phi=n_phi)
        rows.append({
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
        })
    return rows
```

- [ ] **Step 4: Run GREEN**

Run the RED command again. Expected: `1 passed`.

---

## Task 7: Plot Helpers

**Files:**
- Create: `experiments/plots.py`
- Modify: `experiments/chapter06_em.py`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing test for plot helper with monkeypatched vis**

Append:

```python
def test_plot_three_field_slices_calls_scalar_slice(monkeypatch):
    calls = []

    def fake_plot(u, grid, **kwargs):
        calls.append(kwargs)
        return object(), object()

    monkeypatch.setattr(em3d.vis, "plot_field_scalar_slice", fake_plot)
    case = c6.make_sphere_case(N=4, eps_r=1.2, k0a=0.5, a=0.25)
    problem, _ = c6.build_problem(case)
    figs = c6.plot_three_field_slices(problem.wave, problem.grid, part="abs", component=None)
    assert len(figs) == 3
    assert [call["plane"] for call in calls] == ["xy", "xz", "yz"]
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_plot_three_field_slices_calls_scalar_slice -q -p no:cacheprovider
```

- [ ] **Step 3: Implement `experiments/plots.py`**

Create:

```python
"""Notebook-friendly plot groups for chapter 6 experiments."""
from __future__ import annotations

from pathlib import Path

import em3d


def plot_three_field_slices(u, grid, *, part="abs", component=None, output_dir=None, prefix="field"):
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    figures = []
    for plane in ("xy", "xz", "yz"):
        filename = None
        if output_path is not None:
            suffix = "norm" if component is None else f"component{component}"
            filename = str(output_path / f"{prefix}_{suffix}_{part}_{plane}.png")
        figures.append(
            em3d.vis.plot_field_scalar_slice(
                u,
                grid,
                plane=plane,
                part=part,
                component=component,
                title=f"{part} field slice {plane}",
                filename=filename,
            )
        )
    return figures
```

- [ ] **Step 4: Re-export from facade**

Add:

```python
from .plots import plot_three_field_slices
```

- [ ] **Step 5: Run GREEN**

Run the RED command again. Expected: `1 passed`.

---

## Task 8: Notebook Update

**Files:**
- Modify: `notebooks/chapter-06-em.ipynb`
- Modify: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing structure test for new notebook calls**

Append:

```python
def test_chapter06_notebook_mentions_extended_experiments():
    import json

    path = Path("notebooks") / "chapter-06-em.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    for token in [
        "MaterialSpec.plasma_drude",
        "make_uniaxial_crystal_ellipsoid_case",
        "scan_gamma0",
        "scan_sim_convergence_by_gamma0",
        "run_solver_comparison",
        "benchmark_fft_vs_dense",
        "plot_three_field_slices",
        "scan_mie_rcs_by_k0a",
    ]:
        assert token in source
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_chapter06_notebook_mentions_extended_experiments -q -p no:cacheprovider
```

- [ ] **Step 3: Update notebook cells**

Edit `notebooks/chapter-06-em.ipynb` as valid JSON. Keep the existing setup cell and update sections:

- 6.1 includes material table and:

```python
materials = [
    c6.MaterialSpec.isotropic(2.0),
    c6.MaterialSpec.anisotropic(np.diag([2.0, 1.6, 1.3])),
    c6.MaterialSpec.anisotropic_lossy(np.diag([2.0, 1.6, 1.3]), np.diag([0.05, 0.03, 0.02])),
    c6.MaterialSpec.plasma_drude(eps_inf=1.0, omega_p=2.0, gamma=0.1),
]
[(m.kind, c6.material_eps(m, k0=2.0)) for m in materials]
```

- 6.2 includes:

```python
crystal_case = c6.make_uniaxial_crystal_ellipsoid_case(N=32, eps_o=2.2, eps_e=1.4, k0=3.0)
crystal_problem, crystal_operator = c6.build_problem(crystal_case)
crystal_result = em3d.BiCGStab(em3d.SolverConfig(max_iter=500, rtol=1e-6)).solve(crystal_operator, crystal_problem.wave)
crystal_u = np.asarray(crystal_result.u)
c6.plot_three_field_slices(crystal_u, crystal_problem.grid, part="abs")
```

- 6.3 includes the three `scan_gamma0(...)` scenarios.
- 6.4 includes `scan_sim_convergence_by_gamma0(...)` with `N=64` in the visible code, but guarded by `RUN_FULL = False`:

```python
RUN_FULL = False
if RUN_FULL:
    n64_case = c6.make_anisotropic_ellipsoid_case(N=64, eps_real=np.diag([2.0, 1.6, 1.3]), eps_imag=np.zeros((3, 3)), k0=10.0)
    sim_scan = c6.scan_sim_convergence_by_gamma0(n64_case, coarse_values=[2, 3, 4, 5, 6])
else:
    sim_scan = []
```

- 6.5 includes `run_solver_comparison(...)` similarly guarded.
- 6.6 includes `benchmark_fft_vs_dense(...)`.
- 6.7 includes component plots:

```python
for component in (0, 1, 2):
    c6.plot_three_field_slices(crystal_u, crystal_problem.grid, part="abs", component=component)
```

- 6.8 includes `scan_mie_rcs_by_k0a(...)`.

- [ ] **Step 4: Run GREEN**

Run the RED command again. Expected: `1 passed`.

---

## Task 9: README and Final Verification

**Files:**
- Modify: `README.md`
- Run verification commands

- [ ] **Step 1: Update README section**

In the “Эксперименты главы 6” section, add:

```markdown
Расширенный harness разделён на модули:

- `experiments/materials.py` — изотропные, анизотропные, поглощающие и Drude-плазменные материалы;
- `experiments/cases.py` — сферы, эллипсоиды, одноосный кристалл и слоистый параллелепипед;
- `experiments/scans.py` — `gamma0`, solver convergence, FFT-vs-dense и RCS scans;
- `experiments/plots.py` — групповые графики срезов поля для notebook.
```

- [ ] **Step 2: Run helper tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
```

Expected: all helper tests pass.

- [ ] **Step 3: Run full non-save test suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest -q -p no:cacheprovider -k "not saves_file"
```

Expected: full non-save suite passes with existing skip/xfail counts.

- [ ] **Step 4: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected: exit code 0. LF/CRLF warnings on Windows are acceptable if there are no whitespace errors.

- [ ] **Step 5: Inspect working tree**

Run:

```powershell
git status --short
```

Expected: new/modified files include experiment modules, notebook, tests, README, spec/plan files, plus already-existing unrelated dirty files from earlier work. Do not stage or commit unrelated changes.

---

## Self-Review Checklist

- Spec coverage:
  - Material types including Drude plasma: Task 1.
  - Uniaxial crystal and `N=32` notebook slices: Tasks 2 and 8.
  - `gamma0` dynamics for three scenarios: Task 3 and Task 8.
  - SIM convergence at `N=64` with coarse grids: Task 4 and Task 8.
  - Three-method comparison at `N=64`: Task 4 and Task 8.
  - FFT-vs-dense matvec: Task 5 and Task 8.
  - More field plots: Task 7 and Task 8.
  - RCS scan by `k0a`: Task 6 and Task 8.
- Placeholder scan:
  - No `TODO`, `TBD`, or unspecified implementation steps.
  - Expensive `N=64` notebook blocks are explicitly guarded by `RUN_FULL = False`.
- Type consistency:
  - `ExperimentCase.material` and `ExperimentCase.layers` are introduced before use.
  - `LayerSpec` is used by layered box builder and `build_problem`.
  - `chapter06_em.py` remains the public experiment facade for notebook/tests.

---

## Addendum: Logging, Extended 6.6/6.8, and 6.10 Crash-Test

**Goal:** добавить внешний экспериментальный лог, расширить FFT-vs-dense benchmark до `N=2..10`, расширить Mie/RCS scan до `k0a=[0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0]` при `a=0.5`, и добавить выключенный по умолчанию crash-test TwoStep на слоистой структуре `N=128`.

**Architecture:** логирование выносится в `experiments/experiment_logging.py`, scan-функции получают необязательный `logger=None`, а notebook управляет дорогим `N=128` блоком через `RUN_CRASH_TEST = False`. Графики остаются в `experiments/plots.py`.

### Task A1: Experiment Logger

**Files:**
- Create: `experiments/experiment_logging.py`
- Modify: `experiments/chapter06_em.py`
- Test: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests**

Add tests that instantiate `ExperimentLogger`, write two events, and assert that both `raw/<name>.jsonl` and `raw/<name>.log` exist and contain the event names.

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py::test_experiment_logger_writes_jsonl_and_text -q -p no:cacheprovider
```

Expected: fail because `ExperimentLogger` is not exported.

- [ ] **Step 3: Implement logger**

Create `ExperimentLogger` with method `event(event: str, **payload)`. It must create `raw/`, append JSON lines with UTC timestamp, and append one compact human-readable line to `.log`. NumPy scalars, arrays, complex values and paths must serialize.

- [ ] **Step 4: Verify GREEN**

Run the RED command again. Expected: `1 passed`.

### Task A2: Logging Hooks and Constants

**Files:**
- Modify: `experiments/scans.py`
- Modify: `experiments/chapter06_em.py`
- Test: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
assert c6.FFT_DENSE_N_VALUES == [2, 3, 4, 5, 6, 7, 8, 9, 10]
assert c6.RCS_K0A_VALUES == [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0]
assert c6.RCS_DEFAULT_RADIUS == pytest.approx(0.5)
```

Add one smoke test that passes a logger to `benchmark_fft_vs_dense(..., n_values=[2], repeats=1)` and verifies that `benchmark_done` appears in `.jsonl`.

- [ ] **Step 2: Verify RED**

Run the new tests. Expected: fail because constants and logger hooks are absent.

- [ ] **Step 3: Implement constants and hooks**

Add constants in `scans.py`, update default arguments, and add `_log(logger, event, **payload)` helper. Log start/finish events in public scan functions without changing return row schemas.

- [ ] **Step 4: Verify GREEN**

Run the new tests. Expected: pass.

### Task A3: Timing and RCS Plot Helpers

**Files:**
- Modify: `experiments/plots.py`
- Modify: `experiments/chapter06_em.py`
- Test: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing tests**

Add monkeypatched tests for `plot_fft_vs_dense_timing` and `plot_rcs_scan`. The tests should avoid expensive solvers by passing small synthetic rows and monkeypatching `em3d.vis.plot_rcs_comparison` / `plot_rcs_comparison_polar`.

- [ ] **Step 2: Verify RED**

Run the new tests. Expected: fail because plot helpers are absent.

- [ ] **Step 3: Implement helpers**

`plot_fft_vs_dense_timing(rows, output_dir=None)` builds one matplotlib line plot. `plot_rcs_scan(rows, output_dir=None)` builds Cartesian and polar plots for each row plus a compact summary plot of `shape_err` and `scale_ratio`.

- [ ] **Step 4: Verify GREEN**

Run the new tests. Expected: pass.

### Task A4: Notebook, README, Wiki

**Files:**
- Modify: `notebooks/chapter-06-em.ipynb`
- Modify: `README.md`
- Create/Modify: `wiki/code/chapter06-em-experiments.md`
- Modify: `wiki/index.md`, `wiki/log.md` if `wiki/` is populated in the current workspace
- Test: `tests/test_chapter06_em.py`

- [ ] **Step 1: Write failing notebook structure test**

Add a test that verifies notebook source contains `ExperimentLogger`, `plot_fft_vs_dense_timing`, `plot_rcs_scan`, `RUN_CRASH_TEST`, `N=128`, `TwoStep`, and `plot_field_vector_slice`.

- [ ] **Step 2: Verify RED**

Run the new structure test. Expected: fail until notebook is updated.

- [ ] **Step 3: Update notebook and docs**

Add centralized logger setup, update 6.6, update 6.8, and replace 6.10 markdown-only conclusion with guarded executable crash-test code. Update README and wiki code note.

- [ ] **Step 4: Final verification**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_chapter06_em.py -q -p no:cacheprovider
py -m pytest -q -p no:cacheprovider -k "not saves_file"
git diff --check
```
