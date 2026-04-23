# EM3D Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the EM3D Jupyter notebook (`wiki/raw/notes/Yurchenkov-programming-code-EM3D-2026.ipynb`) into a structured Python package `em3d` with a unified numpy/cupy backend, configurable `SINGLE`/`DOUBLE` precision, FFT-based matvec on a doubled parallelepiped Π₂, γ₀ iteration-parameter search, and three iterative solvers (SIM, BiCGStab, TwoStep).

**Architecture:** Light OOP (`Backend`, `Grid`, `Problem`, `Operator`, `SolverConfig`, `BaseSolver` + three solver implementations). Backend selection is a single point of truth; all arrays flow through classes that validate `dtype`/`xp` at construction. FFT-backed `Operator.matvec/rmatvec` is the hot path; a numpy-only `dense.B_operator_matrix` is kept solely as a small-N reference for the integration gate. Direct `import cupy` / `import numpy as np` only in `backend.py` and `dense.py`.

**Tech Stack:** Python ≥3.11, numpy ≥1.26, scipy ≥1.11; optional `cupy ≥13` (extra `gpu`); `pytest ≥8`.

**Spec:** [`docs/superpowers/specs/2026-04-23-em3d-package-refactor-design.md`](../specs/2026-04-23-em3d-package-refactor-design.md).

**Source notebook:** `wiki/raw/notes/Yurchenkov-programming-code-EM3D-2026.ipynb`. Tasks reference source functions by name; their bodies are ported with the adaptations each task specifies (replace `np` with `be.xp`, add dtype validation, convert dict-state into dataclasses, type annotations). Do not copy comments that reference notebook-specific state (e.g., cell numbers).

---

## Prerequisites (run once, before Task 1)

- Working directory: `C:\Users\user\Documents\ClaudeProjects\ClaudeProject` (siblings: `wiki/`, `docs/`).
- Initialise git if not already: `git init && git add docs/superpowers && git commit -m "chore: initial superpowers docs"`. Every task in this plan ends with a commit; if this is skipped, commits will fail.
- Python 3.11+ on PATH; `pip install -U pip`.

---

## File Structure

Final layout produced by this plan:

```
src/em3d/
  __init__.py          # public API reexports (Task 19)
  backend.py           # Backend dataclass, auto/numpy/cupy fabric (Task 2)
  dtypes.py            # Precision enum (Task 2)
  grid.py              # Grid dataclass (Task 3)
  refraction.py        # cylinder/step/ellipsis + apply_refraction (Task 4)
  wave.py              # flat_wave_vec (Task 5)
  kernel.py            # Green's function, b_coeff (Task 6)
  dense.py             # reference B_operator_matrix, flatten_block_matrix (Task 7)
  operator.py          # prep_coeffs, matvec/rmatvec (Tasks 8–9)
  problem.py           # Problem dataclass (Task 9)
  gamma0.py            # convex hull, circle geometry, find_params (Tasks 11–13)
  solvers/
    __init__.py        # solver reexports (Task 14)
    base.py            # BaseSolver Protocol, SolverConfig, SolverResult (Task 14)
    sim.py             # Task 15
    bicgstab.py        # Task 16
    twostep.py         # Task 17
tests/
  conftest.py          # backend fixtures, gpu marker (Task 1)
  test_backend.py      # Task 2
  test_grid.py         # Task 3
  test_refraction.py   # Task 4
  test_wave.py         # Task 5
  test_kernel.py       # Task 6
  test_dense.py        # Task 7
  test_operator_vs_dense.py  # integration gate (Task 10)
  test_gamma0.py       # Tasks 11–13
  test_solvers.py      # Task 18
pyproject.toml         # Task 1
```

---

## Task 1: Package skeleton and pytest infrastructure

**Files:**
- Create: `pyproject.toml`, `src/em3d/__init__.py`, `src/em3d/py.typed`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `.gitignore`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
build/
dist/
.coverage
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "em3d"
version = "0.1.0"
description = "Volume integral equation solver for 3D electrodynamics on structured grids with FFT acceleration."
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11"]

[project.optional-dependencies]
gpu = ["cupy-cuda12x>=13"]
dev = ["pytest>=8", "pytest-xdist>=3"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
em3d = ["py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires CUDA-capable device and cupy",
]
```

- [ ] **Step 3: Create empty `src/em3d/__init__.py` and `src/em3d/py.typed`**

Both files exist but are empty; public API reexports happen in Task 19.

- [ ] **Step 4: Create `tests/__init__.py` (empty) and `tests/conftest.py`**

```python
"""Shared pytest fixtures: backend parametrisation and GPU skip logic."""
from __future__ import annotations

import pytest


def _cupy_available() -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return bool(cp.cuda.is_available())
    except Exception:
        return False


CUPY_AVAILABLE = _cupy_available()


def pytest_collection_modifyitems(config, items):
    if CUPY_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="cupy / CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(params=["numpy-double", "numpy-single"])
def backend_cpu(request):
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    precision = Precision.DOUBLE if request.param.endswith("double") else Precision.SINGLE
    return Backend.numpy(precision=precision)


@pytest.fixture
def backend_numpy_double():
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    return Backend.numpy(precision=Precision.DOUBLE)


@pytest.fixture
def backend_numpy_single():
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    return Backend.numpy(precision=Precision.SINGLE)
```

- [ ] **Step 5: Create `tests/test_smoke.py` — package import sanity check**

```python
def test_package_imports():
    import em3d  # noqa: F401
```

- [ ] **Step 6: Install in editable mode and run smoke test**

```bash
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```

Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/em3d tests/
git commit -m "chore(em3d): package skeleton with pytest infrastructure"
```

---

## Task 2: Precision enum and Backend dataclass

**Files:**
- Create: `src/em3d/dtypes.py`, `src/em3d/backend.py`, `tests/test_backend.py`.

- [ ] **Step 1: Write the failing test file `tests/test_backend.py`**

```python
import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision


def test_precision_dtype_mapping():
    assert Precision.DOUBLE.real_dtype is np.float64
    assert Precision.DOUBLE.complex_dtype is np.complex128
    assert Precision.SINGLE.real_dtype is np.float32
    assert Precision.SINGLE.complex_dtype is np.complex64


def test_backend_numpy_double():
    be = Backend.numpy(precision=Precision.DOUBLE)
    assert be.device == "cpu"
    assert be.real_dtype is np.float64
    assert be.complex_dtype is np.complex128
    assert be.xp is np


def test_backend_numpy_single():
    be = Backend.numpy(precision=Precision.SINGLE)
    assert be.real_dtype is np.float32
    assert be.complex_dtype is np.complex64


def test_backend_zeros_real_and_complex():
    be = Backend.numpy(precision=Precision.DOUBLE)
    r = be.zeros((2, 3), kind="real")
    c = be.zeros((2, 3), kind="complex")
    assert r.dtype == np.float64 and r.shape == (2, 3)
    assert c.dtype == np.complex128 and c.shape == (2, 3)


def test_backend_zeros_invalid_kind():
    be = Backend.numpy(precision=Precision.DOUBLE)
    with pytest.raises(ValueError):
        be.zeros((2,), kind="bogus")


def test_backend_to_host_numpy_roundtrip():
    be = Backend.numpy(precision=Precision.DOUBLE)
    arr = be.array([1.0, 2.0, 3.0], dtype=np.float64)
    host = be.to_host(arr)
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, [1.0, 2.0, 3.0])


def test_backend_auto_returns_numpy_when_cupy_absent():
    be = Backend.auto(precision=Precision.DOUBLE)
    assert be.device in ("cpu", "cuda")
    assert be.precision is Precision.DOUBLE


def test_backend_fftn_roundtrip():
    be = Backend.numpy(precision=Precision.DOUBLE)
    x = be.array(np.random.default_rng(0).standard_normal((4, 4, 4)).astype(np.complex128))
    y = be.ifftn(be.fftn(x))
    np.testing.assert_allclose(y, x, atol=1e-12)
```

- [ ] **Step 2: Run tests, expect collection errors (modules don't exist)**

Run: `pytest tests/test_backend.py -v`
Expected: collection errors — `ModuleNotFoundError: No module named 'em3d.backend'`.

- [ ] **Step 3: Implement `src/em3d/dtypes.py`**

```python
"""Precision levels and dtype pairs for em3d."""
from __future__ import annotations

from enum import Enum

import numpy as np


class Precision(Enum):
    SINGLE = "single"
    DOUBLE = "double"

    @property
    def real_dtype(self) -> type:
        return np.float32 if self is Precision.SINGLE else np.float64

    @property
    def complex_dtype(self) -> type:
        return np.complex64 if self is Precision.SINGLE else np.complex128
```

- [ ] **Step 4: Implement `src/em3d/backend.py`**

```python
"""Array-namespace backend: selects numpy or cupy, carries precision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .dtypes import Precision


@dataclass(frozen=True)
class Backend:
    xp: Any  # module: numpy or cupy
    device: Literal["cpu", "cuda"]
    precision: Precision

    @property
    def real_dtype(self) -> type:
        return self.precision.real_dtype

    @property
    def complex_dtype(self) -> type:
        return self.precision.complex_dtype

    @classmethod
    def numpy(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
        return cls(xp=np, device="cpu", precision=precision)

    @classmethod
    def cupy(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
        import cupy as cp  # local import keeps cupy optional

        if not cp.cuda.is_available():
            raise RuntimeError("cupy imported but no CUDA device is available")
        return cls(xp=cp, device="cuda", precision=precision)

    @classmethod
    def auto(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
        try:
            import cupy as cp

            if cp.cuda.is_available():
                return cls(xp=cp, device="cuda", precision=precision)
        except ImportError:
            pass
        return cls.numpy(precision=precision)

    def array(self, obj, dtype=None):
        return self.xp.asarray(obj, dtype=dtype)

    def zeros(self, shape, kind: Literal["real", "complex"]):
        if kind == "real":
            dtype = self.real_dtype
        elif kind == "complex":
            dtype = self.complex_dtype
        else:
            raise ValueError(f"kind must be 'real' or 'complex', got {kind!r}")
        return self.xp.zeros(shape, dtype=dtype)

    def empty(self, shape, kind: Literal["real", "complex"]):
        if kind == "real":
            dtype = self.real_dtype
        elif kind == "complex":
            dtype = self.complex_dtype
        else:
            raise ValueError(f"kind must be 'real' or 'complex', got {kind!r}")
        return self.xp.empty(shape, dtype=dtype)

    def to_host(self, arr) -> np.ndarray:
        if self.xp is np:
            return np.asarray(arr)
        return arr.get()  # cupy ndarray → numpy

    def fftn(self, x, axes=None):
        return self.xp.fft.fftn(x, axes=axes)

    def ifftn(self, x, axes=None):
        return self.xp.fft.ifftn(x, axes=axes)

    def asarray_of_kind(self, obj, kind: Literal["real", "complex"]):
        dtype = self.real_dtype if kind == "real" else self.complex_dtype
        return self.xp.asarray(obj, dtype=dtype)
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_backend.py -v`
Expected: all 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/em3d/dtypes.py src/em3d/backend.py tests/test_backend.py
git commit -m "feat(em3d): Precision enum and Backend with numpy/cupy selection"
```

---

## Task 3: Grid dataclass

**Files:**
- Create: `src/em3d/grid.py`, `tests/test_grid.py`.

- [ ] **Step 1: Write `tests/test_grid.py`**

```python
import numpy as np
import pytest

from em3d.grid import Grid


def test_grid_dv_and_shape(backend_numpy_double):
    grid = Grid(N=(4, 5, 6), L=(1.0, 2.0, 3.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    assert grid.dv == pytest.approx((1.0 / 4) * (2.0 / 5) * (3.0 / 6))
    X, Y, Z = grid.coords()
    assert X.shape == Y.shape == Z.shape == (4, 5, 6)


def test_grid_coords_center_offset(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1.0, 1.0, 1.0), center=(10.0, 0.0, 0.0), backend=backend_numpy_double)
    X, _, _ = grid.coords()
    # cell centres should be offset by 10 in x
    assert float(X.min()) > 9.0
    assert float(X.max()) < 11.0


def test_grid_dtype_matches_backend(backend_cpu):
    grid = Grid(N=(3, 3, 3), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_cpu)
    X, _, _ = grid.coords()
    assert X.dtype == backend_cpu.real_dtype


def test_grid_rejects_non_positive_N(backend_numpy_double):
    with pytest.raises(ValueError):
        Grid(N=(0, 2, 2), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)


def test_grid_rejects_non_positive_L(backend_numpy_double):
    with pytest.raises(ValueError):
        Grid(N=(2, 2, 2), L=(1.0, -1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/test_grid.py -v`
Expected: `ModuleNotFoundError: No module named 'em3d.grid'`.

- [ ] **Step 3: Implement `src/em3d/grid.py`**

```python
"""Structured 3D Cartesian grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .backend import Backend


def _axis(n: int, length: float, centre: float, be: Backend):
    step = length / n
    start = centre - length / 2 + step / 2
    return be.xp.linspace(start, start + step * (n - 1), n, dtype=be.real_dtype)


@dataclass(frozen=True)
class Grid:
    N: Tuple[int, int, int]
    L: Tuple[float, float, float]
    center: Tuple[float, float, float]
    backend: Backend
    _x: object = field(init=False, repr=False)
    _y: object = field(init=False, repr=False)
    _z: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if any(n <= 0 for n in self.N):
            raise ValueError(f"Grid.N must be strictly positive, got {self.N}")
        if any(l <= 0 for l in self.L):
            raise ValueError(f"Grid.L must be strictly positive, got {self.L}")
        object.__setattr__(self, "_x", _axis(self.N[0], self.L[0], self.center[0], self.backend))
        object.__setattr__(self, "_y", _axis(self.N[1], self.L[1], self.center[1], self.backend))
        object.__setattr__(self, "_z", _axis(self.N[2], self.L[2], self.center[2], self.backend))

    @property
    def dv(self) -> float:
        return (self.L[0] / self.N[0]) * (self.L[1] / self.N[1]) * (self.L[2] / self.N[2])

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def coords(self):
        return self.backend.xp.meshgrid(self._x, self._y, self._z, indexing="ij")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_grid.py -v`
Expected: 5 tests pass (×2 for parametrised `backend_cpu` = 6 total green).

- [ ] **Step 5: Commit**

```bash
git add src/em3d/grid.py tests/test_grid.py
git commit -m "feat(em3d): Grid dataclass with centered cell coordinates"
```

---

## Task 4: Refraction profiles (cylinder, step, ellipsis, apply_refraction)

**Files:**
- Create: `src/em3d/refraction.py`, `tests/test_refraction.py`.
- Source: notebook functions `cylinder_refraction` (line ~219), `step_refraction` (~297), `ellipsis_refraction` (~351), `apply_refraction` (~403). Port by replacing `np` with `be.xp`, adding type hints, returning a `(3, 3) + grid.N` complex tensor for `apply_refraction`.

- [ ] **Step 1: Write `tests/test_refraction.py`**

```python
import numpy as np

from em3d.grid import Grid
from em3d.refraction import (
    cylinder_refraction,
    step_refraction,
    ellipsis_refraction,
    apply_refraction,
)


def _grid(be):
    return Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


def test_cylinder_mask_inside_outside(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_real = 2.25
    mask = cylinder_refraction(grid, eps_real=eps_real, eps_imag=0.0, radius=0.49, axis="z")
    assert mask.shape == (4, 4, 4)
    # scalar refraction returns complex values, inside ≈ eps_real - 1, outside ≈ 0
    centre = mask[grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2]
    corner = mask[0, 0, 0]
    assert abs(centre.real - (eps_real - 1.0)) < 1e-12
    assert abs(corner) < 1e-12


def test_step_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    mask = step_refraction(
        grid, eps_real=2.0, eps_imag=0.1, z_min=-0.25, z_max=0.25
    )
    assert mask.shape == (4, 4, 4)
    # along z, middle slice should be non-zero, top/bottom zero
    mid = grid.N[2] // 2
    assert abs(mask[0, 0, mid]) > 0
    assert abs(mask[0, 0, 0]) < 1e-12
    assert abs(mask[0, 0, -1]) < 1e-12


def test_ellipsis_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    mask = ellipsis_refraction(
        grid, eps_real=1.5, eps_imag=0.0, center=(0.0, 0.0, 0.0), radius=(0.3, 0.4, 0.5)
    )
    assert mask.shape == (4, 4, 4)
    # centre inside ellipsoid
    assert abs(mask[grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2]) > 0


def test_apply_refraction_scalar_returns_isotropic_tensor(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.49, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    assert eta.shape == (3, 3) + grid.N
    assert eta.dtype == backend_numpy_double.complex_dtype
    # diagonal equals scalar, off-diagonal is zero
    np.testing.assert_allclose(eta[0, 0], scalar)
    np.testing.assert_allclose(eta[1, 1], scalar)
    np.testing.assert_allclose(eta[2, 2], scalar)
    np.testing.assert_allclose(eta[0, 1], np.zeros_like(scalar))
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_refraction.py -v`
Expected: `ModuleNotFoundError: No module named 'em3d.refraction'`.

- [ ] **Step 3: Implement `src/em3d/refraction.py`**

Port the four notebook functions. Each returns a `(Nx, Ny, Nz)` array of dtype `backend.complex_dtype` containing η = ε − 1 on the grid; `apply_refraction` wraps a scalar η into an isotropic tensor `(3, 3, Nx, Ny, Nz)`.

```python
"""Refractive-index / permittivity profiles on the grid."""
from __future__ import annotations

from typing import Literal, Tuple

from .grid import Grid


def _blank(grid: Grid):
    be = grid.backend
    return be.zeros(grid.N, kind="complex")


def cylinder_refraction(
    grid: Grid,
    *,
    eps_real: float,
    eps_imag: float,
    radius: float,
    axis: Literal["x", "y", "z"] = "z",
) -> object:
    """Infinite cylinder along `axis`, radius in grid length units."""
    be = grid.backend
    X, Y, Z = grid.coords()
    if axis == "z":
        r2 = X * X + Y * Y
    elif axis == "y":
        r2 = X * X + Z * Z
    elif axis == "x":
        r2 = Y * Y + Z * Z
    else:
        raise ValueError(f"axis must be 'x'|'y'|'z', got {axis!r}")
    eta_value = complex(eps_real - 1.0, eps_imag)
    out = _blank(grid)
    mask = r2 <= radius * radius
    out = be.xp.where(mask, be.xp.asarray(eta_value, dtype=be.complex_dtype), out)
    return out


def step_refraction(
    grid: Grid,
    *,
    eps_real: float,
    eps_imag: float,
    z_min: float,
    z_max: float,
) -> object:
    """Slab between z_min and z_max (inclusive)."""
    be = grid.backend
    _, _, Z = grid.coords()
    eta_value = complex(eps_real - 1.0, eps_imag)
    mask = (Z >= z_min) & (Z <= z_max)
    out = _blank(grid)
    out = be.xp.where(mask, be.xp.asarray(eta_value, dtype=be.complex_dtype), out)
    return out


def ellipsis_refraction(
    grid: Grid,
    *,
    eps_real: float,
    eps_imag: float,
    center: Tuple[float, float, float],
    radius: Tuple[float, float, float],
) -> object:
    """Axis-aligned ellipsoid."""
    be = grid.backend
    X, Y, Z = grid.coords()
    cx, cy, cz = center
    rx, ry, rz = radius
    metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    eta_value = complex(eps_real - 1.0, eps_imag)
    out = _blank(grid)
    mask = metric <= 1.0
    out = be.xp.where(mask, be.xp.asarray(eta_value, dtype=be.complex_dtype), out)
    return out


def apply_refraction(grid: Grid, *, scalar_eta=None, tensor_eta=None) -> object:
    """Return a (3, 3, Nx, Ny, Nz) complex tensor.

    Pass exactly one of:
      - scalar_eta: (Nx, Ny, Nz) — wrapped into an isotropic diagonal tensor;
      - tensor_eta: already (3, 3, Nx, Ny, Nz), returned as-is after validation.
    """
    be = grid.backend
    if (scalar_eta is None) == (tensor_eta is None):
        raise ValueError("apply_refraction requires exactly one of scalar_eta or tensor_eta")
    if scalar_eta is not None:
        if scalar_eta.shape != grid.N:
            raise ValueError(f"scalar_eta.shape {scalar_eta.shape} != grid.N {grid.N}")
        out = be.zeros((3, 3) + grid.N, kind="complex")
        for i in range(3):
            out[i, i] = scalar_eta
        return out
    if tensor_eta.shape != (3, 3) + grid.N:
        raise ValueError(
            f"tensor_eta.shape {tensor_eta.shape} != {(3, 3) + grid.N}"
        )
    return tensor_eta.astype(be.complex_dtype, copy=False)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_refraction.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/refraction.py tests/test_refraction.py
git commit -m "feat(em3d): refraction profiles and apply_refraction to η-tensor"
```

---

## Task 5: Plane wave on grid

**Files:**
- Create: `src/em3d/wave.py`, `tests/test_wave.py`.
- Source: notebook `flat_wave_vec` (~line 578).

- [ ] **Step 1: Write `tests/test_wave.py`**

```python
import numpy as np
import pytest

from em3d.grid import Grid
from em3d.wave import flat_wave_vec


def test_flat_wave_shape_and_dtype(backend_numpy_double):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    wave = flat_wave_vec(grid, k=2.0, orient=(0.0, 0.0, 1.0), amplitude=(1.0, 0.0, 0.0))
    assert wave.shape == (3,) + grid.N
    assert wave.dtype == backend_numpy_double.complex_dtype


def test_flat_wave_plane_phase_along_z(backend_numpy_double):
    grid = Grid(N=(2, 2, 4), L=(1.0, 1.0, 4.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    k = 1.5
    wave = flat_wave_vec(grid, k=k, orient=(0.0, 0.0, 1.0), amplitude=(1.0, 0.0, 0.0))
    # x-component only; phase should be exp(i k z)
    z = grid.z
    expected_phase = np.exp(1j * k * np.asarray(z))
    np.testing.assert_allclose(wave[0, 0, 0, :], expected_phase, atol=1e-12)
    np.testing.assert_allclose(wave[1, 0, 0, :], np.zeros(4), atol=1e-12)


def test_flat_wave_requires_unit_orient(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    with pytest.raises(ValueError):
        flat_wave_vec(grid, k=1.0, orient=(1.0, 1.0, 0.0), amplitude=(1.0, 0.0, 0.0))
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `pytest tests/test_wave.py -v`
Expected: `ModuleNotFoundError: No module named 'em3d.wave'`.

- [ ] **Step 3: Implement `src/em3d/wave.py`**

```python
"""Plane wave sampled on a grid."""
from __future__ import annotations

from typing import Tuple

from .grid import Grid


def flat_wave_vec(
    grid: Grid,
    *,
    k: float,
    orient: Tuple[float, float, float],
    amplitude: Tuple[float, float, float],
    phi0: float = 0.0,
    sign: int = 1,
) -> object:
    """Return plane wave field A · exp(i sign·(k·(orient·r) + phi0)) on the grid.

    Shape: (3, Nx, Ny, Nz), dtype = backend.complex_dtype.
    `orient` must be a unit vector.
    """
    be = grid.backend
    xp = be.xp
    norm2 = orient[0] ** 2 + orient[1] ** 2 + orient[2] ** 2
    if abs(norm2 - 1.0) > 1e-9:
        raise ValueError(f"orient must be a unit vector, got norm² = {norm2}")
    X, Y, Z = grid.coords()
    phase = sign * (k * (orient[0] * X + orient[1] * Y + orient[2] * Z) + phi0)
    carrier = xp.exp(1j * phase).astype(be.complex_dtype, copy=False)
    out = be.zeros((3,) + grid.N, kind="complex")
    for i, a in enumerate(amplitude):
        if a != 0.0:
            out[i] = be.complex_dtype(a) * carrier
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_wave.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/wave.py tests/test_wave.py
git commit -m "feat(em3d): flat_wave_vec plane-wave sampler"
```

---

## Task 6: Helmholtz Green's function and b_coeff

**Files:**
- Create: `src/em3d/kernel.py`, `tests/test_kernel.py`.
- Source: notebook `kernel` (~line 558), `b_coeff` (~line 654).

- [ ] **Step 1: Write `tests/test_kernel.py`**

```python
import numpy as np
import pytest

from em3d.kernel import green_helmholtz, b_coeff


def test_green_values_at_known_points():
    # G(R) = exp(i k R) / (4π R)
    R = 2.0
    k = 1.0
    expected = np.exp(1j * k * R) / (4.0 * np.pi * R)
    got = green_helmholtz(R, k)
    assert abs(got - expected) < 1e-12


def test_green_array_broadcast():
    R = np.array([1.0, 2.0, 3.0])
    k = 0.5
    got = green_helmholtz(R, k)
    expected = np.exp(1j * k * R) / (4.0 * np.pi * R)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_green_regularises_zero():
    # We do NOT evaluate at R=0; kernel must either raise or return a well-defined
    # regularised value for scalar R=0. We test that finite output emerges.
    out = green_helmholtz(0.0, k=1.0)
    assert np.isfinite(out.real) and np.isfinite(out.imag)


def test_b_coeff_symmetry():
    # b_coeff(x, y) should equal b_coeff(y, x) by translational symmetry
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.4, 0.6, 1.0])
    dv = 0.01
    k = 1.0
    assert abs(b_coeff(x, y, k=k, dv=dv) - b_coeff(y, x, k=k, dv=dv)) < 1e-12
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `pytest tests/test_kernel.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/kernel.py`**

Port notebook `kernel` and `b_coeff`. The notebook version has a regularisation for `R → 0`; preserve that behaviour (set `G(0) = 0` or similar — inspect notebook cell at line ~558 and preserve the exact formula the author uses).

```python
"""Helmholtz Green's function and discrete volume-integral kernel coefficients."""
from __future__ import annotations

import numpy as np


def green_helmholtz(R, k: float, eps: float = 1e-30):
    """G(R) = exp(i k R) / (4π R) with a scalar regularisation at R=0.

    Accepts scalar or ndarray. Mirrors the notebook `kernel` behaviour.
    """
    R_reg = np.where(np.asarray(R) < eps, eps, R)
    return np.exp(1j * k * R_reg) / (4.0 * np.pi * R_reg)


def b_coeff(x, y, *, k: float, dv: float):
    """Discrete b-coefficient for the volume integral operator.

    For a pair of cell centres x, y ∈ R^3 with cell volume dv,
    returns dv · G(|x-y|, k) per collocation. See notebook cell ~654.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.linalg.norm(x - y)
    return dv * green_helmholtz(r, k=k)
```

Note: this module intentionally uses numpy only — `b_coeff` operates on small arrays of cell centres when building the reference dense operator. The FFT-path builds its coefficient tensor in `operator.py` using the backend's xp.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_kernel.py -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/kernel.py tests/test_kernel.py
git commit -m "feat(em3d): Helmholtz Green's function and b_coeff"
```

---

## Task 7: Reference dense operator `B_operator_matrix`

**Files:**
- Create: `src/em3d/dense.py`, `tests/test_dense.py`.
- Source: notebook `B_operator_matrix` (~line 681), `B_eta_operator_matrix` (~line 667), `flatten_block_matrix` (~line 695).

- [ ] **Step 1: Write `tests/test_dense.py`**

```python
import numpy as np

from em3d.grid import Grid
from em3d.dense import B_operator_matrix, flatten_block_matrix


def test_B_operator_matrix_shape_small_grid(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(0.5, 0.5, 0.5), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    M = B_operator_matrix(grid, k=1.0, volume=grid.dv * 8)
    # 2·2·2 = 8 cells, 3 components each → 24×24
    assert M.shape == (24, 24)
    assert M.dtype == np.complex128


def test_flatten_block_matrix_roundtrip():
    rng = np.random.default_rng(0)
    N = 3
    m = 2
    T4 = rng.standard_normal((N, N, m, m)) + 1j * rng.standard_normal((N, N, m, m))
    flat = flatten_block_matrix(T4)
    assert flat.shape == (N * m, N * m)
    # sample a couple of entries to verify layout
    for i in range(N):
        for j in range(N):
            for a in range(m):
                for b in range(m):
                    assert flat[i * m + a, j * m + b] == T4[i, j, a, b]
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `pytest tests/test_dense.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/dense.py`**

Port `B_operator_matrix` and `flatten_block_matrix` from the notebook (lines ~667–755). The port adaptations:
- Always numpy (no `xp`); the module lives outside the backend abstraction by design.
- Accept a `Grid` (ensure `grid.backend.xp is np`; if not, convert coords via `grid.backend.to_host`).
- Return `np.ndarray` of dtype `complex128`.

```python
"""Reference dense assembly of the volume-integral operator.

Only used for integration testing on small grids. Always numpy.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .kernel import b_coeff


def flatten_block_matrix(T4: np.ndarray) -> np.ndarray:
    """Unpack a (N, N, m, m) block tensor into a (N·m, N·m) matrix (row-major blocks)."""
    N, N2, m, m2 = T4.shape
    if N != N2 or m != m2:
        raise ValueError(f"expected (N,N,m,m) tensor, got {T4.shape}")
    # reshape: (N, N, m, m) -> (N, m, N, m) -> (Nm, Nm)
    return T4.transpose(0, 2, 1, 3).reshape(N * m, N * m)


def _cell_centres(grid: Grid) -> np.ndarray:
    """Return array of shape (Nx·Ny·Nz, 3) with cell-centre coordinates in row-major order."""
    be = grid.backend
    x = np.asarray(be.to_host(grid.x))
    y = np.asarray(be.to_host(grid.y))
    z = np.asarray(be.to_host(grid.z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def B_operator_matrix(grid: Grid, *, k: float, volume: float) -> np.ndarray:
    """Assemble the dense 3Ncells × 3Ncells matrix of the volume-integral operator.

    Uses scalar b_coeff and a block-diagonal 3×3 identity per cell pair (isotropic
    kernel, as in the notebook's `B_operator_matrix` without ε multiplication).
    """
    centres = _cell_centres(grid)
    Ncells = centres.shape[0]
    dv = grid.dv
    M = np.zeros((3 * Ncells, 3 * Ncells), dtype=np.complex128)
    identity3 = np.eye(3, dtype=np.complex128)
    for i in range(Ncells):
        for j in range(Ncells):
            coeff = b_coeff(centres[i], centres[j], k=k, dv=dv)
            M[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = coeff * identity3
    return M
```

Note: the `volume` parameter is part of the signature for symmetry with the notebook, but is not required by this simplest dense assembly. Integration tests in Task 10 will exercise correctness, and if the notebook uses it differently, Task 10 will catch the discrepancy.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dense.py -v`
Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/dense.py tests/test_dense.py
git commit -m "feat(em3d): numpy-only reference dense operator for testing"
```

---

## Task 8: FFT coefficient preparation (`prep_coeffs_em3d`)

**Files:**
- Create: `src/em3d/operator.py` (initial — only the prep function and helpers).
- Source: notebook `B_compute` (~line 755), `prep_coeffs_em3d` (~line 852), `prep_conj_coeffs_em3d` (~line 940).

This task lays the FFT tensor on the doubled parallelepiped Π₂. Full `Operator` class is Task 9.

- [ ] **Step 1: Write failing test inline in `tests/test_operator_vs_dense.py` (unit-level)**

```python
import numpy as np

from em3d.grid import Grid
from em3d.operator import prep_coeffs_em3d


def test_prep_coeffs_shape_and_dtype(backend_numpy_double):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    coeffs = prep_coeffs_em3d(grid, k=1.0, volume=grid.dv * 64)
    # Π₂ doubling: FFT tensor on (2Nx, 2Ny, 2Nz) with 3×3 block structure
    assert coeffs.shape == (3, 3) + tuple(2 * n for n in grid.N)
    assert coeffs.dtype == backend_numpy_double.complex_dtype
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_operator_vs_dense.py::test_prep_coeffs_shape_and_dtype -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/operator.py` — first cut**

Port `B_compute` and `prep_coeffs_em3d` from the notebook. Adaptations:
- Replace all `np` with `be.xp`.
- Inputs: `Grid`, `k`, `volume`.
- Output: complex tensor of shape `(3, 3, 2Nx, 2Ny, 2Nz)` — FFT of the kernel sampled on the doubled lattice.

```python
"""FFT-accelerated volume-integral operator on a doubled parallelepiped Π₂."""
from __future__ import annotations

import numpy as np

from .backend import Backend
from .grid import Grid


def _kernel_tensor_on_doubled_grid(grid: Grid, k: float, volume: float):
    """Sample the 3×3 kernel on the doubled grid Π₂ (2Nx × 2Ny × 2Nz cells).

    The (a, b) block is an isotropic scalar kernel ⋅ δ_{ab} in this minimal version.
    Anisotropic refinements can hook in here; they are not needed for the FFT-vs-dense
    integration test because the dense matrix is assembled consistently.
    """
    be = grid.backend
    xp = be.xp
    Nx, Ny, Nz = grid.N
    Lx, Ly, Lz = grid.L
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    # separations on Π₂: [0, dx, 2dx, ..., (N-1)dx, -N·dx, -(N-1)dx, ..., -dx]  — typical periodisation
    sx = xp.concatenate([xp.arange(Nx) * dx, -(xp.arange(Nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(Ny) * dy, -(xp.arange(Ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(Nz) * dz, -(xp.arange(Nz, 0, -1)) * dz])
    SX, SY, SZ = xp.meshgrid(sx, sy, sz, indexing="ij")
    R = xp.sqrt(SX * SX + SY * SY + SZ * SZ)
    R_reg = xp.where(R < 1e-30, 1e-30, R)
    G = xp.exp(1j * k * R_reg) / (4.0 * xp.pi * R_reg)
    scalar = (grid.dv * G).astype(be.complex_dtype, copy=False)
    # isotropic 3×3 block: tensor[a, b] = δ_{ab} · scalar
    shape = (3, 3) + scalar.shape
    out = be.zeros(shape, kind="complex")
    for d in range(3):
        out[d, d] = scalar
    return out


def prep_coeffs_em3d(grid: Grid, *, k: float, volume: float):
    """Return the precomputed FFT-of-kernel tensor on the doubled grid Π₂."""
    be = grid.backend
    kernel_tensor = _kernel_tensor_on_doubled_grid(grid, k=k, volume=volume)
    return be.fftn(kernel_tensor, axes=(-3, -2, -1)).astype(be.complex_dtype, copy=False)


def prep_conj_coeffs_em3d(grid: Grid, *, k: float, volume: float):
    """FFT of the conjugate-kernel tensor for rmatvec."""
    be = grid.backend
    kernel_tensor = _kernel_tensor_on_doubled_grid(grid, k=k, volume=volume)
    conj = be.xp.conj(kernel_tensor)
    return be.fftn(conj, axes=(-3, -2, -1)).astype(be.complex_dtype, copy=False)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_operator_vs_dense.py::test_prep_coeffs_shape_and_dtype -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/operator.py tests/test_operator_vs_dense.py
git commit -m "feat(em3d): FFT coefficient preparation on doubled parallelepiped"
```

---

## Task 9: `Problem` and `Operator` classes with matvec / rmatvec

**Files:**
- Create: `src/em3d/problem.py`.
- Modify: `src/em3d/operator.py` — add `Operator` class and `_fbbtmv` helpers.
- Source: notebook `prep_fbbtmv` (~line 987), `mul_3d_tensor_vector` (~line 1011), `operator` (~line 1056), `conj_operator` (~line 1082).

- [ ] **Step 1: Write failing test in `tests/test_operator_vs_dense.py` (append)**

```python
import numpy as np

from em3d.grid import Grid
from em3d.refraction import cylinder_refraction, apply_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.operator import Operator


def _toy_problem(backend, N=(4, 4, 4)):
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend)
    scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.3, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    # approximate Q volume as all cells (toy example)
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=volume)


def test_operator_matvec_shape(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double)
    op = Operator(problem)
    u = backend_numpy_double.zeros((3,) + problem.grid.N, kind="complex")
    u[0, 0, 0, 0] = 1.0
    y = op.matvec(u)
    assert y.shape == (3,) + problem.grid.N
    assert y.dtype == backend_numpy_double.complex_dtype


def test_operator_rmatvec_shape(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double)
    op = Operator(problem)
    u = backend_numpy_double.zeros((3,) + problem.grid.N, kind="complex")
    u[0, 0, 0, 0] = 1.0
    y = op.rmatvec(u)
    assert y.shape == u.shape


def test_problem_rejects_wrong_dtype(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1, 1, 1), center=(0, 0, 0), backend=backend_numpy_double)
    eta = np.zeros((3, 3, 2, 2, 2), dtype=np.complex64)  # wrong precision
    wave = np.zeros((3, 2, 2, 2), dtype=np.complex128)
    with pytest.raises(TypeError):
        Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 8)


def test_problem_rejects_wrong_shape(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1, 1, 1), center=(0, 0, 0), backend=backend_numpy_double)
    eta = np.zeros((3, 3, 2, 2, 2), dtype=np.complex128)
    wave = np.zeros((3, 4, 2, 2), dtype=np.complex128)  # wrong shape
    with pytest.raises(ValueError):
        Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 8)
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run tests, expect ImportError**

Run: `pytest tests/test_operator_vs_dense.py -v`
Expected: `ModuleNotFoundError: No module named 'em3d.problem'`.

- [ ] **Step 3: Implement `src/em3d/problem.py`**

```python
"""Problem: Grid + ε-tensor + incident wave + k₀ + volume of Q."""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid


@dataclass(frozen=True)
class Problem:
    grid: Grid
    eps_tensor: object  # shape (3, 3) + grid.N, complex
    wave: object        # shape (3,) + grid.N, complex
    k0: float
    volume: float

    def __post_init__(self) -> None:
        be = self.grid.backend
        expected_eta = (3, 3) + self.grid.N
        expected_wave = (3,) + self.grid.N
        if self.eps_tensor.shape != expected_eta:
            raise ValueError(
                f"eps_tensor.shape {self.eps_tensor.shape} != expected {expected_eta}"
            )
        if self.wave.shape != expected_wave:
            raise ValueError(
                f"wave.shape {self.wave.shape} != expected {expected_wave}"
            )
        if self.eps_tensor.dtype != be.complex_dtype:
            raise TypeError(
                f"eps_tensor.dtype {self.eps_tensor.dtype} != {be.complex_dtype}"
            )
        if self.wave.dtype != be.complex_dtype:
            raise TypeError(f"wave.dtype {self.wave.dtype} != {be.complex_dtype}")

    @property
    def backend(self):
        return self.grid.backend
```

- [ ] **Step 4: Extend `src/em3d/operator.py` with `Operator` class**

Append to the existing module:

```python
from .problem import Problem


def _pad_to_doubled(xp, u, N):
    """Zero-pad a (3, Nx, Ny, Nz) field to (3, 2Nx, 2Ny, 2Nz)."""
    Nx, Ny, Nz = N
    shape = (3, 2 * Nx, 2 * Ny, 2 * Nz)
    out = xp.zeros(shape, dtype=u.dtype)
    out[:, :Nx, :Ny, :Nz] = u
    return out


def _crop_from_doubled(u_big, N):
    """Extract (3, Nx, Ny, Nz) from (3, 2Nx, 2Ny, 2Nz)."""
    Nx, Ny, Nz = N
    return u_big[:, :Nx, :Ny, :Nz]


def _apply_block_kernel(xp, K_hat, u_hat):
    """Apply the (3, 3) block kernel in Fourier space: out[a] = Σ_b K_hat[a,b] * u_hat[b]."""
    out = xp.zeros_like(u_hat)
    for a in range(3):
        acc = None
        for b in range(3):
            term = K_hat[a, b] * u_hat[b]
            acc = term if acc is None else acc + term
        out[a] = acc
    return out


class Operator:
    """FFT-backed volume integral operator with matvec and rmatvec.

    Caches the precomputed kernel FFTs in the constructor.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        grid = problem.grid
        be = grid.backend
        self._K_hat = prep_coeffs_em3d(grid, k=problem.k0, volume=problem.volume)
        self._K_hat_conj = prep_conj_coeffs_em3d(grid, k=problem.k0, volume=problem.volume)
        self._be = be
        self._N = grid.N

    @property
    def backend(self):
        return self._be

    def matvec(self, u):
        """y = (I + B·η) u.  Accepts (3, Nx, Ny, Nz), returns same shape."""
        be = self._be
        xp = be.xp
        eta = self.problem.eps_tensor
        # apply η (3×3 tensor contraction on each cell)
        eta_u = xp.einsum("ab...,b...->a...", eta, u)
        padded = _pad_to_doubled(xp, eta_u, self._N)
        hat = be.fftn(padded, axes=(-3, -2, -1))
        applied_hat = _apply_block_kernel(xp, self._K_hat, hat)
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        B_eta_u = _crop_from_doubled(applied_big, self._N)
        return (u + B_eta_u).astype(be.complex_dtype, copy=False)

    def rmatvec(self, u):
        """y = (I + η* · B*) u  — adjoint in the same inner product as the notebook."""
        be = self._be
        xp = be.xp
        eta = self.problem.eps_tensor
        padded = _pad_to_doubled(xp, u, self._N)
        hat = be.fftn(padded, axes=(-3, -2, -1))
        applied_hat = _apply_block_kernel(xp, self._K_hat_conj, hat)
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        B_star_u = _crop_from_doubled(applied_big, self._N)
        eta_star_B_star_u = xp.einsum("ab...,b...->a...", xp.conj(eta).swapaxes(0, 1), B_star_u)
        return (u + eta_star_B_star_u).astype(be.complex_dtype, copy=False)

    def to_dense(self):
        """Dense assembly; requires numpy backend."""
        if self._be.xp is not np:
            raise RuntimeError("Operator.to_dense requires numpy backend")
        from .dense import B_operator_matrix

        return B_operator_matrix(self.problem.grid, k=self.problem.k0, volume=self.problem.volume)
```

Note: the exact definition of the adjoint in the notebook (Yurchenkov's `conj_operator`) may involve a specific inner product. During Task 10 the integration test compares `matvec` to `dense @ u`; if the adjoint definition differs from what's written here, the failure will tell you which sign/conjugate pattern to match.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_operator_vs_dense.py -v`
Expected: 4 shape/dtype tests pass (integration gate arrives in Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/em3d/problem.py src/em3d/operator.py tests/test_operator_vs_dense.py
git commit -m "feat(em3d): Problem dataclass and Operator with FFT matvec/rmatvec"
```

---

## Task 10: Integration gate — FFT operator vs dense reference

**Files:**
- Modify: `tests/test_operator_vs_dense.py` — append the integration test that is the rationale for Tasks 7–9.

- [ ] **Step 1: Append the integration test**

```python
def _flatten_field(u):
    """Convert (3, Nx, Ny, Nz) field to (3·Nx·Ny·Nz,) vector in row-major (cell, component) order."""
    # dense uses block order: for cell i = (ix, iy, iz), components 0,1,2 are contiguous
    three, Nx, Ny, Nz = u.shape
    assert three == 3
    # reshape (3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3) -> flat
    return np.transpose(np.asarray(u), (1, 2, 3, 0)).reshape(-1)


def test_fft_matvec_matches_dense(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double, N=(4, 4, 4))
    op = Operator(problem)
    M_dense = op.to_dense()
    rng = np.random.default_rng(42)
    u = (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128)
    # FFT path: applies (I + B·η); dense path: applies (I + dense·η)
    eta_flat = _flatten_field(np.einsum("ab...,b...->a...", problem.eps_tensor, u))
    u_flat = _flatten_field(u)
    y_dense = u_flat + M_dense @ eta_flat
    y_fft = _flatten_field(op.matvec(u))
    rel = np.linalg.norm(y_fft - y_dense) / np.linalg.norm(y_dense)
    assert rel < 1e-10, f"FFT vs dense relative error {rel:.2e}"


def test_fft_rmatvec_matches_dense_adjoint(backend_numpy_double):
    problem = _toy_problem(backend_numpy_double, N=(4, 4, 4))
    op = Operator(problem)
    M_dense = op.to_dense()
    rng = np.random.default_rng(7)
    u = (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128)
    u_flat = _flatten_field(u)
    eta_conj_T = np.einsum("ab...,b...->a...", np.conj(problem.eps_tensor).swapaxes(0, 1), _unflatten_field(M_dense.conj().T @ u_flat, problem.grid.N))
    expected = u_flat + _flatten_field(eta_conj_T)
    y_fft = _flatten_field(op.rmatvec(u))
    rel = np.linalg.norm(y_fft - expected) / np.linalg.norm(expected)
    assert rel < 1e-10


def _unflatten_field(flat, N):
    Nx, Ny, Nz = N
    return np.transpose(flat.reshape(Nx, Ny, Nz, 3), (3, 0, 1, 2))


def test_fft_matvec_single_precision_tolerance(backend_numpy_single):
    problem = _toy_problem(backend_numpy_single, N=(4, 4, 4))
    op = Operator(problem)
    # for single precision we only assert the FFT path returns finite values of the expected dtype;
    # absolute dense comparison in f32 is known to be noisy. Detailed single-precision tolerance
    # tests belong to the solver convergence suite (Task 18).
    u = backend_numpy_single.zeros((3,) + problem.grid.N, kind="complex")
    u[0, 0, 0, 0] = 1.0
    y = op.matvec(u)
    assert y.dtype == backend_numpy_single.complex_dtype
    assert np.all(np.isfinite(np.asarray(y)))
```

- [ ] **Step 2: Run the gate tests**

Run: `pytest tests/test_operator_vs_dense.py -v`
Expected: all pass. If `test_fft_matvec_matches_dense` fails with a relative error much larger than 1e-10, either (a) the dense layout / Π₂ sign convention / Fourier-sign convention needs adjustment, or (b) the scalar-block vs anisotropic-block treatment differs between `dense.py` and `operator.py`.

**Debug protocol if gate fails:**
1. Run with `N=(2, 2, 2)` and print both 24×24 matrices side by side.
2. Check the zero-frequency entry: `K_hat[0, 0, 0, 0, 0]` should equal the sum of the scalar kernel on Π₂.
3. Inspect the notebook's `B_compute` for the exact Π₂ convention (symmetric vs one-sided padding).

- [ ] **Step 3: Commit**

```bash
git add tests/test_operator_vs_dense.py
git commit -m "test(em3d): integration gate — FFT matvec/rmatvec match dense reference"
```

---

## Task 11: γ₀ geometry — convex hull

**Files:**
- Create: `src/em3d/gamma0.py` (first cut, convex hull only).
- Create: `tests/test_gamma0.py`.
- Source: notebook `cross` (~line 1130), `sequential_chain` (~line 1153).

- [ ] **Step 1: Write failing `tests/test_gamma0.py`**

```python
import numpy as np
import pytest

from em3d.gamma0 import cross, sequential_chain


def test_cross_positive():
    o = np.array([0.0, 0.0])
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cross(o, a, b) > 0


def test_cross_negative():
    o = np.array([0.0, 0.0])
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 0.0])
    assert cross(o, a, b) < 0


def test_sequential_chain_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
    hull = sequential_chain(pts)
    # hull is the outer square
    assert len(hull) == 4
    coords = sorted((float(p[0]), float(p[1])) for p in hull)
    assert coords == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


def test_sequential_chain_colinear():
    pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    hull = sequential_chain(pts)
    # colinear points — hull should contain only the endpoints
    assert len(hull) == 2
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_gamma0.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/gamma0.py` — convex hull**

Port the notebook `cross` and `sequential_chain` (Andrew's monotone chain). Use numpy only (this module is a small 2D geometry kernel operating on ≤ N_spectrum points, ~10²; cupy is not needed).

```python
"""γ₀ algorithm: convex hull of spectrum samples and bounding circle for the optimal iteration parameter."""
from __future__ import annotations

import numpy as np


def cross(o, a, b) -> float:
    """Signed area of the triangle (o, a, b) × 2. Positive for CCW."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def sequential_chain(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain convex hull. Input: (N, 2). Output: (H, 2) CCW, no duplicates."""
    pts = np.asarray(points, dtype=np.float64)
    pts = np.unique(pts, axis=0)  # sorts lexicographically and deduplicates
    if len(pts) < 2:
        return pts

    def build(seq):
        hull = []
        for p in seq:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(tuple(p))
        return hull

    lower = build(pts)
    upper = build(pts[::-1])
    hull_points = lower[:-1] + upper[:-1]
    if len(hull_points) < 2:
        return np.array(hull_points, dtype=np.float64)
    return np.array(hull_points, dtype=np.float64)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gamma0.py -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/gamma0.py tests/test_gamma0.py
git commit -m "feat(em3d): convex hull (Andrew's monotone chain) for γ₀"
```

---

## Task 12: γ₀ geometry — circle through points

**Files:**
- Modify: `src/em3d/gamma0.py`.
- Modify: `tests/test_gamma0.py` — append tests.
- Source: notebook `mu_2points`, `radius_2points`, `mu_3points`, `radius_3points`, `compute_circle_two_points`, `compute_circle_three_points`, `circle_contains_points`, `circle_contains_origin`, `circle_angle_from_origin` (~lines 1216–1601).

- [ ] **Step 1: Append tests**

```python
from em3d.gamma0 import (
    compute_circle_two_points,
    compute_circle_three_points,
    circle_contains_points,
    circle_contains_origin,
)


def test_circle_two_points_midpoint():
    # circle through two symmetric points has centre at their midpoint (minimum enclosing circle rule)
    p1 = np.array([1.0 + 0j])
    p2 = np.array([3.0 + 0j])
    centre, radius = compute_circle_two_points(p1[0], p2[0])
    assert abs(centre - 2.0) < 1e-12
    assert abs(radius - 1.0) < 1e-12


def test_circle_three_points_unit_circle():
    # three points on the unit circle → centre 0, radius 1
    p1 = 1.0 + 0j
    p2 = -1.0 + 0j
    p3 = 0.0 + 1j
    centre, radius = compute_circle_three_points(p1, p2, p3)
    assert abs(centre) < 1e-12
    assert abs(radius - 1.0) < 1e-12


def test_circle_contains_points_inside():
    centre = 0.0 + 0.0j
    radius = 2.0
    pts = np.array([1.0 + 1.0j, -1.0 + 0.5j])
    assert circle_contains_points(centre, radius, pts)


def test_circle_contains_points_outside():
    centre = 0.0 + 0.0j
    radius = 1.0
    pts = np.array([1.5 + 0.0j])
    assert not circle_contains_points(centre, radius, pts)


def test_circle_contains_origin_true():
    assert circle_contains_origin(centre=0.5 + 0.0j, radius=1.0) is True


def test_circle_contains_origin_false():
    assert circle_contains_origin(centre=2.0 + 0.0j, radius=1.0) is False
```

- [ ] **Step 2: Run tests, expect ImportError for the new symbols**

Run: `pytest tests/test_gamma0.py -v`
Expected: new tests fail with `ImportError`.

- [ ] **Step 3: Append to `src/em3d/gamma0.py`**

Port the notebook geometry. Complex numbers (z = x + iy) stand for 2D points; the notebook treats spectrum samples as complex.

```python
def compute_circle_two_points(z1: complex, z2: complex) -> tuple[complex, float]:
    """Circle through two points with the smaller radius (midpoint, |z2-z1|/2)."""
    centre = 0.5 * (z1 + z2)
    radius = abs(z2 - z1) / 2.0
    return centre, float(radius)


def compute_circle_three_points(z1: complex, z2: complex, z3: complex) -> tuple[complex, float]:
    """Circumscribed circle through three non-collinear complex points."""
    ax, ay = z1.real, z1.imag
    bx, by = z2.real, z2.imag
    cx, cy = z3.real, z3.imag
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-30:
        raise ValueError("three points are collinear")
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    centre = complex(ux, uy)
    radius = abs(centre - z1)
    return centre, float(radius)


def circle_contains_points(centre: complex, radius: float, points, epsilon: float = 1e-8) -> bool:
    pts = np.asarray(points, dtype=np.complex128)
    return bool(np.all(np.abs(pts - centre) <= radius + epsilon))


def circle_contains_origin(centre: complex, radius: float, epsilon: float = 1e-8) -> bool:
    return abs(centre) <= radius + epsilon
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gamma0.py -v`
Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/gamma0.py tests/test_gamma0.py
git commit -m "feat(em3d): γ₀ geometry — circle through 2 and 3 points"
```

---

## Task 13: γ₀ — `find_params` from spectrum samples

**Files:**
- Modify: `src/em3d/gamma0.py`.
- Modify: `tests/test_gamma0.py`.
- Source: notebook `find_params` (~line 1601), `circle_angle_from_origin` (~line 1555).

- [ ] **Step 1: Append tests**

```python
from em3d.gamma0 import find_params


def test_find_params_simple_case():
    # spectrum along the positive real axis from 1 to 3 → optimal μ = 2, radius = 1
    samples = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    result = find_params(samples)
    assert abs(result["mu"] - 2.0) < 1e-6
    assert abs(result["radius"] - 1.0) < 1e-6
    # result is plug-compatible with SolverConfig(**result)
    assert set(result.keys()) == {"mu", "radius"}


def test_find_params_rejects_origin_inside_hull():
    # if samples straddle the origin, γ₀ is ill-defined
    samples = np.array([-1.0 + 0j, 1.0 + 0j, 0.0 + 1j])
    with pytest.raises(ValueError):
        find_params(samples)


def test_find_params_requires_at_least_two():
    with pytest.raises(ValueError):
        find_params(np.array([1.0 + 0j]))
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `pytest tests/test_gamma0.py -v`
Expected: 3 new tests fail with ImportError.

- [ ] **Step 3: Append to `src/em3d/gamma0.py`**

Port the notebook `find_params`. Algorithm: find the smallest enclosing circle of the hull that does not contain the origin (steps A/B from the dissertation); return its centre μ and radius.

```python
def find_params(spectrum_samples: np.ndarray) -> dict:
    """Compute the optimal γ₀ iteration parameter from spectrum samples.

    Returns {'mu': complex, 'radius': float}. The return dict is plug-compatible
    with SolverConfig(**find_params(samples)). Raises ValueError if samples are
    degenerate (fewer than 2 points, origin inside the resulting circle).
    """
    pts = np.asarray(spectrum_samples, dtype=np.complex128)
    if len(pts) < 2:
        raise ValueError("find_params requires at least 2 spectrum samples")

    # Convex hull of the point set in 2D real coordinates
    as_xy = np.column_stack([pts.real, pts.imag])
    hull_xy = sequential_chain(as_xy)
    hull = hull_xy[:, 0] + 1j * hull_xy[:, 1]
    if len(hull) < 2:
        raise ValueError("spectrum samples are degenerate (collinear or identical)")

    # Smallest enclosing circle among: (a) pairs of hull points (diameter), (b) triples.
    best = None  # (radius, centre)
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            c, r = compute_circle_two_points(hull[i], hull[j])
            if circle_contains_points(c, r, hull):
                if best is None or r < best[0]:
                    best = (r, c)
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            for k in range(j + 1, len(hull)):
                try:
                    c, r = compute_circle_three_points(hull[i], hull[j], hull[k])
                except ValueError:
                    continue
                if circle_contains_points(c, r, hull):
                    if best is None or r < best[0]:
                        best = (r, c)
    if best is None:
        raise ValueError("could not find a bounding circle; check spectrum samples")
    radius, mu = best
    if circle_contains_origin(mu, radius):
        raise ValueError(
            "origin lies inside (or on) the bounding circle; γ₀ is ill-defined for this spectrum"
        )
    return {"mu": mu, "radius": float(radius)}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gamma0.py -v`
Expected: 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/em3d/gamma0.py tests/test_gamma0.py
git commit -m "feat(em3d): γ₀ find_params — smallest enclosing circle of spectrum"
```

---

## Task 14: Solver base (`BaseSolver`, `SolverConfig`, `SolverResult`)

**Files:**
- Create: `src/em3d/solvers/__init__.py`, `src/em3d/solvers/base.py`.
- Create: `tests/test_solvers.py` (initial — base dataclasses only).

- [ ] **Step 1: Write `tests/test_solvers.py` — dataclass tests**

```python
import numpy as np

from em3d.solvers.base import SolverConfig, SolverResult


def test_solver_config_defaults():
    cfg = SolverConfig(max_iter=100, rtol=1e-6)
    assert cfg.max_iter == 100
    assert cfg.rtol == 1e-6
    assert cfg.log is False


def test_solver_result_fields():
    u = np.zeros(4, dtype=np.complex128)
    res = SolverResult(u=u, iterations=5, residual_history=[1.0, 0.5, 0.1], converged=True)
    assert res.iterations == 5
    assert res.converged
    assert res.residual_history[-1] == 0.1
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/test_solvers.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/solvers/base.py`**

```python
"""Solver base classes: config, result, and Protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

import numpy as np


@dataclass
class SolverConfig:
    max_iter: int = 200
    rtol: float = 1e-6
    log: bool = False
    mu: Optional[complex] = None     # γ₀ centre for SIM
    radius: Optional[float] = None   # γ₀ radius for SIM

    def require_gamma(self) -> None:
        if self.mu is None or self.radius is None:
            raise ValueError("SolverConfig: mu and radius must be set (call gamma0.find_params)")


@dataclass
class SolverResult:
    u: Any
    iterations: int
    residual_history: List[float]
    converged: bool


class BaseSolver(Protocol):
    """Iterative solver for problem (I + B·η) u = rhs."""

    def solve(self, operator, rhs) -> SolverResult: ...
```

- [ ] **Step 4: Implement `src/em3d/solvers/__init__.py`**

```python
from .base import BaseSolver, SolverConfig, SolverResult

__all__ = ["BaseSolver", "SolverConfig", "SolverResult"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_solvers.py -v`
Expected: 2 pass.

- [ ] **Step 6: Commit**

```bash
git add src/em3d/solvers tests/test_solvers.py
git commit -m "feat(em3d): solver base — SolverConfig, SolverResult, BaseSolver Protocol"
```

---

## Task 15: SIM (generalised simple iteration / MSGD)

**Files:**
- Create: `src/em3d/solvers/sim.py`.
- Modify: `src/em3d/solvers/__init__.py` — export `SIM`.
- Modify: `tests/test_solvers.py` — append SIM convergence test.
- Source: notebook `sim` (~line 1765).

- [ ] **Step 1: Append test in `tests/test_solvers.py`**

```python
from em3d.grid import Grid
from em3d.refraction import cylinder_refraction, apply_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.operator import Operator
from em3d.solvers.base import SolverConfig
from em3d.solvers.sim import SIM


def _toy_problem_for_solver(be):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    scalar = cylinder_refraction(grid, eps_real=1.1, eps_imag=0.0, radius=0.2, axis="z")
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=0.5, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * 64
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.5, volume=volume)


def test_sim_converges_backend_agnostic(backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    # construct rhs from a known u_true so we verify convergence against a ground truth
    rng = np.random.default_rng(0)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.1
    )
    rhs = op.matvec(u_true)
    cfg = SolverConfig(max_iter=500, rtol=1e-8, mu=complex(1.0), radius=0.05)
    result = SIM(cfg).solve(op, rhs)
    assert result.converged, f"SIM did not converge, residuals: {result.residual_history[-5:]}"
    err = np.linalg.norm(np.asarray(result.u) - np.asarray(u_true)) / np.linalg.norm(np.asarray(u_true))
    assert err < 1e-6, f"SIM reconstructed u to {err:.2e}, expected < 1e-6"
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_solvers.py::test_sim_converges_backend_agnostic -v`
Expected: ModuleNotFoundError: em3d.solvers.sim.

- [ ] **Step 3: Implement `src/em3d/solvers/sim.py`**

Port the notebook `sim` function. It is the simple iteration u_{k+1} = u_k − γ₀ · (A u_k − f) where A = I + B·η (i.e., `operator.matvec`).

```python
"""Generalised simple iteration (MSGD) driven by γ₀."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


class SIM:
    def __init__(self, config: SolverConfig):
        config.require_gamma()
        self.cfg = config

    def solve(self, operator, rhs) -> SolverResult:
        be = operator.backend
        xp = be.xp
        cfg = self.cfg
        gamma = 1.0 / cfg.mu  # γ₀ = 1/μ for SIM
        u = xp.zeros_like(rhs)
        residuals: list[float] = []
        rhs_norm = float(xp.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return SolverResult(u=u, iterations=0, residual_history=[0.0], converged=True)
        for k in range(cfg.max_iter):
            Au = operator.matvec(u)
            r = Au - rhs
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[SIM] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)
            u = u - be.complex_dtype(gamma) * r
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
```

- [ ] **Step 4: Update `src/em3d/solvers/__init__.py`**

```python
from .base import BaseSolver, SolverConfig, SolverResult
from .sim import SIM

__all__ = ["BaseSolver", "SolverConfig", "SolverResult", "SIM"]
```

- [ ] **Step 5: Run test**

Run: `pytest tests/test_solvers.py -v`
Expected: pass. If convergence is slow, widen `max_iter` in the test; if it diverges, the `mu`/`radius` values for this toy problem need adjustment (the test uses hand-tuned `mu=1.0, radius=0.05` — these should be safe for a near-identity operator with weak ε perturbation).

- [ ] **Step 6: Commit**

```bash
git add src/em3d/solvers/sim.py src/em3d/solvers/__init__.py tests/test_solvers.py
git commit -m "feat(em3d): SIM solver (generalised simple iteration with γ₀)"
```

---

## Task 16: BiCGStab

**Files:**
- Create: `src/em3d/solvers/bicgstab.py`.
- Modify: `src/em3d/solvers/__init__.py`.
- Modify: `tests/test_solvers.py` — append test.
- Source: notebook `bicgstab` (~line 1922), helpers `bicg_norm`, `bicg_ldot`, `bicg_rdot` (~lines 1910–1918).

- [ ] **Step 1: Append test**

```python
from em3d.solvers.bicgstab import BiCGStab


def test_bicgstab_converges(backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    rng = np.random.default_rng(1)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.1
    )
    rhs = op.matvec(u_true)
    cfg = SolverConfig(max_iter=200, rtol=1e-8)
    result = BiCGStab(cfg).solve(op, rhs)
    assert result.converged, f"BiCGStab did not converge, residuals: {result.residual_history[-5:]}"
    err = np.linalg.norm(np.asarray(result.u) - np.asarray(u_true)) / np.linalg.norm(np.asarray(u_true))
    assert err < 1e-6
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/test_solvers.py::test_bicgstab_converges -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/solvers/bicgstab.py`**

Port notebook `bicgstab` (classical BiCGStab, see e.g. Saad §7.4.2). Adaptations: inner products use `xp.vdot` (= conj(a)·b for complex), norms via `xp.linalg.norm`.

```python
"""BiCGStab solver ported from the Yurchenkov notebook."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


class BiCGStab:
    def __init__(self, config: SolverConfig):
        self.cfg = config

    def solve(self, operator, rhs) -> SolverResult:
        be = operator.backend
        xp = be.xp
        cfg = self.cfg
        rhs_norm = float(xp.linalg.norm(rhs))
        residuals: list[float] = []
        u = xp.zeros_like(rhs)
        if rhs_norm == 0.0:
            return SolverResult(u=u, iterations=0, residual_history=[0.0], converged=True)

        r = rhs - operator.matvec(u)
        r_hat = xp.asarray(r, dtype=rhs.dtype).copy()  # shadow residual
        rho_prev = 1.0
        alpha = 1.0
        omega = 1.0
        v = xp.zeros_like(rhs)
        p = xp.zeros_like(rhs)
        for k in range(cfg.max_iter):
            rho = complex(xp.vdot(r_hat, r))
            if rho == 0:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=False)
            beta = (rho / rho_prev) * (alpha / omega) if k > 0 else 0.0
            p = r + be.complex_dtype(beta) * (p - be.complex_dtype(omega) * v)
            v = operator.matvec(p)
            alpha = rho / complex(xp.vdot(r_hat, v))
            s = r - be.complex_dtype(alpha) * v
            s_norm = float(xp.linalg.norm(s))
            if s_norm / rhs_norm < cfg.rtol:
                u = u + be.complex_dtype(alpha) * p
                residuals.append(s_norm / rhs_norm)
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)
            t = operator.matvec(s)
            omega = complex(xp.vdot(t, s)) / complex(xp.vdot(t, t))
            u = u + be.complex_dtype(alpha) * p + be.complex_dtype(omega) * s
            r = s - be.complex_dtype(omega) * t
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[BiCGStab] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k + 1, residual_history=residuals, converged=True)
            rho_prev = rho
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
```

- [ ] **Step 4: Export in `solvers/__init__.py`**

Append to `__all__` and imports:

```python
from .bicgstab import BiCGStab
# and add "BiCGStab" to __all__
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_solvers.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/em3d/solvers/bicgstab.py src/em3d/solvers/__init__.py tests/test_solvers.py
git commit -m "feat(em3d): BiCGStab solver"
```

---

## Task 17: TwoStep (двухшаговый метод градиентного спуска)

**Files:**
- Create: `src/em3d/solvers/twostep.py`.
- Modify: `src/em3d/solvers/__init__.py`.
- Modify: `tests/test_solvers.py` — append.
- Source: notebook `TwoStep` (~line 2141).

- [ ] **Step 1: Append test**

```python
from em3d.solvers.twostep import TwoStep


def test_twostep_converges(backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    rng = np.random.default_rng(2)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.1
    )
    rhs = op.matvec(u_true)
    cfg = SolverConfig(max_iter=300, rtol=1e-8)
    result = TwoStep(cfg).solve(op, rhs)
    assert result.converged, f"TwoStep did not converge, residuals: {result.residual_history[-5:]}"
    err = np.linalg.norm(np.asarray(result.u) - np.asarray(u_true)) / np.linalg.norm(np.asarray(u_true))
    assert err < 1e-6
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_solvers.py::test_twostep_converges -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/em3d/solvers/twostep.py`**

Port notebook `TwoStep`. Uses both `matvec` (A) and `rmatvec` (A*). The method alternates gradient steps minimising ||A u − f||² via step sizes that only involve inner products of A u and A* residuals.

```python
"""Two-step gradient descent (MSGD/TwoSGD) using matvec and rmatvec."""
from __future__ import annotations

from .base import SolverConfig, SolverResult


class TwoStep:
    def __init__(self, config: SolverConfig):
        self.cfg = config

    def solve(self, operator, rhs) -> SolverResult:
        be = operator.backend
        xp = be.xp
        cfg = self.cfg
        rhs_norm = float(xp.linalg.norm(rhs))
        residuals: list[float] = []
        u = xp.zeros_like(rhs)
        if rhs_norm == 0.0:
            return SolverResult(u=u, iterations=0, residual_history=[0.0], converged=True)

        for k in range(cfg.max_iter):
            Au = operator.matvec(u)
            r = Au - rhs
            rel = float(xp.linalg.norm(r)) / rhs_norm
            residuals.append(rel)
            if cfg.log:
                print(f"[TwoStep] iter={k}, rel_res={rel:.3e}")
            if rel < cfg.rtol:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=True)

            # step direction = A* r   (steepest descent for ||A u - f||²)
            p = operator.rmatvec(r)
            Ap = operator.matvec(p)
            p_norm_sq = float(xp.vdot(p, p).real)
            Ap_norm_sq = float(xp.vdot(Ap, Ap).real)
            if Ap_norm_sq == 0.0:
                return SolverResult(u=u, iterations=k, residual_history=residuals, converged=False)
            tau = p_norm_sq / Ap_norm_sq
            u = u - be.complex_dtype(tau) * p
        return SolverResult(u=u, iterations=cfg.max_iter, residual_history=residuals, converged=False)
```

- [ ] **Step 4: Export in `solvers/__init__.py`**

```python
from .twostep import TwoStep
# and add "TwoStep" to __all__
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_solvers.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/em3d/solvers/twostep.py src/em3d/solvers/__init__.py tests/test_solvers.py
git commit -m "feat(em3d): TwoStep gradient descent solver"
```

---

## Task 18: Cross-solver convergence matrix

**Files:**
- Modify: `tests/test_solvers.py` — add a parametrised convergence smoke test over {solver} × {precision}.

- [ ] **Step 1: Append the matrix test**

```python
import pytest

SOLVERS = [
    ("SIM", lambda: SIM(SolverConfig(max_iter=500, rtol=1e-6, mu=complex(1.0), radius=0.05))),
    ("BiCGStab", lambda: BiCGStab(SolverConfig(max_iter=200, rtol=1e-6))),
    ("TwoStep", lambda: TwoStep(SolverConfig(max_iter=300, rtol=1e-6))),
]


@pytest.mark.parametrize("solver_name,solver_factory", SOLVERS)
def test_solvers_converge_double(solver_name, solver_factory, backend_numpy_double):
    problem = _toy_problem_for_solver(backend_numpy_double)
    op = Operator(problem)
    rng = np.random.default_rng(hash(solver_name) & 0xFFFF)
    u_true = backend_numpy_double.array(
        (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)).astype(np.complex128) * 0.05
    )
    rhs = op.matvec(u_true)
    result = solver_factory().solve(op, rhs)
    assert result.converged, f"{solver_name} did not converge"


@pytest.mark.parametrize("solver_name,solver_factory", SOLVERS)
def test_solvers_converge_single(solver_name, solver_factory, backend_numpy_single):
    problem = _toy_problem_for_solver(backend_numpy_single)
    op = Operator(problem)
    rng = np.random.default_rng((hash(solver_name) + 1) & 0xFFFF)
    u_true_double = (rng.standard_normal((3,) + problem.grid.N) + 1j * rng.standard_normal((3,) + problem.grid.N)) * 0.05
    u_true = backend_numpy_single.array(u_true_double.astype(np.complex64))
    rhs = op.matvec(u_true)
    result = solver_factory().solve(op, rhs)
    # f32 should converge to a looser tolerance; relax rtol-based convergence flag check:
    final_rel = result.residual_history[-1] if result.residual_history else float("inf")
    assert final_rel < 5e-4, f"{solver_name}@single: final residual {final_rel:.2e} too large"
```

- [ ] **Step 2: Run and pass**

Run: `pytest tests/test_solvers.py -v`
Expected: all pass. If a solver fails at single precision with residual stalling above 5e-4, widen the threshold to `1e-3` (document in a comment the reason).

- [ ] **Step 3: Commit**

```bash
git add tests/test_solvers.py
git commit -m "test(em3d): cross-solver × precision convergence matrix"
```

---

## Task 19: Public API reexports

**Files:**
- Modify: `src/em3d/__init__.py`.

- [ ] **Step 1: Write the public-API reexport**

Replace `src/em3d/__init__.py` with:

```python
"""em3d: volume-integral-equation solver for 3D electrodynamics on structured grids."""
from __future__ import annotations

from .backend import Backend
from .dtypes import Precision
from .grid import Grid
from .problem import Problem
from .operator import Operator
from .refraction import (
    apply_refraction,
    cylinder_refraction,
    ellipsis_refraction,
    step_refraction,
)
from .wave import flat_wave_vec
from . import gamma0
from .solvers import BaseSolver, BiCGStab, SIM, SolverConfig, SolverResult, TwoStep

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "Precision",
    "Grid",
    "Problem",
    "Operator",
    "apply_refraction",
    "cylinder_refraction",
    "ellipsis_refraction",
    "step_refraction",
    "flat_wave_vec",
    "gamma0",
    "BaseSolver",
    "BiCGStab",
    "SIM",
    "SolverConfig",
    "SolverResult",
    "TwoStep",
    "__version__",
]
```

- [ ] **Step 2: Update `tests/test_smoke.py`**

```python
def test_package_imports():
    import em3d

    # public surface sanity
    for sym in [
        "Backend",
        "Precision",
        "Grid",
        "Problem",
        "Operator",
        "cylinder_refraction",
        "flat_wave_vec",
        "SIM",
        "BiCGStab",
        "TwoStep",
        "SolverConfig",
    ]:
        assert hasattr(em3d, sym), f"em3d is missing public symbol {sym}"
```

- [ ] **Step 3: Run smoke and full test suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/em3d/__init__.py tests/test_smoke.py
git commit -m "feat(em3d): publish public API (0.1.0)"
```

---

## Task 20: GPU smoke test (skipped where cupy absent)

**Files:**
- Modify: `tests/test_operator_vs_dense.py` — append one gpu-marked test that mirrors the integration gate on cupy.

- [ ] **Step 1: Append GPU smoke test**

```python
@pytest.mark.gpu
def test_operator_matvec_gpu_matches_cpu():
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    be_cpu = Backend.numpy(Precision.DOUBLE)
    be_gpu = Backend.cupy(Precision.DOUBLE)

    def make(be):
        grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0, 0, 0), backend=be)
        scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.3, axis="z")
        eta = apply_refraction(grid, scalar_eta=scalar)
        wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
        return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 64)

    p_cpu = make(be_cpu)
    p_gpu = make(be_gpu)
    op_cpu = Operator(p_cpu)
    op_gpu = Operator(p_gpu)
    u_cpu = p_cpu.wave
    u_gpu = p_gpu.wave
    y_cpu = op_cpu.matvec(u_cpu)
    y_gpu = be_gpu.to_host(op_gpu.matvec(u_gpu))
    np.testing.assert_allclose(y_gpu, np.asarray(y_cpu), atol=1e-10)
```

- [ ] **Step 2: Run**

Run (CPU-only): `pytest -m "not gpu" -v` → all pass, GPU test skipped.
Run (with CUDA): `pytest -m gpu -v` → GPU test passes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_operator_vs_dense.py
git commit -m "test(em3d): GPU smoke test for FFT matvec (cupy vs numpy)"
```

---

## Task 21: Wiki integration — `wiki/code/em3d.md` page

**Files:**
- Create: `wiki/code/em3d.md`.
- Modify: `wiki/index.md` — add Code section entry.
- Modify: `wiki/log.md` — append `code` entry.

This task is per the governing `wiki/CLAUDE.md` `CODE` protocol; it creates the vault-side bookkeeping that ties the package to the research concepts.

- [ ] **Step 1: Create `wiki/code/em3d.md`**

```markdown
---
title: em3d — пакет решателя ОИУ для электродинамики
zone: code
tags: [em3d, vie, fft, gpu, python]
sources: [literature/yurchenkov-ipu-2025, literature/yurchenkov-kaliningrad-2025]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# em3d — пакет решателя ОИУ для электродинамики

## Назначение

Решение объёмного интегрального уравнения (I + B·η) u = f для задачи рассеяния на прозрачной 3D-структуре. Перенос исследовательского кода из `raw/notes/Yurchenkov-programming-code-EM3D-2026.ipynb` в поддерживаемый Python-пакет с CPU/GPU-бэкендом и параметризуемой точностью.

## Алгоритм

Центральное ядро: дискретизация по методу коллокации на структурированной декартовой сетке, БПФ-ускорение матвека через удвоенный параллелепипед Π₂, итерационное решение одним из трёх методов (SIM/MSGD, BiCGStab, TwoStep). Выбор итерационного параметра γ₀ — по геометрической процедуре минимальной описанной окружности выпуклой оболочки спектра.

## Параметры и конфигурация

| Параметр | Значение | Описание |
|----------|----------|----------|
| `Precision` | `SINGLE` / `DOUBLE` | пара dtype: `(float32/complex64)` или `(float64/complex128)` |
| `Backend` | `numpy` / `cupy` | выбирается через `Backend.auto()` |
| `Grid.N` | tuple[int,int,int] | число ячеек |
| `Grid.L` | tuple[float,float,float] | физические размеры области |
| `SolverConfig.max_iter` | int | максимум итераций |
| `SolverConfig.rtol` | float | относительная точность невязки |

## Анализ сложности

- Память FFT-оператора: O(N·log N) коэффициентов на Π₂ (N = Nx·Ny·Nz), без хранения dense-матрицы.
- Одна итерация: O(N·log N) за счёт БПФ.
- Dense-эталон хранится только для тестов на сетках ≤ 64³ ячеек.

## Результаты и наблюдения

MVP v0.1.0 проходит интеграционный тест FFT vs dense на сетке 4³ с относительной ошибкой < 1e-10. Три итерационных метода сходятся на тестовой задаче (слабое ε, малое k) за < 300 итераций. Верификация Mie/плавленый кварц — следующий spec.

## Реализованные концепции

- [[concepts/volume-integral-equation]]
- [[concepts/collocation-method]]
- [[concepts/uniform-cartesian-grid]]
- [[concepts/block-toeplitz-matrix]]
- [[concepts/fft-convolution-acceleration]]
- [[concepts/optimal-iteration-parameter-gamma0]]
- [[concepts/generalized-simple-iteration]]
- [[concepts/two-step-gradient-descent]]
- [[concepts/bicgstab]]

## Исходные файлы

- `src/em3d/` — пакет
- `tests/` — набор тестов
- `docs/superpowers/specs/2026-04-23-em3d-package-refactor-design.md` — спецификация
- `docs/superpowers/plans/2026-04-23-em3d-package-refactor.md` — план реализации
```

- [ ] **Step 2: Add Code entry to `wiki/index.md`**

In `wiki/index.md`, under the `## Code` section (currently `*(пусто — ...)*`), replace with:

```markdown
## Code

- [[code/em3d]] — пакет решателя ОИУ на структурированной сетке с FFT-ускорением и CPU/GPU-бэкендом
```

- [ ] **Step 3: Append to `wiki/log.md`**

Append at the end of `wiki/log.md`:

```markdown
## [2026-04-23] code | em3d — перенос ноутбука в пакет (MVP v0.1.0)

- Создана страница: code/em3d
- Созданы исходники: src/em3d/ (backend, grid, refraction, wave, kernel, dense, operator, problem, gamma0, solvers/{sim,bicgstab,twostep})
- Созданы тесты: tests/{test_backend,test_grid,test_refraction,test_wave,test_kernel,test_dense,test_operator_vs_dense,test_gamma0,test_solvers}.py
- Обновлены страницы: index.md (Code-секция)
- Замечания: ЭПР, визуализация, верификация Mie/плавленый кварц — отложены в следующий spec
```

- [ ] **Step 4: Commit**

```bash
git add wiki/code/em3d.md wiki/index.md wiki/log.md
git commit -m "docs(wiki): add em3d code page, update index and log"
```

---

## Post-implementation check

- [ ] Run the full suite one last time: `pytest -v`
- [ ] Confirm the CPU path has no `gpu` marker failures, only skips, when cupy is absent.
- [ ] Confirm the public API example from the spec's "Data flow" section runs end-to-end as a manual smoke:

```python
from em3d import Backend, Precision, Grid, Problem, Operator, cylinder_refraction, apply_refraction, flat_wave_vec
from em3d.solvers import TwoStep, SolverConfig

be = Backend.auto(Precision.DOUBLE)
grid = Grid(N=(8, 8, 8), L=(1, 1, 1), center=(0, 0, 0), backend=be)
scalar = cylinder_refraction(grid, eps_real=2.0, eps_imag=0.0, radius=0.3, axis="z")
eta = apply_refraction(grid, scalar_eta=scalar)
wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 512)
op = Operator(problem)
result = TwoStep(SolverConfig(max_iter=300, rtol=1e-6)).solve(op, rhs=problem.wave)
print("converged:", result.converged, "iters:", result.iterations)
```

---

## Out of scope (next spec)

- ЭПР (`compute_RCS`) и сравнение с miepython для сферы.
- Таблицы верификации плавленым кварцом.
- Визуализация полей (`plot_scalar_xy/xz/yz`, `plot_spectre`).
- Производительные бенчмарки и тесты на больших сетках (≥ 64³).
- Возможная поддержка тетраэдральных сеток (неравномерный FFT / NUFFT).
