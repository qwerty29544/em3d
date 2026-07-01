# Acoustic Scattering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать отдельный подпакет `em3d.acoustics` для скалярного акустического рассеяния с FFT-оператором, dense-верификацией, итерационными методами, `gamma0`, диаграммой рассеяния, визуализациями, пакетным экспериментом и notebook-запуском.

**Architecture:** Акустическая часть отделяется от существующего диадического электродинамического `Operator`: новый scalar operator работает с массивами формы `(Nx, Ny, Nz)` и реализует `H u = u - B((eta - 1) * u)`. Существующие `SIM`, `BiCGStab`, `TwoStep`, `Grid`, `Backend`, `gamma0.analyze_spectrum` и экспериментальный стиль переиспользуются через стабильные интерфейсы.

**Tech Stack:** Python 3.11+, NumPy >= 1.26, SciPy >= 1.11, pytest, optional matplotlib via `em3d[vis]`, existing `py` interpreter on Windows.

## Global Constraints

- Публичный acoustic API принимает `eta`, а оператор внутри всегда использует `chi = eta - 1`.
- `eta = 1` означает фоновую среду без рассеяния.
- Не менять существующий `src/em3d/operator.py` и поведение электродинамического `em3d.Operator`, кроме импорта нового acoustic подпакета в `src/em3d/__init__.py`.
- Акустическое поле имеет форму `(Nx, Ny, Nz)`, не `(3, Nx, Ny, Nz)`.
- Acoustic `rmatvec` обязан быть настоящим сопряжённым оператором для корректной работы `TwoStep`.
- Dense matrix и FFT matvec должны совпадать на малых сетках.
- Все команды проверки выполнять через системный Python `py`.
- Перед тестами в PowerShell задавать `PYTHONPATH`: `$env:PYTHONPATH=(Resolve-Path 'src').Path`.

---

## File Structure

### New package files

- `src/em3d/acoustics/__init__.py` — публичный экспорт acoustic API.
- `src/em3d/acoustics/problem.py` — `AcousticProblem`, нормализация `direction`, построение падающей волны, соглашение `chi = eta - 1`.
- `src/em3d/acoustics/materials.py` — генераторы scalar `eta` для homogeneous/slab/sphere/ellipsoid/lattice.
- `src/em3d/acoustics/kernel.py` — scalar Helmholtz kernel on doubled grid и self-cell coefficient.
- `src/em3d/acoustics/dense.py` — dense scalar `B` и `H = I - B diag(chi)` для малых сеток.
- `src/em3d/acoustics/operator.py` — FFT-backed `AcousticOperator`.
- `src/em3d/acoustics/gamma0.py` — coarse scalar matrix и оценка `Gamma0Analysis`.
- `src/em3d/acoustics/farfield.py` — acoustic far-field amplitude и normalized scattering pattern.
- `src/em3d/acoustics/visualization.py` — scalar field slices и acoustic pattern plots.
- `src/em3d/experiments/acoustic_scattering.py` — пакетные сценарии, логи, CSV/JSON artifacts.

### Modified files

- `src/em3d/__init__.py` — добавить `from . import acoustics` и `"acoustics"` в `__all__`.
- `README.md` — добавить короткий пример запуска acoustic experiment после реализации.
- `wiki/code/acoustic-scattering.md` — добавить wiki-страницу по реализации.
- `wiki/index.md` — добавить ссылку на acoustic code page.
- `wiki/log.md` — добавить запись `code`.

### New tests

- `tests/test_acoustics_problem_materials.py`
- `tests/test_acoustics_operator.py`
- `tests/test_acoustics_gamma0_farfield.py`
- `tests/test_acoustic_experiment.py`

### New notebook

- `notebooks/acoustic-scattering-kaggle.ipynb`

---

## Task 1: Acoustic Problem And Materials

**Files:**
- Create: `src/em3d/acoustics/__init__.py`
- Create: `src/em3d/acoustics/problem.py`
- Create: `src/em3d/acoustics/materials.py`
- Modify: `src/em3d/__init__.py`
- Test: `tests/test_acoustics_problem_materials.py`

**Interfaces:**
- Consumes: `em3d.Grid`, `em3d.Backend`, `em3d.Precision`.
- Produces:
  - `AcousticProblem(grid: Grid, eta: object, wave: object, k0: float, volume: float)`
  - `AcousticProblem.chi -> object`
  - `plane_wave_scalar(grid, k, direction, amplitude=1.0)`
  - `make_acoustic_problem(grid, eta, k0, direction=(0,0,1), amplitude=1.0)`
  - `eta_homogeneous(grid, eta_value)`
  - `eta_slab(grid, eta_inside, eta_outside=1.0, axis=0, width_fraction=0.5)`
  - `eta_sphere(grid, center, radius, eta_inside, eta_outside=1.0)`
  - `eta_ellipsoid(grid, center, radii, eta_inside, eta_outside=1.0)`

- [ ] **Step 1: Write failing problem/material tests**

Create `tests/test_acoustics_problem_materials.py`:

```python
import numpy as np
import pytest

import em3d
from em3d.acoustics import (
    AcousticProblem,
    eta_ellipsoid,
    eta_homogeneous,
    eta_slab,
    eta_sphere,
    make_acoustic_problem,
    plane_wave_scalar,
)


def _grid(N=(4, 4, 4)):
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


def test_acoustic_problem_eta_convention_and_wave_shape():
    grid = _grid()
    eta = eta_homogeneous(grid, 1.0 + 0.0j)
    problem = make_acoustic_problem(grid, eta, k0=2.0, direction=(0.0, 0.0, 2.0), amplitude=2.0 + 0.5j)
    assert isinstance(problem, AcousticProblem)
    assert problem.eta.shape == grid.N
    assert problem.wave.shape == grid.N
    assert problem.eta.dtype == grid.backend.complex_dtype
    assert problem.wave.dtype == grid.backend.complex_dtype
    assert np.allclose(problem.chi, 0.0)
    assert np.allclose(np.abs(problem.wave), abs(2.0 + 0.5j))


def test_plane_wave_scalar_uses_normalized_direction():
    grid = _grid()
    w1 = plane_wave_scalar(grid, k=3.0, direction=(0.0, 0.0, 1.0))
    w2 = plane_wave_scalar(grid, k=3.0, direction=(0.0, 0.0, 5.0))
    assert np.allclose(w1, w2)


def test_acoustic_problem_rejects_bad_eta_shape_and_zero_direction():
    grid = _grid()
    wave = plane_wave_scalar(grid, k=1.0, direction=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="eta.shape"):
        AcousticProblem(grid=grid, eta=np.ones((2, 2), dtype=np.complex128), wave=wave, k0=1.0, volume=1.0)
    with pytest.raises(ValueError, match="direction"):
        plane_wave_scalar(grid, k=1.0, direction=(0.0, 0.0, 0.0))


def test_material_generators_return_eta_not_chi():
    grid = _grid()
    homogeneous = eta_homogeneous(grid, 2.5 + 0.25j)
    assert np.allclose(homogeneous, 2.5 + 0.25j)

    slab = eta_slab(grid, eta_inside=3.0 + 1.0j, eta_outside=1.0, axis=0, width_fraction=0.5)
    assert set(np.unique(np.asarray(slab)).tolist()) == {1.0 + 0.0j, 3.0 + 1.0j}

    sphere = eta_sphere(grid, center=(0.0, 0.0, 0.0), radius=0.4, eta_inside=4.0, eta_outside=1.0)
    assert np.max(np.asarray(sphere).real) == 4.0
    assert np.min(np.asarray(sphere).real) == 1.0

    ellipsoid = eta_ellipsoid(grid, center=(0.0, 0.0, 0.0), radii=(0.5, 0.25, 0.25), eta_inside=2.0, eta_outside=1.0)
    assert ellipsoid.shape == grid.N
    assert np.count_nonzero(np.asarray(ellipsoid) == 2.0 + 0.0j) > 0
```

- [ ] **Step 2: Run tests and confirm import failure**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_problem_materials.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'em3d.acoustics'`.

- [ ] **Step 3: Implement `problem.py`**

Create `src/em3d/acoustics/problem.py`:

```python
"""Scalar acoustic scattering problem definitions."""
from __future__ import annotations

from dataclasses import dataclass

from ..grid import Grid


def _as_direction_tuple(direction) -> tuple[float, float, float]:
    values = tuple(float(x) for x in direction)
    if len(values) != 3:
        raise ValueError(f"direction must have length 3, got {direction!r}")
    return values


def _normalized_direction(direction, xp):
    d = xp.asarray(_as_direction_tuple(direction), dtype=xp.float64)
    norm = float(xp.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    return d / norm


def plane_wave_scalar(
    grid: Grid,
    *,
    k: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    amplitude: complex = 1.0,
):
    """Return scalar plane wave ``amplitude * exp(i k direction·x)``."""
    be = grid.backend
    xp = be.xp
    d = _normalized_direction(direction, xp)
    X, Y, Z = grid.coords()
    phase = d[0] * X + d[1] * Y + d[2] * Z
    return (be.complex_dtype(amplitude) * xp.exp(be.complex_dtype(1j * float(k)) * phase)).astype(
        be.complex_dtype,
        copy=False,
    )


@dataclass(frozen=True)
class AcousticProblem:
    """Grid, scalar eta, incident wave and wave number for acoustic scattering."""

    grid: Grid
    eta: object
    wave: object
    k0: float
    volume: float

    def __post_init__(self) -> None:
        be = self.grid.backend
        expected = self.grid.N
        if getattr(self.eta, "shape", None) != expected:
            raise ValueError(f"eta.shape {getattr(self.eta, 'shape', None)} != expected {expected}")
        if getattr(self.wave, "shape", None) != expected:
            raise ValueError(f"wave.shape {getattr(self.wave, 'shape', None)} != expected {expected}")
        if self.eta.dtype != be.complex_dtype:
            raise TypeError(f"eta.dtype {self.eta.dtype} != {be.complex_dtype}")
        if self.wave.dtype != be.complex_dtype:
            raise TypeError(f"wave.dtype {self.wave.dtype} != {be.complex_dtype}")
        if float(self.k0) <= 0.0:
            raise ValueError(f"k0 must be positive, got {self.k0}")
        if float(self.volume) <= 0.0:
            raise ValueError(f"volume must be positive, got {self.volume}")

    @property
    def backend(self):
        return self.grid.backend

    @property
    def chi(self):
        return (self.eta - self.backend.complex_dtype(1.0)).astype(self.backend.complex_dtype, copy=False)


def make_acoustic_problem(
    grid: Grid,
    eta,
    *,
    k0: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    amplitude: complex = 1.0,
) -> AcousticProblem:
    """Build an acoustic problem from eta and a scalar incident plane wave."""
    be = grid.backend
    eta_arr = be.asarray_of_kind(eta, "complex")
    wave = plane_wave_scalar(grid, k=float(k0), direction=direction, amplitude=amplitude)
    return AcousticProblem(grid=grid, eta=eta_arr, wave=wave, k0=float(k0), volume=grid.dv * int(grid.N[0] * grid.N[1] * grid.N[2]))
```

- [ ] **Step 4: Implement `materials.py`**

Create `src/em3d/acoustics/materials.py`:

```python
"""Scalar eta field generators for acoustic scattering."""
from __future__ import annotations

from ..grid import Grid


def _as_complex(be, value):
    return be.complex_dtype(value)


def eta_homogeneous(grid: Grid, eta_value):
    """Return scalar eta field with a constant value."""
    be = grid.backend
    out = be.zeros(grid.N, kind="complex")
    out[...] = _as_complex(be, eta_value)
    return out


def eta_slab(
    grid: Grid,
    *,
    eta_inside,
    eta_outside=1.0,
    axis: int = 0,
    width_fraction: float = 0.5,
):
    """Return eta field with a centered slab along one coordinate axis."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if not (0.0 < float(width_fraction) <= 1.0):
        raise ValueError(f"width_fraction must be in (0, 1], got {width_fraction!r}")
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    coords = (X, Y, Z)
    half_width = 0.5 * float(grid.L[axis]) * float(width_fraction)
    centre = float(grid.center[axis])
    mask = xp.abs(coords[axis] - centre) <= half_width
    return xp.where(mask, _as_complex(be, eta_inside), _as_complex(be, eta_outside)).astype(be.complex_dtype, copy=False)


def eta_sphere(
    grid: Grid,
    *,
    center: tuple[float, float, float],
    radius: float,
    eta_inside,
    eta_outside=1.0,
):
    """Return eta field with a spherical inclusion."""
    return eta_ellipsoid(
        grid,
        center=center,
        radii=(float(radius), float(radius), float(radius)),
        eta_inside=eta_inside,
        eta_outside=eta_outside,
    )


def eta_ellipsoid(
    grid: Grid,
    *,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    eta_inside,
    eta_outside=1.0,
):
    """Return eta field with an axis-aligned ellipsoidal inclusion."""
    if len(center) != 3:
        raise ValueError(f"center must have length 3, got {center!r}")
    if len(radii) != 3 or any(float(r) <= 0.0 for r in radii):
        raise ValueError(f"radii must contain three positive values, got {radii!r}")
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    cx, cy, cz = (float(x) for x in center)
    rx, ry, rz = (float(x) for x in radii)
    metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    return xp.where(metric <= 1.0, _as_complex(be, eta_inside), _as_complex(be, eta_outside)).astype(
        be.complex_dtype,
        copy=False,
    )
```

- [ ] **Step 5: Add package exports**

Create `src/em3d/acoustics/__init__.py`:

```python
"""Scalar acoustic scattering tools."""
from __future__ import annotations

from .materials import eta_ellipsoid, eta_homogeneous, eta_slab, eta_sphere
from .problem import AcousticProblem, make_acoustic_problem, plane_wave_scalar

__all__ = [
    "AcousticProblem",
    "eta_ellipsoid",
    "eta_homogeneous",
    "eta_slab",
    "eta_sphere",
    "make_acoustic_problem",
    "plane_wave_scalar",
]
```

Modify `src/em3d/__init__.py`:

```python
from . import acoustics
```

Add `"acoustics"` to `__all__`.

- [ ] **Step 6: Run Task 1 tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_problem_materials.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/em3d/acoustics/__init__.py src/em3d/acoustics/problem.py src/em3d/acoustics/materials.py src/em3d/__init__.py tests/test_acoustics_problem_materials.py
git commit -m "feat(acoustics): add scalar problem and materials"
```

---

## Task 2: Scalar Kernel And Dense Operator

**Files:**
- Create: `src/em3d/acoustics/kernel.py`
- Create: `src/em3d/acoustics/dense.py`
- Modify: `src/em3d/acoustics/__init__.py`
- Test: `tests/test_acoustics_operator.py`

**Interfaces:**
- Consumes: `AcousticProblem.chi`, `Grid`.
- Produces:
  - `self_cell_coefficient(k0: float, dv: float) -> complex`
  - `kernel_on_doubled_grid(grid, k0: float) -> object`
  - `B_scalar_matrix(grid, k0: float) -> np.ndarray`
  - `H_scalar_matrix(problem: AcousticProblem) -> np.ndarray`
  - `flatten_scalar_field(u) -> np.ndarray`
  - `unflatten_scalar_field(values, N) -> np.ndarray`

- [ ] **Step 1: Add failing kernel and dense tests**

Create or append to `tests/test_acoustics_operator.py`:

```python
import numpy as np

import em3d
from em3d.acoustics import eta_homogeneous, make_acoustic_problem
from em3d.acoustics.dense import B_scalar_matrix, H_scalar_matrix, flatten_scalar_field, unflatten_scalar_field
from em3d.acoustics.kernel import kernel_on_doubled_grid, self_cell_coefficient


def _grid(N=(3, 2, 2)):
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=N, L=(1.0, 0.8, 0.6), center=(0.0, 0.0, 0.0), backend=be)


def test_acoustic_self_cell_is_finite_and_low_frequency_small():
    value = self_cell_coefficient(k0=1e-3, dv=1e-6)
    assert np.isfinite(value.real)
    assert np.isfinite(value.imag)
    assert abs(value) < 1e-6


def test_acoustic_kernel_doubled_grid_shape_and_self_term():
    grid = _grid()
    kernel = kernel_on_doubled_grid(grid, k0=2.0)
    assert kernel.shape == (2 * grid.N[0], 2 * grid.N[1], 2 * grid.N[2])
    assert np.isclose(kernel[0, 0, 0], self_cell_coefficient(k0=2.0, dv=grid.dv))


def test_acoustic_dense_identity_when_eta_is_background():
    grid = _grid()
    eta = eta_homogeneous(grid, 1.0)
    problem = make_acoustic_problem(grid, eta, k0=1.5)
    H = H_scalar_matrix(problem)
    assert H.shape == (np.prod(grid.N), np.prod(grid.N))
    assert np.allclose(H, np.eye(H.shape[0]))


def test_acoustic_dense_flatten_roundtrip():
    grid = _grid()
    u = np.arange(np.prod(grid.N), dtype=np.complex128).reshape(grid.N)
    flat = flatten_scalar_field(u)
    assert flat.shape == (np.prod(grid.N),)
    assert np.allclose(unflatten_scalar_field(flat, grid.N), u)


def test_acoustic_dense_matrix_has_nonzero_scattering_for_eta_not_one():
    grid = _grid((2, 2, 2))
    eta = eta_homogeneous(grid, 2.0 + 0.1j)
    problem = make_acoustic_problem(grid, eta, k0=1.0)
    B = B_scalar_matrix(grid, k0=problem.k0)
    H = H_scalar_matrix(problem)
    assert B.shape == H.shape
    assert not np.allclose(H, np.eye(H.shape[0]))
```

- [ ] **Step 2: Run tests and confirm missing modules**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_operator.py -q
```

Expected: FAIL with missing `em3d.acoustics.kernel` or `em3d.acoustics.dense`.

- [ ] **Step 3: Implement scalar kernel**

Create `src/em3d/acoustics/kernel.py`:

```python
"""Scalar Helmholtz kernel coefficients for acoustic volume integral equations."""
from __future__ import annotations

from ..grid import Grid


def self_cell_coefficient(*, k0: float, dv: float) -> complex:
    """Equivalent-sphere approximation of ``k0^2 ∫cell G(r) dV`` at r=0."""
    import numpy as np

    k = float(k0)
    if k <= 0.0:
        raise ValueError(f"k0 must be positive, got {k0}")
    if float(dv) <= 0.0:
        raise ValueError(f"dv must be positive, got {dv}")
    a = (3.0 * float(dv) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return complex(np.exp(1j * k * a) * (1.0 - 1j * k * a) - 1.0)


def kernel_on_doubled_grid(grid: Grid, *, k0: float):
    """Sample scalar ``k0^2 G`` cell coefficients on a doubled grid."""
    be = grid.backend
    xp = be.xp
    Nx, Ny, Nz = grid.N
    dx = grid.L[0] / Nx
    dy = grid.L[1] / Ny
    dz = grid.L[2] / Nz
    sx = xp.concatenate([xp.arange(Nx) * dx, -(xp.arange(Nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(Ny) * dy, -(xp.arange(Ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(Nz) * dz, -(xp.arange(Nz, 0, -1)) * dz])
    SX, SY, SZ = xp.meshgrid(sx, sy, sz, indexing="ij")
    R = xp.sqrt(SX * SX + SY * SY + SZ * SZ)
    is_self = R < 1e-15
    R_safe = xp.where(is_self, xp.ones_like(R), R)
    ik = be.complex_dtype(1j * float(k0))
    green_coeff = (float(k0) ** 2) * grid.dv * xp.exp(ik * R_safe) / (4.0 * xp.pi * R_safe)
    self_value = be.complex_dtype(self_cell_coefficient(k0=float(k0), dv=grid.dv))
    out = xp.where(is_self, self_value, green_coeff)
    return out.astype(be.complex_dtype, copy=False)


def prep_coeffs_acoustic(grid: Grid, *, k0: float):
    """Return FFT of scalar acoustic kernel on the doubled grid."""
    return grid.backend.fftn(kernel_on_doubled_grid(grid, k0=k0), axes=(-3, -2, -1)).astype(
        grid.backend.complex_dtype,
        copy=False,
    )


def prep_conj_coeffs_acoustic(grid: Grid, *, k0: float):
    """Return FFT of conjugate scalar acoustic kernel for adjoint convolution."""
    be = grid.backend
    return be.fftn(be.xp.conj(kernel_on_doubled_grid(grid, k0=k0)), axes=(-3, -2, -1)).astype(
        be.complex_dtype,
        copy=False,
    )
```

- [ ] **Step 4: Implement dense scalar matrix**

Create `src/em3d/acoustics/dense.py`:

```python
"""Dense scalar acoustic operators for small-grid verification."""
from __future__ import annotations

import numpy as np

from ..grid import Grid
from .kernel import self_cell_coefficient
from .problem import AcousticProblem


def _cell_centres(grid: Grid) -> np.ndarray:
    be = grid.backend
    x = np.asarray(be.to_host(grid.x), dtype=np.float64)
    y = np.asarray(be.to_host(grid.y), dtype=np.float64)
    z = np.asarray(be.to_host(grid.z), dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def flatten_scalar_field(u) -> np.ndarray:
    """Flatten scalar field in row-major order."""
    return np.asarray(u, dtype=np.complex128).reshape(-1)


def unflatten_scalar_field(values, N: tuple[int, int, int]) -> np.ndarray:
    """Restore scalar field from row-major vector."""
    return np.asarray(values, dtype=np.complex128).reshape(tuple(N))


def B_scalar_matrix(grid: Grid, *, k0: float) -> np.ndarray:
    """Assemble dense scalar convolution matrix B without eta multiplication."""
    centres = _cell_centres(grid)
    n = centres.shape[0]
    B = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            R = float(np.linalg.norm(centres[i] - centres[j]))
            if R < 1e-15:
                B[i, j] = self_cell_coefficient(k0=float(k0), dv=grid.dv)
            else:
                B[i, j] = (float(k0) ** 2) * grid.dv * np.exp(1j * float(k0) * R) / (4.0 * np.pi * R)
    return B


def H_scalar_matrix(problem: AcousticProblem) -> np.ndarray:
    """Build dense H = I - B diag(eta - 1)."""
    B = B_scalar_matrix(problem.grid, k0=problem.k0)
    chi = flatten_scalar_field(problem.grid.backend.to_host(problem.chi))
    return np.eye(B.shape[0], dtype=np.complex128) - B @ np.diag(chi)
```

- [ ] **Step 5: Export kernel/dense helpers**

Modify `src/em3d/acoustics/__init__.py`:

```python
from .dense import B_scalar_matrix, H_scalar_matrix
from .kernel import kernel_on_doubled_grid, self_cell_coefficient
```

Add these names to `__all__`.

- [ ] **Step 6: Run Task 2 tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_operator.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/em3d/acoustics/__init__.py src/em3d/acoustics/kernel.py src/em3d/acoustics/dense.py tests/test_acoustics_operator.py
git commit -m "feat(acoustics): add scalar kernel and dense operator"
```

---

## Task 3: FFT Acoustic Operator

**Files:**
- Create: `src/em3d/acoustics/operator.py`
- Modify: `src/em3d/acoustics/__init__.py`
- Test: `tests/test_acoustics_operator.py`

**Interfaces:**
- Consumes: `prep_coeffs_acoustic`, `prep_conj_coeffs_acoustic`, `AcousticProblem.chi`, `H_scalar_matrix`.
- Produces:
  - `AcousticOperator(problem: AcousticProblem)`
  - `AcousticOperator.backend`
  - `AcousticOperator.matvec(u)`
  - `AcousticOperator.rmatvec(v)`
  - `AcousticOperator.to_dense()`

- [ ] **Step 1: Add failing FFT-vs-dense and adjoint tests**

Append to `tests/test_acoustics_operator.py`:

```python
from em3d.acoustics import AcousticOperator, eta_sphere


def test_acoustic_fft_operator_matches_dense_matvec():
    grid = _grid((3, 3, 2))
    eta = eta_sphere(grid, center=(0.0, 0.0, 0.0), radius=0.45, eta_inside=2.0 + 0.2j, eta_outside=1.0)
    problem = make_acoustic_problem(grid, eta, k0=1.2)
    operator = AcousticOperator(problem)
    rng = np.random.default_rng(123)
    u = rng.normal(size=grid.N) + 1j * rng.normal(size=grid.N)
    u = u.astype(np.complex128)

    y_fft = np.asarray(operator.matvec(u))
    H = H_scalar_matrix(problem)
    y_dense = H @ u.reshape(-1)
    assert np.allclose(y_fft.reshape(-1), y_dense, rtol=1e-12, atol=1e-12)


def test_acoustic_fft_operator_is_identity_for_eta_one():
    grid = _grid((3, 2, 2))
    problem = make_acoustic_problem(grid, eta_homogeneous(grid, 1.0), k0=1.0)
    operator = AcousticOperator(problem)
    u = np.ones(grid.N, dtype=np.complex128) * (2.0 - 0.5j)
    assert np.allclose(operator.matvec(u), u)
    assert np.allclose(operator.rmatvec(u), u)


def test_acoustic_rmatvec_is_adjoint():
    grid = _grid((3, 3, 2))
    eta = eta_sphere(grid, center=(0.0, 0.0, 0.0), radius=0.45, eta_inside=2.0 + 0.2j, eta_outside=1.0)
    problem = make_acoustic_problem(grid, eta, k0=1.2)
    operator = AcousticOperator(problem)
    rng = np.random.default_rng(456)
    u = (rng.normal(size=grid.N) + 1j * rng.normal(size=grid.N)).astype(np.complex128)
    v = (rng.normal(size=grid.N) + 1j * rng.normal(size=grid.N)).astype(np.complex128)
    left = np.vdot(operator.matvec(u), v)
    right = np.vdot(u, operator.rmatvec(v))
    assert np.allclose(left, right, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run tests and confirm missing operator**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_operator.py -q
```

Expected: FAIL with import error for `AcousticOperator`.

- [ ] **Step 3: Implement FFT operator**

Create `src/em3d/acoustics/operator.py`:

```python
"""FFT-backed scalar acoustic operator."""
from __future__ import annotations

import numpy as np

from .dense import H_scalar_matrix
from .kernel import prep_coeffs_acoustic, prep_conj_coeffs_acoustic
from .problem import AcousticProblem


def _pad_to_doubled(xp, u, N):
    """Zero-pad scalar field to doubled parallelepiped."""
    Nx, Ny, Nz = N
    out = xp.zeros((2 * Nx, 2 * Ny, 2 * Nz), dtype=u.dtype)
    out[:Nx, :Ny, :Nz] = u
    return out


def _crop_from_doubled(u_big, N):
    """Extract original scalar field from doubled parallelepiped."""
    Nx, Ny, Nz = N
    return u_big[:Nx, :Ny, :Nz]


class AcousticOperator:
    """FFT-backed acoustic operator ``H u = u - B((eta - 1)u)``."""

    def __init__(self, problem: AcousticProblem):
        self.problem = problem
        self._be = problem.grid.backend
        self._N = problem.grid.N
        self._K_hat = prep_coeffs_acoustic(problem.grid, k0=problem.k0)
        self._K_hat_conj = prep_conj_coeffs_acoustic(problem.grid, k0=problem.k0)
        self._chi_conj = self._be.xp.conj(problem.chi)

    @property
    def backend(self):
        return self._be

    def matvec(self, u):
        """Apply ``u - B(chi*u)`` to a scalar field."""
        be = self._be
        xp = be.xp
        u_arr = xp.asarray(u, dtype=be.complex_dtype)
        if u_arr.shape != self._N:
            raise ValueError(f"u.shape {u_arr.shape} != expected {self._N}")
        source = self.problem.chi * u_arr
        padded = _pad_to_doubled(xp, source, self._N)
        applied = be.ifftn(self._K_hat * be.fftn(padded, axes=(-3, -2, -1)), axes=(-3, -2, -1))
        return (u_arr - _crop_from_doubled(applied, self._N)).astype(be.complex_dtype, copy=False)

    def rmatvec(self, v):
        """Apply adjoint ``v - conj(chi)*B* v`` to a scalar field."""
        be = self._be
        xp = be.xp
        v_arr = xp.asarray(v, dtype=be.complex_dtype)
        if v_arr.shape != self._N:
            raise ValueError(f"v.shape {v_arr.shape} != expected {self._N}")
        padded = _pad_to_doubled(xp, v_arr, self._N)
        applied = be.ifftn(self._K_hat_conj * be.fftn(padded, axes=(-3, -2, -1)), axes=(-3, -2, -1))
        B_star_v = _crop_from_doubled(applied, self._N)
        return (v_arr - self._chi_conj * B_star_v).astype(be.complex_dtype, copy=False)

    def to_dense(self) -> np.ndarray:
        """Return dense H matrix; requires numpy backend."""
        if self._be.xp is not np:
            raise RuntimeError("AcousticOperator.to_dense requires numpy backend")
        return H_scalar_matrix(self.problem)
```

- [ ] **Step 4: Export operator**

Modify `src/em3d/acoustics/__init__.py`:

```python
from .operator import AcousticOperator
```

Add `"AcousticOperator"` to `__all__`.

- [ ] **Step 5: Run FFT operator tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_operator.py -q
```

Expected: PASS.

- [ ] **Step 6: Run regression tests for EM operator**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_operator_vs_dense.py tests/test_solvers.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/em3d/acoustics/__init__.py src/em3d/acoustics/operator.py tests/test_acoustics_operator.py
git commit -m "feat(acoustics): add fft scalar operator"
```

---

## Task 4: Solver Smoke Tests And Acoustic Gamma0

**Files:**
- Create: `src/em3d/acoustics/gamma0.py`
- Modify: `src/em3d/acoustics/__init__.py`
- Test: `tests/test_acoustics_gamma0_farfield.py`

**Interfaces:**
- Consumes: `AcousticProblem`, `H_scalar_matrix`, `em3d.gamma0.analyze_spectrum`, existing solvers.
- Produces:
  - `coarse_operator_matrix(problem, coarse_N=(4,4,4)) -> np.ndarray`
  - `estimate_from_problem(problem, coarse_N=(4,4,4)) -> Gamma0Analysis`
  - `find_params_from_problem(problem, coarse_N=(4,4,4)) -> dict`

- [ ] **Step 1: Add failing gamma0 and solver smoke tests**

Create `tests/test_acoustics_gamma0_farfield.py`:

```python
import numpy as np

import em3d
from em3d.acoustics import AcousticOperator, eta_sphere, make_acoustic_problem
from em3d.acoustics.gamma0 import coarse_operator_matrix, estimate_from_problem, find_params_from_problem


def _grid(N=(5, 5, 5)):
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


def _problem(N=(5, 5, 5), k0=1.0):
    grid = _grid(N)
    eta = eta_sphere(grid, center=(0.0, 0.0, 0.0), radius=0.35, eta_inside=1.4 + 0.15j, eta_outside=1.0)
    return make_acoustic_problem(grid, eta, k0=k0)


def test_acoustic_gamma0_coarse_matrix_and_analysis():
    problem = _problem()
    H = coarse_operator_matrix(problem, coarse_N=(3, 3, 3))
    assert H.shape == (27, 27)
    analysis = estimate_from_problem(problem, coarse_N=(3, 3, 3))
    assert analysis.coarse_N == (3, 3, 3)
    assert analysis.matrix_shape == H.shape
    assert np.isfinite(analysis.mu.real)
    assert np.isfinite(analysis.mu.imag)
    assert analysis.radius > 0.0
    params = find_params_from_problem(problem, coarse_N=3)
    assert set(params) == {"mu", "radius"}


def test_acoustic_existing_solvers_reduce_residual():
    problem = _problem(N=(4, 4, 4), k0=0.6)
    operator = AcousticOperator(problem)
    analysis = estimate_from_problem(problem, coarse_N=(3, 3, 3))
    solvers = [
        em3d.SIM(em3d.SolverConfig(max_iter=25, rtol=1e-8, **analysis.as_solver_config_kwargs())),
        em3d.BiCGStab(em3d.SolverConfig(max_iter=25, rtol=1e-8)),
        em3d.TwoStep(em3d.SolverConfig(max_iter=25, rtol=1e-8)),
    ]
    for solver in solvers:
        result = solver.solve(operator, problem.wave)
        assert result.residual_history
        assert result.residual_history[-1] < result.residual_history[0]
        assert result.u.shape == problem.grid.N
```

- [ ] **Step 2: Run tests and confirm missing gamma0**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_gamma0_farfield.py -q
```

Expected: FAIL with missing `em3d.acoustics.gamma0`.

- [ ] **Step 3: Implement acoustic gamma0**

Create `src/em3d/acoustics/gamma0.py`:

```python
"""Gamma0 estimation for scalar acoustic problems."""
from __future__ import annotations

import numpy as np

from ..backend import Backend
from ..dtypes import Precision
from ..gamma0 import analyze_spectrum
from ..grid import Grid
from .dense import H_scalar_matrix
from .problem import AcousticProblem, make_acoustic_problem


def _normalize_grid_shape(coarse_N) -> tuple[int, int, int]:
    if isinstance(coarse_N, int):
        shape = (int(coarse_N), int(coarse_N), int(coarse_N))
    else:
        shape = tuple(int(n) for n in coarse_N)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f"coarse_N must be a positive int or 3-tuple, got {coarse_N!r}")
    return shape


def _nearest_indices(source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    source = np.asarray(source_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    indices = np.searchsorted(source, target)
    indices = np.clip(indices, 0, len(source) - 1)
    left = np.clip(indices - 1, 0, len(source) - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(target - source[indices])
    return np.where(choose_left, left, indices)


def _resample_eta(problem: AcousticProblem, coarse_grid: Grid) -> np.ndarray:
    be = problem.grid.backend
    eta = np.asarray(be.to_host(problem.eta), dtype=np.complex128)
    ix = _nearest_indices(be.to_host(problem.grid.x), coarse_grid.x)
    iy = _nearest_indices(be.to_host(problem.grid.y), coarse_grid.y)
    iz = _nearest_indices(be.to_host(problem.grid.z), coarse_grid.z)
    return eta[ix, :, :][:, iy, :][:, :, iz]


def coarse_operator_matrix(problem: AcousticProblem, coarse_N=(4, 4, 4)) -> np.ndarray:
    """Build dense scalar H for a nearest-neighbour coarse acoustic problem."""
    shape = _normalize_grid_shape(coarse_N)
    coarse_backend = Backend.numpy(Precision.DOUBLE)
    coarse_grid = Grid(N=shape, L=problem.grid.L, center=problem.grid.center, backend=coarse_backend)
    eta = _resample_eta(problem, coarse_grid)
    coarse_problem = make_acoustic_problem(coarse_grid, eta, k0=problem.k0)
    return H_scalar_matrix(coarse_problem)


def estimate_from_problem(problem: AcousticProblem, coarse_N=(4, 4, 4)):
    """Estimate gamma0 from dense spectrum of a coarse scalar acoustic problem."""
    shape = _normalize_grid_shape(coarse_N)
    H = coarse_operator_matrix(problem, coarse_N=shape)
    spectrum = np.linalg.eigvals(H)
    return analyze_spectrum(spectrum, coarse_N=shape, matrix_shape=H.shape)


def find_params_from_problem(problem: AcousticProblem, coarse_N=(4, 4, 4)) -> dict:
    """Return SolverConfig-compatible gamma0 parameters."""
    return estimate_from_problem(problem, coarse_N=coarse_N).as_solver_config_kwargs()
```

- [ ] **Step 4: Export gamma0 module**

Modify `src/em3d/acoustics/__init__.py`:

```python
from . import gamma0
```

Add `"gamma0"` to `__all__`.

- [ ] **Step 5: Run gamma0 and solver tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_gamma0_farfield.py -q
```

Expected: PASS.

- [ ] **Step 6: Run solver regression tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_solvers.py tests/test_gamma0.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/em3d/acoustics/__init__.py src/em3d/acoustics/gamma0.py tests/test_acoustics_gamma0_farfield.py
git commit -m "feat(acoustics): add gamma0 and solver smoke tests"
```

---

## Task 5: Acoustic Far Field And Scalar Visualization

**Files:**
- Create: `src/em3d/acoustics/farfield.py`
- Create: `src/em3d/acoustics/visualization.py`
- Modify: `src/em3d/acoustics/__init__.py`
- Test: `tests/test_acoustics_gamma0_farfield.py`

**Interfaces:**
- Consumes: `AcousticProblem`, scalar solution `u`.
- Produces:
  - `farfield_amplitude(u, problem, directions, batch_size=64) -> np.ndarray`
  - `scattering_pattern(u, problem, directions, normalize="max") -> np.ndarray`
  - `pattern_plane(u, problem, plane="xy", n_angles=360, normalize="max") -> tuple[np.ndarray, np.ndarray]`
  - `plot_scalar_slices(u, grid, output_dir, prefix="acoustic", parts=("abs","real","imag","angle")) -> list[Path]`
  - `plot_pattern(phi, sigma, filename=None, polar=False, title=None) -> tuple`

- [ ] **Step 1: Add failing far-field tests**

Append to `tests/test_acoustics_gamma0_farfield.py`:

```python
from em3d.acoustics.farfield import farfield_amplitude, pattern_plane, scattering_pattern


def test_acoustic_farfield_zero_for_background_eta():
    grid = _grid((4, 4, 4))
    problem = make_acoustic_problem(grid, np.ones(grid.N, dtype=np.complex128), k0=1.0)
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    amp = farfield_amplitude(problem.wave, problem, directions)
    sigma = scattering_pattern(problem.wave, problem, directions)
    assert amp.shape == (2,)
    assert np.allclose(amp, 0.0)
    assert np.allclose(sigma, 0.0)


def test_acoustic_pattern_plane_shapes_and_normalization():
    problem = _problem(N=(4, 4, 4), k0=1.0)
    phi, sigma = pattern_plane(problem.wave, problem, plane="xy", n_angles=24, normalize="max")
    assert phi.shape == (24,)
    assert sigma.shape == (24,)
    assert np.all(np.isfinite(sigma))
    assert np.max(sigma) <= 1.0 + 1e-12
```

- [ ] **Step 2: Run tests and confirm missing farfield**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_gamma0_farfield.py -q
```

Expected: FAIL with missing `em3d.acoustics.farfield`.

- [ ] **Step 3: Implement far-field**

Create `src/em3d/acoustics/farfield.py`:

```python
"""Far-field diagnostics for scalar acoustic scattering."""
from __future__ import annotations

import numpy as np

from .problem import AcousticProblem


def _directions_array(directions) -> np.ndarray:
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError(f"directions must have shape (M, 3), got {dirs.shape}")
    norms = np.linalg.norm(dirs, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("directions must be non-zero")
    return dirs / norms[:, None]


def _plane_directions(plane: str, phi: np.ndarray) -> np.ndarray:
    if plane == "xy":
        return np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    if plane == "xz":
        return np.column_stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)])
    if plane == "yz":
        return np.column_stack([np.zeros_like(phi), np.cos(phi), np.sin(phi)])
    raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")


def farfield_amplitude(u, problem: AcousticProblem, directions, *, batch_size: int = 64) -> np.ndarray:
    """Return scalar acoustic far-field amplitude for observation directions."""
    dirs = _directions_array(directions)
    be = problem.grid.backend
    xp = be.xp
    u_arr = xp.asarray(u, dtype=be.complex_dtype)
    if u_arr.shape != problem.grid.N:
        raise ValueError(f"u.shape {u_arr.shape} != expected {problem.grid.N}")
    source = (problem.chi * u_arr).reshape(-1)
    X, Y, Z = problem.grid.coords()
    r_flat = xp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=0)
    source_np = be.to_host(source).astype(np.complex128)
    r_np = be.to_host(r_flat).astype(np.float64)
    out = np.zeros(len(dirs), dtype=np.complex128)
    coeff = (float(problem.k0) ** 2) * problem.grid.dv / (4.0 * np.pi)
    for start in range(0, len(dirs), int(batch_size)):
        d = dirs[start : start + int(batch_size)]
        phase = np.exp(-1j * float(problem.k0) * (d @ r_np))
        out[start : start + len(d)] = coeff * (phase @ source_np)
    return out


def scattering_pattern(u, problem: AcousticProblem, directions, *, normalize: str | None = "max") -> np.ndarray:
    """Return ``|f(s)|^2`` with optional max normalization."""
    amp = farfield_amplitude(u, problem, directions)
    sigma = np.abs(amp) ** 2
    if normalize is None:
        return sigma.astype(np.float64, copy=False)
    if normalize != "max":
        raise ValueError(f"normalize must be None or 'max', got {normalize!r}")
    max_value = float(np.max(sigma)) if sigma.size else 0.0
    if max_value > 0.0:
        sigma = sigma / max_value
    return sigma.astype(np.float64, copy=False)


def pattern_plane(
    u,
    problem: AcousticProblem,
    *,
    plane: str = "xy",
    n_angles: int = 360,
    normalize: str | None = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Return angle grid and acoustic scattering pattern in a coordinate plane."""
    if int(n_angles) <= 0:
        raise ValueError(f"n_angles must be positive, got {n_angles!r}")
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    dirs = _plane_directions(plane, phi)
    return phi, scattering_pattern(u, problem, dirs, normalize=normalize)
```

- [ ] **Step 4: Implement scalar visualization**

Create `src/em3d/acoustics/visualization.py`:

```python
"""Visualization helpers for scalar acoustic fields."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..vis import _require_matplotlib


def _field_part(u, part: str) -> np.ndarray:
    arr = np.asarray(u)
    if arr.ndim != 3:
        raise ValueError(f"u must have shape (Nx, Ny, Nz), got {arr.shape}")
    if part == "abs":
        return np.abs(arr).astype(np.float64)
    if part == "real":
        return arr.real.astype(np.float64)
    if part == "imag":
        return arr.imag.astype(np.float64)
    if part == "angle":
        return np.angle(arr).astype(np.float64)
    raise ValueError(f"part must be 'abs', 'real', 'imag', or 'angle', got {part!r}")


def _slice_scalar(values: np.ndarray, grid, plane: str, idx: int | None):
    if plane == "xy":
        idx = values.shape[2] // 2 if idx is None else int(idx)
        return np.asarray(grid.x), np.asarray(grid.y), values[:, :, idx], "x", "y"
    if plane == "xz":
        idx = values.shape[1] // 2 if idx is None else int(idx)
        return np.asarray(grid.x), np.asarray(grid.z), values[:, idx, :], "x", "z"
    if plane == "yz":
        idx = values.shape[0] // 2 if idx is None else int(idx)
        return np.asarray(grid.y), np.asarray(grid.z), values[idx, :, :], "y", "z"
    raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")


def plot_scalar_slice(
    u,
    grid,
    *,
    plane: str = "xy",
    idx: int | None = None,
    part: str = "abs",
    cmap: str = "viridis",
    title: str | None = None,
    filename: str | Path | None = None,
) -> tuple:
    """Plot one scalar acoustic field slice."""
    plt = _require_matplotlib()
    values = _field_part(u, part)
    horiz, vert, scalar, hlabel, vlabel = _slice_scalar(values, grid, plane, idx)
    H, V = np.meshgrid(horiz, vert, indexing="ij")
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(H, V, scalar, cmap=cmap, shading="auto")
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(hlabel)
    ax.set_ylabel(vlabel)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(str(filename), dpi=150, bbox_inches="tight")
    return fig, ax


def plot_scalar_slices(
    u,
    grid,
    *,
    output_dir: str | Path,
    prefix: str = "acoustic",
    parts: tuple[str, ...] = ("abs", "real", "imag", "angle"),
) -> list[Path]:
    """Save xy/xz/yz scalar slices for selected field parts."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for part in parts:
        for plane in ("xy", "xz", "yz"):
            path = root / f"{prefix}_{part}_{plane}.png"
            plot_scalar_slice(u, grid, plane=plane, part=part, title=f"{prefix} {part} {plane}", filename=path)
            written.append(path)
    return written


def plot_pattern(phi, sigma, *, filename: str | Path | None = None, polar: bool = False, title: str | None = None) -> tuple:
    """Plot acoustic scattering pattern in Cartesian or polar axes."""
    plt = _require_matplotlib()
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if polar:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel("phi (rad)")
        ax.set_ylabel("normalized pattern")
    ax.plot(phi, sigma)
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(str(filename), dpi=150, bbox_inches="tight")
    return fig, ax
```

- [ ] **Step 5: Export farfield and visualization**

Modify `src/em3d/acoustics/__init__.py`:

```python
from . import farfield
from . import visualization
from .farfield import farfield_amplitude, pattern_plane, scattering_pattern
```

Add names to `__all__`.

- [ ] **Step 6: Run far-field tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_gamma0_farfield.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```powershell
git add src/em3d/acoustics/__init__.py src/em3d/acoustics/farfield.py src/em3d/acoustics/visualization.py tests/test_acoustics_gamma0_farfield.py
git commit -m "feat(acoustics): add far field and scalar visualization"
```

---

## Task 6: Packaged Acoustic Experiment

**Files:**
- Create: `src/em3d/experiments/acoustic_scattering.py`
- Modify: `src/em3d/experiments/__init__.py` if it exists and exports experiment names.
- Test: `tests/test_acoustic_experiment.py`

**Interfaces:**
- Consumes: `em3d.acoustics`, existing solvers, `ExperimentLogger` style from `structured_lattice`.
- Produces:
  - `AcousticCase`
  - `make_homogeneous_case`
  - `make_layered_case`
  - `make_sphere_case`
  - `build_acoustic_problem(case)`
  - `run_acoustic_experiment(case=None, output_root=..., max_iter=..., rtol=..., make_plots=True)`

- [ ] **Step 1: Add failing packaged experiment tests**

Create `tests/test_acoustic_experiment.py`:

```python
import json
import shutil
import uuid
from pathlib import Path

import em3d


def _fresh_output_root(name: str) -> Path:
    return Path("experiments") / "outputs" / f"test-acoustic-{name}-{uuid.uuid4().hex}"


def test_packaged_acoustic_experiment_imports_and_defaults():
    from em3d.experiments.acoustic_scattering import AcousticCase, make_sphere_case

    case = make_sphere_case()
    assert isinstance(case, AcousticCase)
    assert case.N == (64, 64, 64)
    assert case.coarse_N == (6, 6, 6)
    assert case.solver_names == ("SIM", "BiCGStab", "TwoStep")
    assert case.eta_background == 1.0 + 0.0j


def test_packaged_acoustic_small_run_writes_artifacts(monkeypatch):
    from em3d.experiments.acoustic_scattering import build_acoustic_problem, make_sphere_case, run_acoustic_experiment

    root = _fresh_output_root("small-run")
    try:
        monkeypatch.setattr(em3d.acoustics.visualization, "plot_scalar_slices", lambda *args, **kwargs: [])
        monkeypatch.setattr(em3d.acoustics.visualization, "plot_pattern", lambda *args, **kwargs: (object(), object()))

        case = make_sphere_case(
            N=6,
            coarse_N=3,
            radius=0.25,
            eta_inside=1.3 + 0.05j,
            eta_background=1.0,
            k0=0.8,
            solver_names=("BiCGStab",),
        )
        problem, operator = build_acoustic_problem(case)
        assert problem.grid.N == (6, 6, 6)
        assert isinstance(operator, em3d.acoustics.AcousticOperator)

        summary = run_acoustic_experiment(
            case=case,
            output_root=root,
            max_iter=5,
            rtol=1e-4,
            n_angles=12,
        )
        assert summary["case_name"] == case.name
        assert summary["N"] == "6x6x6"
        assert summary["coarse_N"] == "3x3x3"
        assert summary["solver_names"] == ["BiCGStab"]
        assert (root / "tables" / "acoustic_solver_runs.csv").is_file()
        assert (root / "raw" / "acoustic_summary.json").is_file()
        assert (root / "raw" / "acoustic_residual_histories.json").is_file()
        assert '"event": "finish"' in (root / "raw" / "acoustic_scattering.jsonl").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)
```

- [ ] **Step 2: Run tests and confirm missing experiment**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustic_experiment.py -q
```

Expected: FAIL with missing `em3d.experiments.acoustic_scattering`.

- [ ] **Step 3: Implement experiment dataclasses and logger**

Create `src/em3d/experiments/acoustic_scattering.py` with these top-level pieces:

```python
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
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": str(event), **payload}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        with self.text_path.open("a", encoding="utf-8") as f:
            text_payload = " ".join(f"{key}={json.dumps(value, ensure_ascii=False, default=_json_default)}" for key, value in payload.items())
            f.write(f"{row['time']} {row['event']} {text_payload}".rstrip() + "\n")
        return row


@dataclass(frozen=True)
class AcousticCase:
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
```

- [ ] **Step 4: Implement case factories and build function**

Add to `src/em3d/experiments/acoustic_scattering.py`:

```python
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
        direction=direction,
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
        direction=direction,
        amplitude=complex(amplitude),
        solver_names=tuple(solver_names),
        slab_axis=int(slab_axis),
        slab_width_fraction=float(slab_width_fraction),
    )


def make_homogeneous_case(**kwargs) -> AcousticCase:
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


def build_acoustic_problem(case: AcousticCase, *, precision: em3d.Precision = em3d.Precision.DOUBLE):
    be = em3d.Backend.numpy(precision)
    grid = em3d.Grid(N=case.N, L=case.L, center=(0.0, 0.0, 0.0), backend=be)
    if case.kind == "sphere":
        eta = em3d.acoustics.eta_sphere(
            grid,
            center=(0.0, 0.0, 0.0),
            radius=float(case.radius),
            eta_inside=case.eta_inside,
            eta_outside=case.eta_background,
        )
    elif case.kind == "slab":
        eta = em3d.acoustics.eta_slab(
            grid,
            eta_inside=case.eta_inside,
            eta_outside=case.eta_background,
            axis=int(case.slab_axis),
            width_fraction=float(case.slab_width_fraction),
        )
    elif case.kind == "homogeneous":
        eta = em3d.acoustics.eta_homogeneous(grid, case.eta_inside)
    else:
        raise ValueError(f"unknown acoustic case kind {case.kind!r}")
    problem = em3d.acoustics.make_acoustic_problem(
        grid,
        eta,
        k0=case.k0,
        direction=case.direction,
        amplitude=case.amplitude,
    )
    return problem, em3d.acoustics.AcousticOperator(problem)
```

- [ ] **Step 5: Implement run function and artifacts**

Add to `src/em3d/experiments/acoustic_scattering.py`:

```python
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


def _ensure_output_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    paths = {"root": root, "raw": root / "raw", "tables": root / "tables", "figures": root / "figures"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _save_runs_csv(runs: list[_SolverRun], path: str | Path) -> None:
    fieldnames = ["case_name", "solver_name", "N", "dof", "converged", "iterations", "final_residual", "elapsed_sec"]
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
    case = case if case is not None else make_sphere_case()
    paths = _ensure_output_dirs(output_root)
    logger = logger if logger is not None else ExperimentLogger(paths["root"], "acoustic_scattering")
    logger.event("start", case_name=case.name, N=case.N, coarse_N=case.coarse_N, solver_names=case.solver_names)

    problem, operator = build_acoustic_problem(case)
    analysis = em3d.acoustics.gamma0.estimate_from_problem(problem, coarse_N=case.coarse_N)
    logger.event("gamma0_estimated", coarse_N=analysis.coarse_N, mu=analysis.mu, radius=analysis.radius, rho=analysis.rho)

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
    _save_json({"runs": [{**run.to_row(), "residual_history": run.residual_history} for run in runs]}, paths["raw"] / "acoustic_residual_histories.json")

    phi, sigma = em3d.acoustics.pattern_plane(u, problem, plane="xy", n_angles=n_angles, normalize="max")
    if make_plots:
        em3d.acoustics.visualization.plot_scalar_slices(u, problem.grid, output_dir=paths["figures"], prefix="acoustic")
        em3d.acoustics.visualization.plot_pattern(phi, sigma, filename=paths["figures"] / "acoustic_pattern_xy.png", polar=False, title="Acoustic scattering pattern")
        em3d.acoustics.visualization.plot_pattern(phi, sigma, filename=paths["figures"] / "acoustic_pattern_xy_polar.png", polar=True, title="Acoustic scattering pattern")

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
        "gamma0": {"mu_real": float(np.real(analysis.mu)), "mu_imag": float(np.imag(analysis.mu)), "radius": float(analysis.radius), "rho": float(analysis.rho)},
        "pattern": {"max": float(np.max(sigma)) if sigma.size else 0.0, "mean": float(np.mean(sigma)) if sigma.size else 0.0},
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
```

- [ ] **Step 6: Run experiment tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustic_experiment.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```powershell
git add src/em3d/experiments/acoustic_scattering.py tests/test_acoustic_experiment.py
git commit -m "feat(experiments): add packaged acoustic scattering run"
```

---

## Task 7: Notebook, README, Wiki

**Files:**
- Create: `notebooks/acoustic-scattering-kaggle.ipynb`
- Modify: `README.md`
- Create: `wiki/code/acoustic-scattering.md`
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`
- Test: `tests/test_acoustic_experiment.py`

**Interfaces:**
- Consumes: packaged experiment API from Task 6.
- Produces: user-facing runnable notebook and documentation.

- [ ] **Step 1: Add failing notebook source test**

Append to `tests/test_acoustic_experiment.py`:

```python
def test_acoustic_kaggle_notebook_uses_packaged_api():
    path = Path("notebooks") / "acoustic-scattering-kaggle.ipynb"
    assert path.is_file()
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "pip install" in source
    assert "git+https://github.com/qwerty29544/em3d.git" in source
    assert "from em3d.experiments.acoustic_scattering import" in source
    assert "make_sphere_case" in source
    assert "run_acoustic_experiment" in source
    assert "zipfile" in source
```

- [ ] **Step 2: Run notebook source test and confirm failure**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustic_experiment.py::test_acoustic_kaggle_notebook_uses_packaged_api -q
```

Expected: FAIL because notebook is absent.

- [ ] **Step 3: Create notebook with packaged API**

Create `notebooks/acoustic-scattering-kaggle.ipynb` as a valid JSON notebook containing these code cells:

```python
!pip install -q "em3d[vis] @ git+https://github.com/qwerty29544/em3d.git"
```

```python
from pathlib import Path
import shutil
import zipfile

from em3d.experiments.acoustic_scattering import make_sphere_case, run_acoustic_experiment
```

```python
case = make_sphere_case(
    N=64,
    coarse_N=6,
    radius=0.25,
    eta_inside=2.0 + 0.25j,
    eta_background=1.0 + 0.0j,
    k0=5.0,
    solver_names=("SIM", "BiCGStab", "TwoStep"),
)
summary = run_acoustic_experiment(
    case=case,
    output_root="/kaggle/working/acoustic-scattering",
    max_iter=500,
    rtol=1e-6,
    n_angles=180,
    make_plots=True,
)
summary
```

```python
root = Path("/kaggle/working/acoustic-scattering")
zip_path = Path("/kaggle/working/acoustic-scattering-results.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(root))
zip_path
```

Use `py` or a minimal JSON patcher only if editing `.ipynb` manually is too error-prone; the notebook content is data, not package source code.

- [ ] **Step 4: Update README with compact example**

Add a section to `README.md`:

````markdown
## Acoustic Scalar Scattering

```python
import em3d
from em3d.experiments.acoustic_scattering import make_sphere_case, run_acoustic_experiment

case = make_sphere_case(
    N=64,
    coarse_N=6,
    radius=0.25,
    eta_inside=2.0 + 0.25j,
    eta_background=1.0,
    k0=5.0,
    solver_names=("SIM", "BiCGStab", "TwoStep"),
)

summary = run_acoustic_experiment(
    case=case,
    output_root="experiments/outputs/acoustic-scattering",
    max_iter=500,
    rtol=1e-6,
)
print(summary["reference_solver"], summary["pattern"]["max"])
```

The acoustic API accepts `eta` as the material parameter and internally solves with `chi = eta - 1`, so `eta = 1` is the non-scattering background.
````

- [ ] **Step 5: Add wiki code page**

Create `wiki/code/acoustic-scattering.md`:

```markdown
---
title: Акустическое скалярное рассеяние
zone: code
tags: [акустика, ОИУ, Фредгольм-2, БПФ, итерационные-методы]
sources: [literature/samokhin-yurchenkov-mathematics-2024, literature/yurchenkov-ipu-2025]
status: draft
created: 2026-07-01
updated: 2026-07-01
---

# Акустическое скалярное рассеяние

## Назначение

Реализация `em3d.acoustics` моделирует стационарное рассеяние скалярной акустической волны на трёхмерной прозрачной структуре через объёмное интегральное уравнение Фредгольма 2-го рода.

## Алгоритм

Поле `u` определяется из системы `u - B((eta - 1)u) = u0`, где `B` — дискретная свёртка со скалярной функцией Грина Гельмгольца. Свёртка ускоряется FFT на удвоенной области. Параметр `eta` задаётся пользователем, а контраст `chi = eta - 1` вычисляется внутри acoustic API.

## Параметры и конфигурация

| Параметр | Значение | Описание |
|----------|----------|----------|
| `eta` | complex scalar field | акустический параметр среды |
| `chi` | `eta - 1` | контраст, используемый в интегральном операторе |
| `k0` | float | волновое число фоновой среды |
| `N` | `(Nx, Ny, Nz)` | число ячеек декартовой сетки |
| `coarse_N` | `(nx, ny, nz)` | грубая сетка для оценки `gamma0` |

## Анализ сложности

FFT-матвек имеет сложность `O(N log N)` по числу ячеек и хранит scalar kernel на удвоенной области. Dense matrix используется только для малых сеток и проверки.

## Результаты и наблюдения

Пакетный эксперимент `em3d.experiments.acoustic_scattering` сохраняет логи, историю невязок, срезы скалярного поля и нормированную диаграмму рассеяния.

## Реализованные концепции

- [[concepts/volume-integral-equation]]
- [[concepts/helmholtz-green-function]]
- [[concepts/fft-convolution-acceleration]]
- [[concepts/two-step-gradient-descent]]

## Исходные файлы

- `src/em3d/acoustics/`
- `src/em3d/experiments/acoustic_scattering.py`
- `notebooks/acoustic-scattering-kaggle.ipynb`
```

- [ ] **Step 6: Update wiki index and log**

Add to `wiki/index.md` under `## Code`:

```markdown
- [[code/acoustic-scattering]] — подпакет `em3d.acoustics` для скалярного акустического рассеяния с FFT-оператором, `gamma0`, итерационными методами и пакетным экспериментом.
```

Append to `wiki/log.md`:

```markdown
## [2026-07-01] code | Акустическое скалярное рассеяние

- Создана страница: code/acoustic-scattering
- Обновлены страницы: index
- Замечания: зафиксирована реализация `em3d.acoustics` для скалярного ОИУ акустики с соглашением `chi = eta - 1`, FFT-оператором, dense-верификацией, `gamma0`, диаграммой рассеяния, пакетным экспериментом и Kaggle notebook.
```

- [ ] **Step 7: Run notebook/docs test**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustic_experiment.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 7**

```powershell
git add README.md notebooks/acoustic-scattering-kaggle.ipynb tests/test_acoustic_experiment.py wiki/code/acoustic-scattering.md wiki/index.md wiki/log.md
git commit -m "docs(acoustics): add acoustic experiment docs and notebook"
```

If `wiki/` is not tracked by the current git root, commit tracked files only and report that wiki files were updated locally.

---

## Task 8: Full Verification And Cleanup

**Files:**
- Modify only files with discovered failures.
- Test: full acoustic and targeted regression suite.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified implementation ready for user review.

- [ ] **Step 1: Run acoustic test suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_acoustics_problem_materials.py tests/test_acoustics_operator.py tests/test_acoustics_gamma0_farfield.py tests/test_acoustic_experiment.py -q
```

Expected: PASS.

- [ ] **Step 2: Run targeted EM regression suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest tests/test_operator_vs_dense.py tests/test_solvers.py tests/test_gamma0.py tests/test_farfield.py tests/test_packaged_experiments.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -m pytest -q
```

Expected: PASS, with any pre-existing `skip` or `xfail` unchanged.

- [ ] **Step 4: Run diff whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Existing CRLF warnings from Git may appear and do not require code changes.

- [ ] **Step 5: Inspect public API manually**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
py -c "import em3d; print(em3d.acoustics.AcousticOperator); print(em3d.acoustics.eta_homogeneous)"
```

Expected: prints class/function objects without import errors.

- [ ] **Step 6: Commit final verification fixes if any**

If Step 1-5 required fixes:

```powershell
git add src/em3d/acoustics tests README.md notebooks wiki
git commit -m "fix(acoustics): address verification issues"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review Notes

### Spec coverage

- Separate `em3d.acoustics` package: Tasks 1-5.
- `eta` input and internal `chi = eta - 1`: Tasks 1, 2, 3 and tests.
- Scalar FFT operator and dense comparison: Tasks 2-3.
- Correct `rmatvec` for `TwoStep`: Task 3.
- Existing solvers reuse: Task 4.
- Acoustic `gamma0`: Task 4.
- Far-field/scattering pattern: Task 5.
- Scalar visualization: Task 5.
- Packaged experiments and logs: Task 6.
- Kaggle notebook, README, wiki: Task 7.
- Regression verification: Task 8.

### Type consistency

- `AcousticProblem.eta`, `AcousticProblem.wave`, `AcousticProblem.chi` use scalar shape `grid.N`.
- `AcousticOperator.matvec` and `rmatvec` accept and return scalar shape `grid.N`.
- `H_scalar_matrix(problem)` returns dense matrix shape `(Nx*Ny*Nz, Nx*Ny*Nz)`.
- `gamma0.estimate_from_problem` returns existing `Gamma0Analysis`.
- Experiment solver results use existing `SolverResult`.

### Risk controls

- Dense-vs-FFT test pins kernel embedding and scalar convolution.
- Adjoint test pins `rmatvec`.
- Background identity test pins `eta=1 -> chi=0`.
- EM regression tests keep existing electrodynamics stable.
