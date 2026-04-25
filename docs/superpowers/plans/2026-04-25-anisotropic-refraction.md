# Anisotropic Refraction Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `cylinder_refraction`, `step_refraction`, and `ellipsis_refraction` to accept scalar or `(3,3)` matrix `eps_real`/`eps_imag` and always return `(3, 3, Nx, Ny, Nz)` complex tensor.

**Architecture:** Add private `_to_eta_mat(eps_real, eps_imag)` helper that promotes scalars to diagonal matrices; update all three geometry functions to call it and return a `(3,3,Nx,Ny,Nz)` tensor. `apply_refraction` is unchanged. Call sites in tests replace the two-step `scalar = geom(...); eta = apply_refraction(scalar_eta=scalar)` with a single `eta = geom(...)`.

**Tech Stack:** Python 3.11, numpy, pytest, em3d backend abstraction (`be.zeros`, `be.xp`)

---

## File Map

- **Modify:** `src/em3d/refraction.py` — add `import numpy as np`, add `_to_eta_mat`, rewrite 3 geometry functions
- **Modify:** `tests/test_refraction.py` — update shape assertions in 3 tests, add 2 new tests
- **Modify:** `tests/test_farfield.py` — update `_make_problem` and 2 tests that call `apply_refraction(scalar_eta=geom(...))`

---

## Task 1: Update `tests/test_refraction.py` — failing tests first

**Files:**
- Modify: `tests/test_refraction.py`

Context: The current tests assert `mask.shape == (4, 4, 4)`. After the implementation change
they will assert `(3, 3, 4, 4, 4)`. Writing these first gives us a red-green cycle.
The `conftest.py` in tests/ provides `backend_numpy_double` fixture.

- [ ] **Step 1: Replace the full contents of `tests/test_refraction.py`**

```python
import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.grid import Grid
from em3d.refraction import (
    cylinder_refraction,
    step_refraction,
    ellipsis_refraction,
    apply_refraction,
)


def _grid(be):
    return Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


# --- Updated geometry tests: shape now (3,3,Nx,Ny,Nz) ---

def test_cylinder_mask_inside_outside(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_real = 2.25
    eta = cylinder_refraction(grid, eps_real=eps_real, eps_imag=0.0, radius=0.49, axis="z")
    assert eta.shape == (3, 3, 4, 4, 4)
    cx, cy = grid.N[0] // 2, grid.N[1] // 2
    expected_eta = eps_real - 1.0
    for d in range(3):
        assert abs(eta[d, d, cx, cy, 0].real - expected_eta) < 1e-12
    np.testing.assert_allclose(eta[0, 1], np.zeros(grid.N))     # off-diagonal = 0
    np.testing.assert_allclose(eta[0, 0, 0, 0, :], np.zeros(grid.N[2]))  # corner = 0


def test_step_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eta = step_refraction(grid, eps_real=2.0, eps_imag=0.1, z_min=-0.25, z_max=0.25)
    assert eta.shape == (3, 3, 4, 4, 4)
    mid = grid.N[2] // 2
    assert abs(eta[0, 0, 0, 0, mid]) > 0
    assert abs(eta[0, 0, 0, 0, 0]) < 1e-12
    assert abs(eta[0, 0, 0, 0, -1]) < 1e-12


def test_ellipsis_refraction(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eta = ellipsis_refraction(
        grid, eps_real=1.5, eps_imag=0.0, center=(0.0, 0.0, 0.0), radius=(0.3, 0.4, 0.5)
    )
    assert eta.shape == (3, 3, 4, 4, 4)
    cx, cy, cz = grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2
    assert abs(eta[0, 0, cx, cy, cz]) > 0


# --- apply_refraction still works with manually-built scalar field ---

def test_apply_refraction_scalar_returns_isotropic_tensor(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_real = 2.0
    scalar = np.full(grid.N, complex(eps_real - 1.0, 0.0))
    eta = apply_refraction(grid, scalar_eta=scalar)
    assert eta.shape == (3, 3) + grid.N
    assert eta.dtype == backend_numpy_double.complex_dtype
    np.testing.assert_allclose(eta[0, 0], scalar)
    np.testing.assert_allclose(eta[1, 1], scalar)
    np.testing.assert_allclose(eta[2, 2], scalar)
    np.testing.assert_allclose(eta[0, 1], np.zeros_like(scalar))


# --- New: anisotropic input ---

def test_ellipsis_anisotropic(backend_numpy_double):
    """Diagonal anisotropic eps: each component of eta diagonal differs."""
    grid = _grid(backend_numpy_double)
    eps_r = np.diag([2.0, 1.5, 1.2])
    eps_i = np.zeros((3, 3))
    eta = ellipsis_refraction(
        grid, eps_real=eps_r, eps_imag=eps_i,
        center=(0, 0, 0), radius=(0.3, 0.3, 0.3),
    )
    assert eta.shape == (3, 3) + grid.N
    cx, cy, cz = grid.N[0] // 2, grid.N[1] // 2, grid.N[2] // 2
    np.testing.assert_allclose(eta[0, 0, cx, cy, cz], 1.0, atol=1e-12)  # 2.0 - 1
    np.testing.assert_allclose(eta[1, 1, cx, cy, cz], 0.5, atol=1e-12)  # 1.5 - 1
    np.testing.assert_allclose(eta[2, 2, cx, cy, cz], 0.2, atol=1e-12)  # 1.2 - 1
    np.testing.assert_allclose(eta[0, 1, cx, cy, cz], 0.0, atol=1e-12)  # off-diagonal
    np.testing.assert_allclose(eta[:, :, 0, 0, 0], np.zeros((3, 3)), atol=1e-12)


# --- New: type mismatch raises TypeError ---

def test_to_eta_mat_type_mismatch(backend_numpy_double):
    """Mixing scalar eps_real with matrix eps_imag must raise TypeError."""
    grid = _grid(backend_numpy_double)
    with pytest.raises(TypeError):
        ellipsis_refraction(
            grid, eps_real=2.0, eps_imag=np.zeros((3, 3)),
            center=(0, 0, 0), radius=(0.1, 0.1, 0.1),
        )
```

- [ ] **Step 2: Run new tests — verify they FAIL**

```
pytest tests/test_refraction.py -v
```

Expected: 5 failures — `AssertionError: assert (4, 4, 4) == (3, 3, 4, 4, 4)` and
`TypeError` from `complex(eps_real - 1.0, eps_imag)` when `eps_imag` is a matrix.

---

## Task 2: Implement `_to_eta_mat` and update `src/em3d/refraction.py`

**Files:**
- Modify: `src/em3d/refraction.py`

Context: The file currently has no `import numpy as np`. The `_to_eta_mat` helper is pure numpy
(no backend dependency). Geometry functions call it, then fill a `(3,3,Nx,Ny,Nz)` backend array
cell-by-cell using `xp.where(mask, scalar, out[i,j])`. `out` is pre-zeroed, so
`xp.where(mask, eta_mat[i,j], out[i,j])` sets the value inside the mask and keeps zero outside.

- [ ] **Step 3: Replace the full contents of `src/em3d/refraction.py`**

```python
"""Refractive-index / permittivity profiles on the grid."""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .grid import Grid


def _to_eta_mat(eps_real, eps_imag) -> np.ndarray:
    """Convert eps_real/eps_imag to a (3,3) complex128 eta matrix η = ε − I.

    Parameters
    ----------
    eps_real : float or array-like (3,3) — real part of permittivity ε
    eps_imag : float or array-like (3,3) — imaginary part of permittivity ε

    Both arguments must be the same kind: both scalars or both (3,3) arrays.
    Scalar inputs are promoted to scalar * eye(3) (isotropic case).

    Returns
    -------
    np.ndarray shape (3,3), dtype complex128
    """
    real_is_scalar = np.ndim(eps_real) == 0
    imag_is_scalar = np.ndim(eps_imag) == 0
    if real_is_scalar != imag_is_scalar:
        raise TypeError(
            "eps_real and eps_imag must both be scalars or both be (3,3) arrays; "
            f"got ndim={np.ndim(eps_real)} and ndim={np.ndim(eps_imag)}"
        )
    if real_is_scalar:
        return (float(eps_real) - 1.0 + 1j * float(eps_imag)) * np.eye(3, dtype=np.complex128)
    E = np.asarray(eps_real, dtype=np.float64)
    F = np.asarray(eps_imag, dtype=np.float64)
    if E.shape != (3, 3):
        raise ValueError(f"eps_real must have shape (3,3), got {E.shape}")
    if F.shape != (3, 3):
        raise ValueError(f"eps_imag must have shape (3,3), got {F.shape}")
    return (E - np.eye(3)) + 1j * F


def cylinder_refraction(
    grid: Grid,
    *,
    eps_real,
    eps_imag,
    radius: float,
    axis: Literal["x", "y", "z"] = "z",
) -> object:
    """Infinite cylinder along `axis`, radius in grid length units.

    Returns
    -------
    ndarray shape (3, 3, Nx, Ny, Nz) complex — contrast tensor η = ε − I
    """
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    if axis == "z":
        r2 = X * X + Y * Y
    elif axis == "y":
        r2 = X * X + Z * Z
    elif axis == "x":
        r2 = Y * Y + Z * Z
    else:
        raise ValueError(f"axis must be 'x'|'y'|'z', got {axis!r}")
    mask = r2 <= radius * radius
    eta_mat = _to_eta_mat(eps_real, eps_imag)          # (3,3) complex128, numpy
    out = be.zeros((3, 3) + grid.N, kind="complex")    # pre-zeroed backend array
    for i in range(3):
        for j in range(3):
            out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def step_refraction(
    grid: Grid,
    *,
    eps_real,
    eps_imag,
    z_min: float,
    z_max: float,
) -> object:
    """Slab between z_min and z_max (inclusive).

    Returns
    -------
    ndarray shape (3, 3, Nx, Ny, Nz) complex — contrast tensor η = ε − I
    """
    be = grid.backend
    xp = be.xp
    _, _, Z = grid.coords()
    mask = (Z >= z_min) & (Z <= z_max)
    eta_mat = _to_eta_mat(eps_real, eps_imag)
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for i in range(3):
        for j in range(3):
            out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def ellipsis_refraction(
    grid: Grid,
    *,
    eps_real,
    eps_imag,
    center: Tuple[float, float, float],
    radius: Tuple[float, float, float],
) -> object:
    """Axis-aligned ellipsoid.

    Returns
    -------
    ndarray shape (3, 3, Nx, Ny, Nz) complex — contrast tensor η = ε − I
    """
    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    cx, cy, cz = center
    rx, ry, rz = radius
    metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    mask = metric <= 1.0
    eta_mat = _to_eta_mat(eps_real, eps_imag)
    out = be.zeros((3, 3) + grid.N, kind="complex")
    for i in range(3):
        for j in range(3):
            out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
    return out


def apply_refraction(grid: Grid, *, scalar_eta=None, tensor_eta=None) -> object:
    """Return a (3, 3, Nx, Ny, Nz) complex tensor.

    Parameters
    ----------
    scalar_eta : array (Nx, Ny, Nz) complex — isotropic η field; placed on diagonal
    tensor_eta : array (3, 3, Nx, Ny, Nz) complex — full anisotropic η tensor

    Exactly one of scalar_eta or tensor_eta must be provided.
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

- [ ] **Step 4: Run `test_refraction.py` — verify all pass**

```
pytest tests/test_refraction.py -v
```

Expected output (7 passed):
```
test_refraction.py::test_cylinder_mask_inside_outside PASSED
test_refraction.py::test_step_refraction PASSED
test_refraction.py::test_ellipsis_refraction PASSED
test_refraction.py::test_apply_refraction_scalar_returns_isotropic_tensor PASSED
test_refraction.py::test_ellipsis_anisotropic PASSED
test_refraction.py::test_to_eta_mat_type_mismatch PASSED
```

- [ ] **Step 5: Commit**

```bash
git add src/em3d/refraction.py tests/test_refraction.py
git commit -m "feat: geometry functions accept anisotropic eps tensors, return (3,3,N) always"
```

---

## Task 3: Fix `tests/test_farfield.py` call sites

**Files:**
- Modify: `tests/test_farfield.py`

Context: `test_farfield.py` uses `ellipsis_refraction` in 3 places with the old two-step pattern
`scalar = geom(...); eta = apply_refraction(grid, scalar_eta=scalar)`.
After Task 2 the geometry functions return `(3,3,Nx,Ny,Nz)` — passing that to
`apply_refraction(scalar_eta=...)` raises `ValueError` (shape mismatch).
`test_zero_contrast` (manual `be.zeros`) and `test_single_cell_analytic`
(manual `apply_refraction(scalar_eta=np.full(...))`) are NOT affected — leave them alone.

- [ ] **Step 6: Run `test_farfield.py` before changes — confirm breakage**

```
pytest tests/test_farfield.py -v
```

Expected: failures in `test_rcs_nonnegative`, `test_rcs_plane_shape`,
`test_fft_vs_direct_agreement`, `test_rcs_plane_symmetry` — all call `_make_problem` or
directly use `apply_refraction(scalar_eta=geom(...))`.

- [ ] **Step 7: Replace the full contents of `tests/test_farfield.py`**

```python
"""Tests for em3d.farfield — scatter_integral, rcs, rcs_plane."""
import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.grid import Grid
from em3d.refraction import apply_refraction, ellipsis_refraction
from em3d.wave import flat_wave_vec
from em3d.problem import Problem
from em3d.farfield import scatter_integral, rcs, rcs_plane


def _be():
    return Backend.numpy(Precision.DOUBLE)


def _make_problem(N=(4, 4, 4), eps_real=2.0, eps_imag=0.0, k0=1.0):
    be = _be()
    grid = Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=eps_real, eps_imag=eps_imag,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    wave = flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=k0, volume=volume)


# --- Test 1: zero contrast (direct method) ---

@pytest.mark.parametrize("method", ["direct", "fft"])
def test_zero_contrast(method):
    """eta=0 → F=0, sigma=0."""
    be = _be()
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = be.zeros((3, 3) + grid.N, kind="complex")
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 64)
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = scatter_integral(wave, problem, directions, method=method)
    np.testing.assert_allclose(F, 0.0, atol=1e-14)


# --- Test 2: non-negative ---

def test_rcs_nonnegative():
    """RCS >= 0 for arbitrary non-zero field."""
    problem = _make_problem()
    direction = np.array([1.0, 0.0, 0.0])
    result = rcs(problem.wave, problem, direction)
    assert result >= 0.0


# --- Test 3: single cell analytic ---

def test_single_cell_analytic():
    """1x1x1 grid at origin: F = dv * eta @ u (phase=1 since r_cell=0)."""
    be = _be()
    grid = Grid(N=(1, 1, 1), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    # Grid cell is at r=(0,0,0): start = center - L/2 + dx/2 = 0 - 0.5 + 0.5 = 0
    eta_val = complex(1.5, 0.1)
    scalar = np.full(grid.N, eta_val, dtype=np.complex128)
    eta = apply_refraction(grid, scalar_eta=scalar)
    u = np.zeros((3, 1, 1, 1), dtype=np.complex128)
    u[0, 0, 0, 0] = 1.0 + 0.5j          # x-component only
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv)
    direction = np.array([1.0, 0.0, 0.0])
    F = scatter_integral(u, problem, direction[np.newaxis, :])[0]   # (3,)
    # analytic: r_cell=(0,0,0), phase=exp(0)=1
    # J = eta_val * u  (isotropic diagonal eta)
    # F = dv * J = 1.0 * eta_val * (1+0.5j, 0, 0)
    # dv = 1.0
    F_analytic = np.array([eta_val * (1.0 + 0.5j), 0.0, 0.0])
    np.testing.assert_allclose(F, F_analytic, rtol=1e-12)


# --- Test 4: fft vs direct agreement ---

def test_fft_vs_direct_agreement():
    """FFT backend matches direct to atol=1e-4 on an 8x8x8 grid."""
    be = _be()
    grid = Grid(N=(8, 8, 8), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0, volume=grid.dv * 512)
    rng = np.random.default_rng(42)
    u = (rng.standard_normal((3, 8, 8, 8))
         + 1j * rng.standard_normal((3, 8, 8, 8))).astype(np.complex128)
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ])
    F_direct = scatter_integral(u, problem, directions, method="direct")
    F_fft    = scatter_integral(u, problem, directions, method="fft")
    np.testing.assert_allclose(F_fft, F_direct, atol=1e-4,
                                err_msg="FFT and direct backends disagree")


# --- Test 5: rcs_plane shape ---

def test_rcs_plane_shape():
    """rcs_plane returns (phi, sigma) of shape (n_phi,)."""
    problem = _make_problem(k0=0.5)
    phi, sigma = rcs_plane(problem.wave, problem, n_phi=12, plane="xy")
    assert phi.shape == (12,)
    assert sigma.shape == (12,)


# --- Test 6: rcs_plane symmetry ---

def test_rcs_plane_symmetry():
    """Isotropic sphere, real eta, uniform x-field:
    sigma(phi) == sigma(phi + pi) in xy-plane.
    Proof: for real eta and u=(1,0,0), F(phi+pi) = conj(F(phi)),
    so |e_p(phi+pi) x F(phi+pi)|^2 = |e_p(phi) x F(phi)|^2.
    """
    be = _be()
    N = (8, 8, 8)
    grid = Grid(N=N, L=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0), backend=be)
    eta = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.4, 0.4, 0.4),
    )
    wave = flat_wave_vec(grid, k=0.1, orient=(0, 0, 1), amplitude=(1, 0, 0))
    u = np.zeros((3,) + N, dtype=np.complex128)
    u[0] = 1.0                                       # uniform x-polarized field
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.1,
                      volume=grid.dv * int(np.prod(N)))
    n_phi = 24
    phi, sigma = rcs_plane(u, problem, n_phi=n_phi, plane="xy")
    half = n_phi // 2
    np.testing.assert_allclose(
        sigma[:half], sigma[half:], rtol=1e-10, atol=1e-35,
        err_msg="RCS not symmetric: sigma(phi) != sigma(phi+pi)",
    )
```

- [ ] **Step 8: Run `test_farfield.py` — verify all pass**

```
pytest tests/test_farfield.py -v
```

Expected (8 items, all PASSED):
```
test_farfield.py::test_zero_contrast[direct] PASSED
test_farfield.py::test_zero_contrast[fft] PASSED
test_farfield.py::test_rcs_nonnegative PASSED
test_farfield.py::test_single_cell_analytic PASSED
test_farfield.py::test_fft_vs_direct_agreement PASSED
test_farfield.py::test_rcs_plane_shape PASSED
test_farfield.py::test_rcs_plane_symmetry PASSED
```

- [ ] **Step 9: Run full test suite — verify nothing else broke**

```
pytest -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add tests/test_farfield.py
git commit -m "fix: update test_farfield call sites for tensor-returning geometry functions"
```
