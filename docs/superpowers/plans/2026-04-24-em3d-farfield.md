# em3d.farfield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `em3d.farfield` subpackage with `scatter_integral`, `rcs`, and `rcs_plane` for computing far-field RCS from a VIE solution, with two backends (batched matmul and FFT+interpolation).

**Architecture:** Three files — `_core.py` (Variant A: direct batched matmul), `_fft.py` (Variant B: 3D FFT + `map_coordinates`), `__init__.py` (public API, routes `method=` arg to correct backend). Tests live in `tests/test_farfield.py`. The existing `em3d.__init__` is updated to expose `farfield` as a submodule.

**Tech Stack:** numpy, scipy.ndimage.map_coordinates (for FFT interpolation), existing `em3d.Backend` / `em3d.Problem` / `em3d.Grid`.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/em3d/farfield/__init__.py` | Create | Public API: `scatter_integral`, `rcs`, `rcs_plane` |
| `src/em3d/farfield/_core.py` | Create | Variant A: batched matmul (backend-agnostic, returns numpy) |
| `src/em3d/farfield/_fft.py` | Create | Variant B: 3D FFT + `map_coordinates` (returns numpy) |
| `tests/test_farfield.py` | Create | 6 unit tests |
| `src/em3d/__init__.py` | Modify | Add `from . import farfield` and `"farfield"` to `__all__` |

---

## Background: key formulas

```
J_i(r) = Σ_j  eta[i,j,r] * u[j,r]          # polarization current, shape (3, N)
F[m, i] = dv * Σ_n  J[i, n] * exp(-1j*k0*(e_p[m] @ r[n]))   # scatter integral, shape (M, 3)
sigma[m] = k0^4 / (16*pi^2) * |e_p[m] × F[m]|^2             # RCS
```

`problem.eps_tensor` already stores η = ε − I (see `apply_refraction`). The notebook bug was `exp(+1j…)` — this implementation uses `exp(-1j…)`.

---

## Task 1: Direct backend + public API + 4 tests

**Files:**
- Create: `src/em3d/farfield/__init__.py`
- Create: `src/em3d/farfield/_core.py`
- Create: `tests/test_farfield.py` (4 tests, covering direct method only)

### Step 1 — Write the 4 failing tests

Create `tests/test_farfield.py`:

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
    scalar = ellipsis_refraction(
        grid, eps_real=eps_real, eps_imag=eps_imag,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=k0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    volume = grid.dv * int(np.prod(N))
    return Problem(grid=grid, eps_tensor=eta, wave=wave, k0=k0, volume=volume)


# --- Test 1: zero contrast (direct method) ---

@pytest.mark.parametrize("method", ["direct"])
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
    """1×1×1 grid at origin: F = dv * eta @ u (phase=1 since r_cell=0)."""
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
    # dv = 1.0 * 1.0 * 1.0 = 1.0
    F_analytic = np.array([eta_val * (1.0 + 0.5j), 0.0, 0.0])
    np.testing.assert_allclose(F, F_analytic, rtol=1e-12)


# --- Test 4 (placeholder for FFT — added in Task 2) ---
# --- Test 5: rcs_plane shape ---

def test_rcs_plane_shape():
    """rcs_plane returns (phi, sigma) of shape (n_phi,)."""
    problem = _make_problem(k0=0.5)
    phi, sigma = rcs_plane(problem.wave, problem, n_phi=12, plane="xy")
    assert phi.shape == (12,)
    assert sigma.shape == (12,)
```

- [ ] **Step 2 — Run tests, verify they fail**

```
pytest tests/test_farfield.py -v
```

Expected: `ModuleNotFoundError: No module named 'em3d.farfield'` or `ImportError`.

- [ ] **Step 3 — Create `src/em3d/farfield/_core.py`**

```python
"""Variant A: batched matmul computation of the scatter integral.

F[m] = dv * J_flat @ phase[:, m].conj().T
where J_flat[i, n] = sum_j eta[i,j,...][n] * u[j,...][n]   (polarization current)
      phase[b, n]  = exp(-1j * k0 * (e_p[b] @ r[n]))
"""
from __future__ import annotations

import numpy as np

from ..problem import Problem


def scatter_integral_direct(
    u,
    problem: Problem,
    directions,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Return F of shape (M, 3) as complex128 numpy array.

    Parameters
    ----------
    u          : array (3, N1, N2, N3) complex — E-field
    problem    : Problem — provides eps_tensor, k0, grid
    directions : array (M, 3) float — unit vectors ê_p
    batch_size : int — directions processed per batch to cap memory
    """
    grid = problem.grid
    be = grid.backend
    xp = be.xp
    k0 = problem.k0
    dv = grid.dv

    # polarization current J = eta @ u  →  (3, N1, N2, N3)
    eta = problem.eps_tensor                                 # (3, 3, N1, N2, N3)
    J = xp.einsum("ij...,j...->i...", eta, u)               # (3, N1, N2, N3)
    J_flat = J.reshape(3, -1)                               # (3, N)

    # grid coordinates (3, N)
    X, Y, Z = grid.coords()
    r_flat = xp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=0)  # (3, N)

    # move to numpy (works for both CPU and GPU backends)
    J_np = be.to_host(J_flat).astype(np.complex128)         # (3, N)
    r_np = be.to_host(r_flat).astype(np.float64)            # (3, N)
    dirs_np = np.asarray(directions, dtype=np.float64)      # (M, 3)
    M = len(dirs_np)
    F = np.zeros((M, 3), dtype=np.complex128)

    for start in range(0, M, batch_size):
        e_batch = dirs_np[start : start + batch_size]       # (b, 3)
        # dot(e_p, r) for every pair:  (b, 3) @ (3, N)  ->  (b, N)
        dot_er = e_batch @ r_np                             # (b, N)
        phase = np.exp(-1j * k0 * dot_er)                  # (b, N)
        # F_batch = dv * J_np @ phase.T  ->  (3, b), then transpose
        F[start : start + batch_size] = (dv * (J_np @ phase.T)).T   # (b, 3)

    return F
```

- [ ] **Step 4 — Create `src/em3d/farfield/__init__.py`**

```python
"""em3d.farfield — far-field scatter integral and RCS post-processing.

Public API
----------
scatter_integral(u, problem, directions, *, method="direct", batch_size=64) -> (M, 3) complex
rcs(u, problem, direction) -> float
rcs_plane(u, problem, n_phi=80, plane="xy", method="direct", batch_size=64) -> (phi, sigma)
"""
from __future__ import annotations

import numpy as np

from ..problem import Problem
from ._core import scatter_integral_direct

__all__ = ["scatter_integral", "rcs", "rcs_plane"]


def scatter_integral(
    u,
    problem: Problem,
    directions,
    *,
    method: str = "direct",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute the far-field scatter integral F(ê_p).

    F[m] = ΔV · Σ_q  η_q · u_q · exp(−ik₀ · ê_p[m] · r_q)

    Parameters
    ----------
    u          : array (3, N1, N2, N3) complex — E-field solution
    problem    : Problem — eps_tensor (η=ε−I), k0, grid
    directions : array (M, 3) float — unit observation vectors; a single (3,)
                 vector is automatically broadcast to (1, 3)
    method     : "direct"  — batched matmul, O(3NM) flops, O(N+batch·N) memory
                 "fft"     — 3D FFT + map_coordinates, O(9N log N + 9M) flops
    batch_size : directions per batch (used only for method="direct")

    Returns
    -------
    F : ndarray (M, 3) complex128 — always numpy, on CPU
    """
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[np.newaxis, :]
    if method == "direct":
        return scatter_integral_direct(u, problem, directions, batch_size=batch_size)
    elif method == "fft":
        from ._fft import scatter_integral_fft
        return scatter_integral_fft(u, problem, directions)
    else:
        raise ValueError(f"method must be 'direct' or 'fft', got {method!r}")


def rcs(u, problem: Problem, direction) -> float:
    """RCS for a single observation direction.

    σ(ê_p) = k₀⁴ / (16π²) · |ê_p × F(ê_p)|²

    Parameters
    ----------
    u         : array (3, N1, N2, N3) complex
    problem   : Problem
    direction : array (3,) float — unit vector (normalised internally)

    Returns
    -------
    float — non-negative RCS value
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    F = scatter_integral(u, problem, direction[np.newaxis, :])[0]  # (3,) complex128
    cross = np.cross(direction, F)                                  # (3,) complex128
    cross_norm_sq = float(np.real(np.dot(cross, cross.conj())))
    return problem.k0 ** 4 / (16.0 * np.pi ** 2) * cross_norm_sq


def rcs_plane(
    u,
    problem: Problem,
    n_phi: int = 80,
    plane: str = "xy",
    method: str = "direct",
    batch_size: int = 64,
) -> tuple:
    """RCS curve over n_phi equally-spaced directions in a coordinate plane.

    Parameters
    ----------
    u         : array (3, N1, N2, N3) complex
    problem   : Problem
    n_phi     : number of directions in [0, 2π)
    plane     : "xy" | "yz" | "xz"
    method    : "direct" | "fft"
    batch_size: used only for method="direct"

    Returns
    -------
    phi   : ndarray (n_phi,) — angles in radians
    sigma : ndarray (n_phi,) — RCS values ≥ 0
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    c, s, z = np.cos(phi), np.sin(phi), np.zeros_like(phi)
    if plane == "xy":
        dirs = np.stack([c, s, z], axis=-1)    # (n_phi, 3)
    elif plane == "yz":
        dirs = np.stack([z, c, s], axis=-1)
    elif plane == "xz":
        dirs = np.stack([c, z, s], axis=-1)
    else:
        raise ValueError(f"plane must be 'xy'|'yz'|'xz', got {plane!r}")

    F = scatter_integral(u, problem, dirs, method=method, batch_size=batch_size)  # (n_phi, 3)
    cross = np.cross(dirs, F)                                                      # (n_phi, 3)
    sigma = (problem.k0 ** 4 / (16.0 * np.pi ** 2)
             * np.real(np.einsum("mi,mi->m", cross, cross.conj())))               # (n_phi,)
    return phi, sigma
```

- [ ] **Step 5 — Run 4 tests, verify they pass**

```
pytest tests/test_farfield.py -v
```

Expected output (4 tests collected):
```
tests/test_farfield.py::test_zero_contrast[direct] PASSED
tests/test_farfield.py::test_rcs_nonnegative PASSED
tests/test_farfield.py::test_single_cell_analytic PASSED
tests/test_farfield.py::test_rcs_plane_shape PASSED
```

- [ ] **Step 6 — Commit**

```bash
git add src/em3d/farfield/__init__.py src/em3d/farfield/_core.py tests/test_farfield.py
git commit -m "feat(farfield): scatter_integral (direct), rcs, rcs_plane + 4 tests"
```

---

## Task 2: FFT backend + 2 additional tests

**Files:**
- Create: `src/em3d/farfield/_fft.py`
- Modify: `tests/test_farfield.py` (add `test_zero_contrast[fft]`, `test_fft_vs_direct`, `test_rcs_plane_symmetry`)

### Step 1 — Add 3 failing tests to `tests/test_farfield.py`

Append to the existing `tests/test_farfield.py` (after `test_rcs_plane_shape`):

```python
# --- Test 1 extended: zero contrast for fft method ---
# Already parametrized with @pytest.mark.parametrize("method", ["direct"]) above.
# Change the parametrize decorator in test_zero_contrast to:
#   @pytest.mark.parametrize("method", ["direct", "fft"])
# (update the existing decorator, don't add a second function)

# --- Test 4: fft vs direct agreement ---

def test_fft_vs_direct_agreement():
    """FFT backend matches direct to atol=1e-4 on an 8×8×8 grid."""
    be = _be()
    grid = Grid(N=(8, 8, 8), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)
    scalar = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,
        center=(0.0, 0.0, 0.0), radius=(0.3, 0.3, 0.3),
    )
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=1.0, orient=(0, 0, 1), amplitude=(1, 0, 0))
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=1.0,
                      volume=grid.dv * 512)
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


# --- Test 6: rcs_plane symmetry ---

def test_rcs_plane_symmetry():
    """Isotropic sphere, real eta, uniform x-field:
    sigma(phi) == sigma(phi + pi) in xy-plane (follows from eta real + sphere symmetry).
    """
    be = _be()
    N = (8, 8, 8)
    grid = Grid(N=N, L=(2.0, 2.0, 2.0), center=(0.0, 0.0, 0.0), backend=be)
    scalar = ellipsis_refraction(
        grid, eps_real=2.0, eps_imag=0.0,           # real eta → conjugate symmetry
        center=(0.0, 0.0, 0.0), radius=(0.4, 0.4, 0.4),
    )
    eta = apply_refraction(grid, scalar_eta=scalar)
    wave = flat_wave_vec(grid, k=0.1, orient=(0, 0, 1), amplitude=(1, 0, 0))
    # Uniform x-polarized field (no VIE solution needed — any field works for this identity)
    u = np.zeros((3,) + N, dtype=np.complex128)
    u[0] = 1.0                                       # u = (1,0,0) everywhere
    problem = Problem(grid=grid, eps_tensor=eta, wave=wave, k0=0.1,
                      volume=grid.dv * int(np.prod(N)))
    n_phi = 24
    phi, sigma = rcs_plane(u, problem, n_phi=n_phi, plane="xy")
    half = n_phi // 2                               # 12
    # sigma[i] corresponds to phi[i], sigma[i+half] to phi[i]+pi
    np.testing.assert_allclose(
        sigma[:half], sigma[half:], rtol=1e-10,
        err_msg="RCS not symmetric: sigma(phi) != sigma(phi+pi)",
    )
```

Also update the `@pytest.mark.parametrize` decorator on `test_zero_contrast` to include `"fft"`:
```python
@pytest.mark.parametrize("method", ["direct", "fft"])
def test_zero_contrast(method):
    ...
```

- [ ] **Step 2 — Run new tests, verify they fail**

```
pytest tests/test_farfield.py::test_zero_contrast[fft] \
       tests/test_farfield.py::test_fft_vs_direct_agreement \
       tests/test_farfield.py::test_rcs_plane_symmetry -v
```

Expected: all three FAIL with `NotImplementedError` or `ValueError: method must be 'direct' or 'fft'`.

- [ ] **Step 3 — Create `src/em3d/farfield/_fft.py`**

```python
"""Variant B: 3D FFT + map_coordinates scatter integral.

Algorithm
---------
1. Compute J_dv = eta @ u * dv  (shape 3×N1×N2×N3)
2. Apply fftn + fftshift on each of the 3 components
3. For each observation direction e_p = (ex, ey, ez), the sample
   coordinates in the fftshifted array are:
       ix = k0 * ex * Lx / (2π) + Nx/2
       iy = k0 * ey * Ly / (2π) + Ny/2
       iz = k0 * ez * Lz / (2π) + Nz/2
4. Interpolate (bilinear, order=1) real and imaginary parts separately
5. Apply phase correction for grid corner r0:
       F *= exp(-1j * k0 * (e_p @ r0))
   where r0 = [cx - Lx/2 + dx/2,  cy - Ly/2 + dy/2,  cz - Lz/2 + dz/2]
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ..problem import Problem


def scatter_integral_fft(
    u,
    problem: Problem,
    directions,
) -> np.ndarray:
    """Return F of shape (M, 3) as complex128 numpy array.

    Parameters
    ----------
    u          : array (3, N1, N2, N3) complex
    problem    : Problem
    directions : array (M, 3) float — unit vectors ê_p

    Returns
    -------
    F : ndarray (M, 3) complex128
    """
    grid = problem.grid
    be = grid.backend
    xp = be.xp
    k0 = problem.k0
    dv = grid.dv
    Nx, Ny, Nz = grid.N
    Lx, Ly, Lz = grid.L
    cx, cy, cz = grid.center

    # 1. Polarization current * dv  →  (3, Nx, Ny, Nz)
    eta = problem.eps_tensor                             # (3, 3, Nx, Ny, Nz)
    J = xp.einsum("ij...,j...->i...", eta, u)           # (3, Nx, Ny, Nz)
    J_dv = J * dv

    # 2. Move to numpy (handles CuPy transparently), then fftn + fftshift
    J_dv_np = be.to_host(J_dv).astype(np.complex128)    # (3, Nx, Ny, Nz)
    J_hat = np.fft.fftshift(
        np.fft.fftn(J_dv_np, axes=(-3, -2, -1)),
        axes=(-3, -2, -1),
    )                                                    # (3, Nx, Ny, Nz)

    # 3. Grid corner r0  (phase correction for non-zero centre)
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    r0 = np.array([cx - Lx / 2.0 + dx / 2.0,
                   cy - Ly / 2.0 + dy / 2.0,
                   cz - Lz / 2.0 + dz / 2.0])

    # 4. Fractional interpolation indices in the fftshifted array
    dirs_np = np.asarray(directions, dtype=np.float64)  # (M, 3)
    ix = k0 * dirs_np[:, 0] * Lx / (2.0 * np.pi) + Nx / 2.0  # (M,)
    iy = k0 * dirs_np[:, 1] * Ly / (2.0 * np.pi) + Ny / 2.0
    iz = k0 * dirs_np[:, 2] * Lz / (2.0 * np.pi) + Nz / 2.0
    coords = np.stack([ix, iy, iz], axis=0)             # (3, M)

    # 5. Bilinear interpolation component by component
    #    map_coordinates does not support complex arrays → split real/imag
    M = len(dirs_np)
    F = np.zeros((M, 3), dtype=np.complex128)
    for i in range(3):
        F[:, i] = (
            map_coordinates(J_hat[i].real, coords, order=1, mode="nearest")
            + 1j * map_coordinates(J_hat[i].imag, coords, order=1, mode="nearest")
        )

    # 6. Phase correction: multiply by exp(-1j * k0 * (e_p @ r0))
    phase_corr = np.exp(-1j * k0 * (dirs_np @ r0))     # (M,)
    F *= phase_corr[:, np.newaxis]

    return F
```

- [ ] **Step 4 — Run all 6 tests, verify they all pass**

```
pytest tests/test_farfield.py -v
```

Expected (6 tests — `test_zero_contrast` is now parametrized with 2 values, counting as 2 items):
```
tests/test_farfield.py::test_zero_contrast[direct]     PASSED
tests/test_farfield.py::test_zero_contrast[fft]        PASSED
tests/test_farfield.py::test_rcs_nonnegative           PASSED
tests/test_farfield.py::test_single_cell_analytic      PASSED
tests/test_farfield.py::test_fft_vs_direct_agreement   PASSED
tests/test_farfield.py::test_rcs_plane_shape           PASSED
tests/test_farfield.py::test_rcs_plane_symmetry        PASSED
```

(7 items total because `test_zero_contrast` is parametrized × 2.)

- [ ] **Step 5 — Commit**

```bash
git add src/em3d/farfield/_fft.py tests/test_farfield.py
git commit -m "feat(farfield): FFT+interpolation backend + 3 additional tests"
```

---

## Task 3: Wire into main `em3d` package

**Files:**
- Modify: `src/em3d/__init__.py`

- [ ] **Step 1 — Open `src/em3d/__init__.py` and add the farfield import**

Current file ends at:
```python
__version__ = "0.1.0"

__all__ = [
    "Backend",
    ...
    "TwoStep",
    "__version__",
]
```

Add `from . import farfield` after the existing imports and `"farfield"` to `__all__`. The updated imports block and `__all__` should look like:

```python
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
from . import farfield
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
    "farfield",
    "BaseSolver",
    "BiCGStab",
    "SIM",
    "SolverConfig",
    "SolverResult",
    "TwoStep",
    "__version__",
]
```

- [ ] **Step 2 — Verify the import works**

```
python -c "import em3d; print(em3d.farfield.rcs)"
```

Expected:
```
<function rcs at 0x...>
```

- [ ] **Step 3 — Run the full test suite**

```
pytest --tb=short -q
```

Expected: all existing tests still pass, 7 farfield tests pass, zero failures.

- [ ] **Step 4 — Commit**

```bash
git add src/em3d/__init__.py
git commit -m "feat: expose em3d.farfield in top-level package"
```
