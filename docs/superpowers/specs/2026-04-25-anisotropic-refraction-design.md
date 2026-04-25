# Anisotropic Refraction Geometry Design

## Goal

Extend `cylinder_refraction`, `step_refraction`, and `ellipsis_refraction` in `src/em3d/refraction.py`
so that `eps_real` and `eps_imag` accept either a scalar `float` or a `(3,3)` array-like, and
always return a `(3, 3, Nx, Ny, Nz)` complex tensor field ready for use as `Problem.eps_tensor`.

## Motivation

The operator (`Operator.matvec`) and `Problem.eps_tensor` already use the full `(3,3,Nx,Ny,Nz)`
contrast tensor η = ε − I. The geometry convenience functions were scalar-only shortcuts.
Making them general-purpose removes the need for `apply_refraction(scalar_eta=...)` in the
common path and enables anisotropic scatterers without manual tensor assembly.

---

## Architecture

### File changed: `src/em3d/refraction.py`

#### New private helper: `_to_eta_mat`

```python
def _to_eta_mat(eps_real, eps_imag) -> np.ndarray:  # (3,3) complex128
```

Rules:
- Both arguments must be **the same kind**: both scalars, or both array-likes of shape `(3,3)`.
  Mixed (scalar + matrix) raises `TypeError`.
- **Scalar path**: `eps_real: float`, `eps_imag: float`
  → `eta = (eps_real - 1.0 + 1j * eps_imag) * np.eye(3, dtype=np.complex128)`
- **Matrix path**: `eps_real: array-like (3,3)`, `eps_imag: array-like (3,3)`
  → `E = np.asarray(eps_real, dtype=np.float64)`, `F = np.asarray(eps_imag, dtype=np.float64)`
  → shape validation: both must be `(3,3)`, else `ValueError`
  → `eta = (E - np.eye(3)) + 1j * F`
- Returns `np.ndarray` shape `(3,3)`, dtype `complex128`. Pure numpy, no backend dependency.

#### Updated geometry functions

All three functions change signature from `eps_real: float, eps_imag: float` to
`eps_real: float | array-like, eps_imag: float | array-like` and return `(3,3,Nx,Ny,Nz)`.

Internal pattern (same for all three; mask differs):

```python
eta_mat = _to_eta_mat(eps_real, eps_imag)   # (3,3) complex128, numpy
out = be.zeros((3, 3) + grid.N, kind="complex")   # pre-zeroed
for i in range(3):
    for j in range(3):
        out[i, j] = xp.where(mask, eta_mat[i, j], out[i, j])
return out   # (3, 3, Nx, Ny, Nz)
```

`apply_refraction` is **not changed**. Its `scalar_eta` path remains useful for users who build
scalar η fields manually (e.g., arithmetic on `(Nx,Ny,Nz)` arrays).

---

## Public API changes

### Before

```python
scalar = ellipsis_refraction(grid, eps_real=2.0, eps_imag=0.0,
                              center=(0,0,0), radius=(0.3,0.3,0.3))
# scalar.shape == (Nx, Ny, Nz)
eta = apply_refraction(grid, scalar_eta=scalar)
# eta.shape == (3, 3, Nx, Ny, Nz)
problem = Problem(..., eps_tensor=eta, ...)
```

### After — isotropic (scalar sugar preserved)

```python
eta = ellipsis_refraction(grid, eps_real=2.0, eps_imag=0.0,
                           center=(0,0,0), radius=(0.3,0.3,0.3))
# eta.shape == (3, 3, Nx, Ny, Nz)
problem = Problem(..., eps_tensor=eta, ...)
```

### After — anisotropic

```python
import numpy as np

eta = ellipsis_refraction(
    grid,
    eps_real=np.array([[2.0, 0.0, 0.0],
                       [0.0, 1.5, 0.0],
                       [0.0, 0.0, 1.2]]),
    eps_imag=np.zeros((3, 3)),
    center=(0.0, 0.0, 0.0),
    radius=(0.3, 0.3, 0.3),
)
# eta.shape == (3, 3, Nx, Ny, Nz); off-diagonal = 0 outside ellipsoid
problem = Problem(..., eps_tensor=eta, ...)
```

### Combining regions (still works via numpy arithmetic)

```python
eta = (
    ellipsis_refraction(grid, eps_real=2.0, eps_imag=0.0, center=(0,0,0), radius=(0.3,0.3,0.3))
    + cylinder_refraction(grid, eps_real=1.5, eps_imag=0.1, radius=0.2, axis="z")
)
# eta.shape == (3, 3, Nx, Ny, Nz); element-wise addition
problem = Problem(..., eps_tensor=eta, ...)
```

---

## Error handling

| Condition | Exception |
|---|---|
| `eps_real` is matrix but `eps_imag` is scalar (or vice versa) | `TypeError` |
| `eps_real` or `eps_imag` matrix shape ≠ `(3,3)` | `ValueError` |

---

## Files affected

| File | Change |
|---|---|
| `src/em3d/refraction.py` | Add `_to_eta_mat`; update 3 geometry functions |
| `tests/test_refraction.py` | Update all geometry-function tests (shape `(4,4,4)` → `(3,3,4,4,4)`); add anisotropic test; update `test_apply_refraction_scalar_returns_isotropic_tensor` to build scalar field manually |
| `tests/test_farfield.py` | Remove intermediate `apply_refraction(scalar_eta=...)` calls where geometry functions are used |

---

## Tests

### Updated tests in `test_refraction.py`

**`test_cylinder_mask_inside_outside`** — assert `eta.shape == (3,3,4,4,4)`,
check diagonal components `eta[0,0]`, `eta[1,1]`, `eta[2,2]` at centre cell equal `eps_real−1`,
off-diagonal `eta[0,1]` zero everywhere.

**`test_step_refraction`** — assert `eta.shape == (3,3,4,4,4)`,
check `eta[0,0,0,0,mid]` non-zero, `eta[0,0,0,0,0]` zero.

**`test_ellipsis_refraction`** — assert `eta.shape == (3,3,4,4,4)`,
check centre cell diagonal non-zero.

**`test_apply_refraction_scalar_returns_isotropic_tensor`** — build scalar field manually:
`scalar = np.full(grid.N, complex(eps_real - 1.0, 0.0))` and pass to `apply_refraction(scalar_eta=scalar)`.
Test stays to verify `apply_refraction` still works.

### New test: `test_ellipsis_anisotropic`

```python
def test_ellipsis_anisotropic(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    eps_r = np.diag([2.0, 1.5, 1.2])
    eps_i = np.zeros((3, 3))
    eta = ellipsis_refraction(grid, eps_real=eps_r, eps_imag=eps_i,
                               center=(0,0,0), radius=(0.3,0.3,0.3))
    assert eta.shape == (3, 3) + grid.N
    cx, cy, cz = grid.N[0]//2, grid.N[1]//2, grid.N[2]//2
    np.testing.assert_allclose(eta[0, 0, cx, cy, cz], 1.0)   # eps_xx − 1
    np.testing.assert_allclose(eta[1, 1, cx, cy, cz], 0.5)   # eps_yy − 1
    np.testing.assert_allclose(eta[2, 2, cx, cy, cz], 0.2)   # eps_zz − 1
    np.testing.assert_allclose(eta[0, 1, cx, cy, cz], 0.0)   # off-diagonal = 0
    # outside ellipsoid: corner cell
    np.testing.assert_allclose(eta[:, :, 0, 0, 0], np.zeros((3,3)))
```

### New test: `test_to_eta_mat_type_mismatch`

```python
def test_to_eta_mat_type_mismatch(backend_numpy_double):
    grid = _grid(backend_numpy_double)
    with pytest.raises(TypeError):
        ellipsis_refraction(grid, eps_real=2.0, eps_imag=np.zeros((3, 3)),
                             center=(0, 0, 0), radius=(0.1, 0.1, 0.1))
```
