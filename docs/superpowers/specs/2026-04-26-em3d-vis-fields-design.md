# em3d.vis — Field Visualization Design

## Goal

Extend `em3d.vis` with two functions for visualizing the complex vector field `u` (shape
`(3, Nx, Ny, Nz)`, dtype `complex128`) produced by the solver. Supports 2D cross-section
views with quiver overlay and a 3D quiver volume view. Pure matplotlib — no extra
dependencies beyond the existing `[vis]` optional group.

---

## Scope

**In scope:**
- `plot_field_slice` — 2D cross-section: `pcolormesh` background + `quiver` arrows
- `plot_field_volume` — 3D quiver via `Axes3D`
- `part` selector: `"real"` / `"imag"` / `"abs"` (per-component absolute value)
- Stride-based decimation for both functions
- Auto-scaling arrows so the longest arrow ≤ 90% of decimated grid spacing
- Arrow colour = local vector norm through `cmap`
- Default colormap: `"RdBu_r"` (red-white-blue diverging)
- Performance warning in `plot_field_volume` when arrow count > 2 000
- Optional file save via `filename=`

**Out of scope:**
- Multi-slice overview (caller calls `plot_field_slice` multiple times)
- Isosurfaces / volume rendering (pyvista/mayavi)
- Animations / interactive widgets

---

## Architecture

### Files

| File | Action |
|---|---|
| `src/em3d/vis.py` | Add `plot_field_slice`, `plot_field_volume`; update `__all__` |
| `tests/test_vis.py` | Append 9 new tests |

No new files. Both functions follow the existing lazy-import pattern via
`_require_matplotlib()`.

---

## Public API

### `plot_field_slice`

```python
def plot_field_slice(
    u,
    grid: Grid,
    *,
    plane: str = "xy",          # "xy" | "xz" | "yz"
    idx: int | None = None,     # slice index along normal axis; None → N//2
    part: str = "real",         # "real" | "imag" | "abs"
    stride: int = 1,            # decimate: show every stride-th point per axis
    cmap: str = "RdBu_r",
    title: str | None = None,
    filename: str | None = None,
) -> tuple:                     # (fig, ax)
```

**Behaviour:**

The `part` selector extracts the working field `F`:
- `"real"` → `F = np.asarray(u).real`  shape `(3, Nx, Ny, Nz)`
- `"imag"` → `F = np.asarray(u).imag`  shape `(3, Nx, Ny, Nz)`
- `"abs"`  → `F = np.abs(np.asarray(u))`  shape `(3, Nx, Ny, Nz)`

For `plane="xy"` at z-index `idx` (analogously for `"xz"` and `"yz"`):

| Element | Description |
|---|---|
| Horizontal axis | `grid.x` |
| Vertical axis | `grid.y` |
| In-plane arrow components | `F[0, :, :, idx]`, `F[1, :, :, idx]` |
| Background scalar | `norm2d = √(F[0]² + F[1]² + F[2]²)` at `idx` |

**Background (`pcolormesh`):**
- `cmap=cmap`
- All parts: `vmin=0, vmax=norm2d.max()` — the norm is always ≥ 0; using `RdBu_r`
  the range 0→max maps to white→blue, which cleanly shows field intensity.
- Colorbar added automatically

**Arrows (`quiver`)** with stride decimation `s = stride`:
```python
X2d = grid.x[::s],  Y2d = grid.y[::s]          # 1-D coordinate arrays
U2d = F[0, ::s, ::s, idx]                        # in-plane x-component
V2d = F[1, ::s, ::s, idx]                        # in-plane y-component
```

Arrow scale: `scale = F_plane_max / (stride * dx * 0.9)` where
`F_plane_max = max(√(U2d² + V2d²))` and `dx = grid.L[0] / grid.N[0]`.
Passed as `scale=scale` to `ax.quiver` (matplotlib units-per-data).
If `F_plane_max == 0` arrows are omitted (all-zero field).

Arrow colour: local norm `√(U2d² + V2d²)` mapped through `cmap` via `ScalarMappable`.

**Axes:** `xlabel`/`ylabel` set to the two axis names (`"x"/"y"`, etc.);
`ax.set_aspect("equal")`.

**Returns:** `(fig, ax)` — 2-D `Axes`.

---

### `plot_field_volume`

```python
def plot_field_volume(
    u,
    grid: Grid,
    *,
    part: str = "real",
    stride: int = 2,
    elev: float = 30.0,         # elevation angle in degrees (matplotlib default)
    azim: float = -60.0,        # azimuth angle in degrees (matplotlib default)
    cmap: str = "RdBu_r",
    title: str | None = None,
    filename: str | None = None,
) -> tuple:                     # (fig, ax3d)
```

**Behaviour:**

Same `part` → `F` extraction as `plot_field_slice`.

Decimation with `s = stride`:
```python
sl = (slice(None, None, s),) * 3
X, Y, Z = grid.coords()          # each (Nx, Ny, Nz)
Xs, Ys, Zs = X[sl], Y[sl], Z[sl]
U, V, W = F[0][sl], F[1][sl], F[2][sl]
```

**Performance warning:** if `Xs.size > 2_000`, emit:
```
UserWarning: plot_field_volume: N arrows after decimation ({n}); consider increasing stride
```

**Arrow length:**
```python
dv13 = grid.dv ** (1/3)
norm_max = np.sqrt(U**2 + V**2 + W**2).max()
length = stride * dv13 * 0.9 / norm_max   if norm_max > 0 else stride * dv13
```
Passed as `length=length` and `normalize=False` to `ax.quiver`.

**Arrow colour:** per-arrow norm `√(U² + V² + W²)`, normalised to `[0, 1]`,
mapped through `cmap` → RGBA array passed as `colors=` to `ax.quiver`.

**View angle:** `ax.view_init(elev=elev, azim=azim)` is called after drawing.
Defaults (`elev=30, azim=-60`) match matplotlib's own defaults so existing calls
are unaffected; the user overrides to any angle:

```python
fig, ax = plot_field_volume(u, grid, elev=45, azim=30)   # isometric-ish
fig, ax = plot_field_volume(u, grid, elev=90, azim=0)    # top-down (xy projection)
fig, ax = plot_field_volume(u, grid, elev=0,  azim=0)    # front view (xz projection)
```

**Axes:** `ax.set_xlabel("x")`, `ax.set_ylabel("y")`, `ax.set_zlabel("z")`.

**Returns:** `(fig, ax)` — 3-D `Axes3D`.

---

## Error Handling

| Condition | Exception |
|---|---|
| `u.ndim != 4` | `ValueError: u must have shape (3, Nx, Ny, Nz), got ndim={u.ndim}` |
| `u.shape[0] != 3` | `ValueError: u must have shape (3, Nx, Ny, Nz), got shape={u.shape}` |
| `plane` not in `{"xy","xz","yz"}` | `ValueError: plane must be 'xy', 'xz', or 'yz'` |
| `part` not in `{"real","imag","abs"}` | `ValueError: part must be 'real', 'imag', or 'abs'` |
| `stride < 1` | `ValueError: stride must be >= 1` |
| `idx` out of range for its axis | `ValueError: idx {idx} out of range [0, {N})` |
| matplotlib unavailable | `ImportError: pip install em3d[vis]` (via `_require_matplotlib()`) |

---

## Private Helpers

### `_extract_part(u, part) -> np.ndarray`

```python
def _extract_part(u, part: str) -> np.ndarray:
    """Convert complex field to real-valued (3, Nx, Ny, Nz) array."""
    u = np.asarray(u)
    if part == "real":
        return u.real.copy()
    if part == "imag":
        return u.imag.copy()
    if part == "abs":
        return np.abs(u)
    raise ValueError(f"part must be 'real', 'imag', or 'abs', got {part!r}")
```

### `_validate_u(u)`

```python
def _validate_u(u) -> np.ndarray:
    u = np.asarray(u)
    if u.ndim != 4:
        raise ValueError(f"u must have shape (3, Nx, Ny, Nz), got ndim={u.ndim}")
    if u.shape[0] != 3:
        raise ValueError(f"u must have shape (3, Nx, Ny, Nz), got shape={u.shape}")
    return u
```

---

## Tests (`tests/test_vis.py` — append)

Synthetic data helper added at module level:

```python
def _field_data(nx=8, ny=8, nz=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, nx, ny, nz))
            + 1j * rng.standard_normal((3, nx, ny, nz)))
```

Grid helper:

```python
def _field_grid():
    import em3d
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=(8, 8, 8), L=(1.0, 1.0, 1.0),
                     center=(0.0, 0.0, 0.0), backend=be)
```

### Tests

**`test_plot_field_slice_returns_fig_ax`**
Call `plot_field_slice(u, grid)` with defaults. Assert `isinstance(fig, Figure)`,
`isinstance(ax, Axes)`. Assert collection list non-empty (pcolormesh present).

**`test_plot_field_slice_all_planes`**
Call `plot_field_slice(u, grid, plane=p)` for each `p` in `["xy", "xz", "yz"]`.
Assert no exception raised, returns `(Figure, Axes)`.

**`test_plot_field_slice_all_parts`**
Call `plot_field_slice(u, grid, part=p)` for each `p` in `["real", "imag", "abs"]`.
Assert no exception raised.

**`test_plot_field_slice_stride`**
Call `plot_field_slice(u, grid, stride=2)`. Assert the quiver `U` array has shape
`≤ (4, 4)` (prune to half per axis on 8×8 grid).

**`test_plot_field_slice_saves_file`**
Call `plot_field_slice(u, grid, filename=str(tmp_path / "field.png"))`.
Assert file exists and `stat().st_size > 0`.

**`test_plot_field_slice_invalid_inputs`**
Assert `ValueError` for: `plane="ab"`, `part="magnitude"`, `stride=0`,
`idx=100` on 8³ grid, `u` with wrong shape `(2, 8, 8, 8)`.

**`test_plot_field_volume_returns_fig_ax3d`**
Call `plot_field_volume(u, grid, stride=2)`. Assert `isinstance(fig, Figure)`,
`ax.name == "3d"`.

**`test_plot_field_volume_warns_large_grid`**
Build `u` shape `(3, 16, 16, 16)` and matching grid; call `plot_field_volume`
with `stride=1`. Assert `pytest.warns(UserWarning, match="stride")`.

**`test_plot_field_volume_all_parts`**
Call `plot_field_volume(u, grid, part=p, stride=2)` for each `p` in
`["real", "imag", "abs"]`. Assert no exception.

**`test_plot_field_volume_view_angles`**
Call `plot_field_volume(u, grid, stride=2, elev=45.0, azim=30.0)`.
Assert `abs(ax.elev - 45.0) < 1.0` and `abs(ax.azim - 30.0) < 1.0`.

---

## Usage Examples

### Jupyter — inspect a solved field

```python
from em3d.vis import plot_field_slice, plot_field_volume
import matplotlib.pyplot as plt

u = np.asarray(result.u)   # (3, Nx, Ny, Nz) complex128

# 2-D cross-section: real part, xy-plane at middle z
fig, ax = plot_field_slice(u, grid, plane="xy", part="real", stride=2)
plt.show()

# imaginary part, xz-plane
fig, ax = plot_field_slice(u, grid, plane="xz", part="imag", idx=16)
plt.show()

# per-component magnitudes, full cross-section
fig, ax = plot_field_slice(u, grid, plane="yz", part="abs")
plt.show()

# 3-D volume quiver (decimate 4x for large grids)
fig, ax = plot_field_volume(u, grid, part="real", stride=4)
plt.show()
```

### Script with file output

```python
plot_field_slice(u, grid, plane="xy", part="real",
                 stride=2, filename="field_xy_real.png")
plot_field_volume(u, grid, part="abs", stride=3,
                  filename="field_volume.png")
```
