# em3d.vis Field Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `plot_field_slice` (2-D cross-section with quiver + pcolormesh) and `plot_field_volume` (3-D quiver with rotation control) to the existing `src/em3d/vis.py`.

**Architecture:** Two new public functions and two private helpers (`_validate_u`, `_extract_part`) appended to the existing `vis.py`. Tests appended to `tests/test_vis.py` in two TDD rounds — slice first, then volume. No new files, no new dependencies (matplotlib is already the `[vis]` extra).

**Tech Stack:** Python 3.11, matplotlib ≥ 3.7 (Agg backend in tests), numpy, pytest

---

## File Map

- **Modify:** `tests/test_vis.py` — add `import em3d`, helpers `_field_data` / `_field_grid`, 10 new tests
- **Modify:** `src/em3d/vis.py` — add `_validate_u`, `_extract_part`, `plot_field_slice`, `plot_field_volume`; update `__all__`

---

## Task 1: Write failing tests for `plot_field_slice`

**Files:**
- Modify: `tests/test_vis.py`

Context: `plot_field_slice` does not exist yet. Adding it to the import line will cause
`ImportError` at collection — that is the expected red state. The existing 5 RCS tests
will also fail to collect; they turn green again once Task 2 lands.

- [ ] **Step 1: Update `tests/test_vis.py` — change the top of the file**

Change the import line and add `import em3d` so the file begins:

```python
import matplotlib
matplotlib.use("Agg")   # headless backend — must precede any pyplot import

import sys
import em3d
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from em3d.vis import plot_rcs, plot_rcs_polar, plot_field_slice
```

- [ ] **Step 2: Append helpers and slice tests to `tests/test_vis.py`**

Add the following after the existing 5 RCS test functions:

```python
# ── Field visualization helpers ───────────────────────────────────────────

def _field_data(nx=8, ny=8, nz=8, seed=0):
    """Return (3, nx, ny, nz) complex128 field with random values."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, nx, ny, nz))
            + 1j * rng.standard_normal((3, nx, ny, nz)))


def _field_grid(n=8):
    """Return an 8³ (or n³) double-precision Grid."""
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=(n, n, n), L=(1.0, 1.0, 1.0),
                     center=(0.0, 0.0, 0.0), backend=be)


# ── plot_field_slice tests ────────────────────────────────────────────────

def test_plot_field_slice_returns_fig_ax():
    """plot_field_slice returns (Figure, Axes) and draws a pcolormesh."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_slice(u, grid)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0, "Expected pcolormesh in ax.collections"


def test_plot_field_slice_all_planes():
    """plane='xy'/'xz'/'yz' all succeed and return (Figure, Axes)."""
    u = _field_data()
    grid = _field_grid()
    for plane in ["xy", "xz", "yz"]:
        fig, ax = plot_field_slice(u, grid, plane=plane)
        assert isinstance(fig, Figure), f"plane={plane!r} did not return Figure"


def test_plot_field_slice_all_parts():
    """part='real'/'imag'/'abs' all succeed without error."""
    u = _field_data()
    grid = _field_grid()
    for part in ["real", "imag", "abs"]:
        fig, ax = plot_field_slice(u, grid, part=part)
        assert isinstance(fig, Figure), f"part={part!r} did not return Figure"


def test_plot_field_slice_stride():
    """stride=2 succeeds and returns (Figure, Axes)."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_slice(u, grid, stride=2)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_plot_field_slice_saves_file(tmp_path):
    """filename= saves a non-empty PNG."""
    u = _field_data()
    grid = _field_grid()
    out = tmp_path / "field_slice.png"
    plot_field_slice(u, grid, filename=str(out))
    assert out.exists(), f"Expected file at {out}"
    assert out.stat().st_size > 0, "Saved PNG is empty"


def test_plot_field_slice_invalid_inputs():
    """Bad arguments raise ValueError with informative messages."""
    u = _field_data()
    grid = _field_grid()

    with pytest.raises(ValueError, match="plane"):
        plot_field_slice(u, grid, plane="ab")

    with pytest.raises(ValueError, match="part"):
        plot_field_slice(u, grid, part="magnitude")

    with pytest.raises(ValueError, match="stride"):
        plot_field_slice(u, grid, stride=0)

    with pytest.raises(ValueError, match="out of range"):
        plot_field_slice(u, grid, plane="xy", idx=100)

    with pytest.raises(ValueError, match=r"\(3, Nx, Ny, Nz\)"):
        plot_field_slice(np.zeros((2, 8, 8, 8)), grid)
```

- [ ] **Step 3: Run tests — verify they FAIL with ImportError**

```
cd C:\Users\user\Documents\ClaudeProjects\ClaudeProject && py -m pytest tests/test_vis.py -v
```

Expected: collection error `ImportError: cannot import name 'plot_field_slice' from 'em3d.vis'`.

---

## Task 2: Implement `plot_field_slice` and private helpers

**Files:**
- Modify: `src/em3d/vis.py`

Context: `_require_matplotlib()` is already defined in `vis.py`. The new helpers and
`plot_field_slice` are appended after the existing `plot_rcs_polar`. The `__all__` list
is updated to expose `plot_field_slice`.

Key details:
- `grid.coords()` returns three `(Nx, Ny, Nz)` arrays via `np.meshgrid(..., indexing="ij")`.
- `grid.x`, `grid.y`, `grid.z` are 1-D arrays of shape `(Nx,)`, `(Ny,)`, `(Nz,)`.
- `grid.dv` is a float; `grid.dv ** (1/3)` gives a representative cell side length.
- `ax.quiver(X, Y, U, V, C, cmap=..., norm=..., scale=..., scale_units="xy")` uses
  the positional `C` array for per-arrow colouring — no need for `ScalarMappable`.
- `pcolormesh` with `shading="auto"` requires coordinate arrays of the same shape as data.

- [ ] **Step 4: Update `__all__` in `src/em3d/vis.py`**

Replace the existing `__all__` line:

```python
__all__ = ["plot_rcs", "plot_rcs_polar", "plot_field_slice"]
```

- [ ] **Step 5: Append helpers and `plot_field_slice` to `src/em3d/vis.py`**

Add the following after `plot_rcs_polar`:

```python
# ── Private helpers for field visualization ───────────────────────────────

def _validate_u(u) -> np.ndarray:
    """Assert u has shape (3, Nx, Ny, Nz) and return as numpy array."""
    u = np.asarray(u)
    if u.ndim != 4:
        raise ValueError(
            f"u must have shape (3, Nx, Ny, Nz), got ndim={u.ndim}"
        )
    if u.shape[0] != 3:
        raise ValueError(
            f"u must have shape (3, Nx, Ny, Nz), got shape={u.shape}"
        )
    return u


def _extract_part(u: np.ndarray, part: str) -> np.ndarray:
    """Return real-valued (3, Nx, Ny, Nz) array for the chosen part."""
    if part == "real":
        return u.real.copy()
    if part == "imag":
        return u.imag.copy()
    if part == "abs":
        return np.abs(u)
    raise ValueError(f"part must be 'real', 'imag', or 'abs', got {part!r}")


# ── Public field visualization ────────────────────────────────────────────

def plot_field_slice(
    u,
    grid,
    *,
    plane: str = "xy",
    idx: int | None = None,
    part: str = "real",
    stride: int = 1,
    cmap: str = "RdBu_r",
    title: str | None = None,
    filename: str | None = None,
) -> tuple:
    """2-D cross-section of the complex vector field u.

    Parameters
    ----------
    u        : array (3, Nx, Ny, Nz) complex — field from solver result.u
    grid     : em3d.Grid
    plane    : "xy" | "xz" | "yz" — orientation of the slice
    idx      : index along the normal axis; None uses N//2 (middle)
    part     : "real" | "imag" | "abs" — which part of u to display
    stride   : show every stride-th grid point per in-plane axis (default 1)
    cmap     : matplotlib colormap name (default "RdBu_r")
    title    : optional figure title
    filename : if given, save figure to this path at dpi=150

    Returns
    -------
    (fig, ax) — matplotlib Figure and 2-D Axes
    """
    plt = _require_matplotlib()
    import matplotlib.colors as mcolors

    u = _validate_u(u)
    if plane not in ("xy", "xz", "yz"):
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    if part not in ("real", "imag", "abs"):
        raise ValueError(f"part must be 'real', 'imag', or 'abs', got {part!r}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    F = _extract_part(u, part)          # (3, Nx, Ny, Nz), float64

    # Select slice axis, 1-D coordinate arrays, and in-plane F components
    if plane == "xy":
        axis_len = F.shape[3]           # Nz
        if idx is None:
            idx = axis_len // 2
        if idx < 0 or idx >= axis_len:
            raise ValueError(f"idx {idx} out of range [0, {axis_len})")
        horiz = np.asarray(grid.x)      # (Nx,)
        vert  = np.asarray(grid.y)      # (Ny,)
        F2d   = F[:, :, :, idx]         # (3, Nx, Ny)
        hlabel, vlabel = "x", "y"
    elif plane == "xz":
        axis_len = F.shape[2]           # Ny
        if idx is None:
            idx = axis_len // 2
        if idx < 0 or idx >= axis_len:
            raise ValueError(f"idx {idx} out of range [0, {axis_len})")
        horiz = np.asarray(grid.x)      # (Nx,)
        vert  = np.asarray(grid.z)      # (Nz,)
        F2d   = F[:, :, idx, :]         # (3, Nx, Nz)
        hlabel, vlabel = "x", "z"
    else:  # yz
        axis_len = F.shape[1]           # Nx
        if idx is None:
            idx = axis_len // 2
        if idx < 0 or idx >= axis_len:
            raise ValueError(f"idx {idx} out of range [0, {axis_len})")
        horiz = np.asarray(grid.y)      # (Ny,)
        vert  = np.asarray(grid.z)      # (Nz,)
        F2d   = F[:, idx, :, :]         # (3, Ny, Nz)
        hlabel, vlabel = "y", "z"

    # Background: norm of all three components on this slice
    norm2d = np.sqrt(F2d[0] ** 2 + F2d[1] ** 2 + F2d[2] ** 2)  # (Nh, Nv)
    vmax = float(norm2d.max()) if norm2d.max() > 0 else 1.0

    fig, ax = plt.subplots()

    # pcolormesh background
    H, V = np.meshgrid(horiz, vert, indexing="ij")
    pcm = ax.pcolormesh(H, V, norm2d, cmap=cmap, vmin=0.0, vmax=vmax,
                        shading="auto")
    fig.colorbar(pcm, ax=ax)

    # Quiver arrows with stride decimation
    s = stride
    Hs  = horiz[::s]
    Vs  = vert[::s]
    U2d = F2d[0, ::s, ::s]             # in-plane component 1
    V2d = F2d[1, ::s, ::s]             # in-plane component 2

    F_max = float(np.sqrt(U2d ** 2 + V2d ** 2).max())
    if F_max > 0:
        dh    = float(grid.dv ** (1.0 / 3.0))
        scale = F_max / (stride * dh * 0.9)
        # Per-arrow colour via positional C array
        arrow_norm = np.sqrt(U2d ** 2 + V2d ** 2)
        Hg, Vg = np.meshgrid(Hs, Vs, indexing="ij")
        ax.quiver(
            Hg, Vg, U2d, V2d, arrow_norm,
            cmap=cmap,
            norm=mcolors.Normalize(vmin=0.0, vmax=F_max),
            scale=scale,
            scale_units="xy",
        )

    ax.set_xlabel(hlabel)
    ax.set_ylabel(vlabel)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax
```

- [ ] **Step 6: Run slice tests — verify all 11 pass**

```
cd C:\Users\user\Documents\ClaudeProjects\ClaudeProject && py -m pytest tests/test_vis.py -v
```

Expected: 11 passed (5 existing RCS tests + 6 new slice tests).

- [ ] **Step 7: Commit**

```
git add src/em3d/vis.py tests/test_vis.py
git commit -m "feat(vis): add plot_field_slice with pcolormesh background and auto-scaled quiver"
```

---

## Task 3: Write failing tests for `plot_field_volume`

**Files:**
- Modify: `tests/test_vis.py`

Context: `plot_field_volume` does not exist yet. Adding it to the import causes ImportError.

- [ ] **Step 8: Update the import line in `tests/test_vis.py`**

Change:
```python
from em3d.vis import plot_rcs, plot_rcs_polar, plot_field_slice
```
To:
```python
from em3d.vis import plot_rcs, plot_rcs_polar, plot_field_slice, plot_field_volume
```

- [ ] **Step 9: Append volume tests to `tests/test_vis.py`**

Add after the last slice test:

```python
# ── plot_field_volume tests ───────────────────────────────────────────────

def test_plot_field_volume_returns_fig_ax3d():
    """plot_field_volume returns (Figure, Axes3D) with ax.name == '3d'."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_volume(u, grid, stride=2)
    assert isinstance(fig, Figure)
    assert ax.name == "3d"


def test_plot_field_volume_warns_large_grid():
    """stride=1 on a 16³ grid (4096 arrows) triggers UserWarning about stride."""
    u = _field_data(nx=16, ny=16, nz=16)
    grid = _field_grid(n=16)
    with pytest.warns(UserWarning, match="stride"):
        plot_field_volume(u, grid, stride=1)


def test_plot_field_volume_all_parts():
    """part='real'/'imag'/'abs' all run without error."""
    u = _field_data()
    grid = _field_grid()
    for part in ["real", "imag", "abs"]:
        fig, ax = plot_field_volume(u, grid, part=part, stride=2)
        assert isinstance(fig, Figure), f"part={part!r} did not return Figure"


def test_plot_field_volume_view_angles():
    """elev and azim are applied via view_init and readable back from the axes."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_volume(u, grid, stride=2, elev=45.0, azim=30.0)
    assert abs(ax.elev - 45.0) < 1.0, f"Expected elev≈45, got {ax.elev}"
    assert abs(ax.azim - 30.0) < 1.0, f"Expected azim≈30, got {ax.azim}"
```

- [ ] **Step 10: Run tests — verify they FAIL with ImportError**

```
cd C:\Users\user\Documents\ClaudeProjects\ClaudeProject && py -m pytest tests/test_vis.py -v
```

Expected: collection error `ImportError: cannot import name 'plot_field_volume' from 'em3d.vis'`.

---

## Task 4: Implement `plot_field_volume`

**Files:**
- Modify: `src/em3d/vis.py`

Context: `_validate_u` and `_extract_part` are already defined (Task 2).
`grid.coords()` returns three `(Nx, Ny, Nz)` arrays; the result may be a CuPy array —
wrap with `np.asarray()` before passing to matplotlib.
`ax.quiver` in 3-D accepts `colors=` as an `(N, 4)` RGBA array for per-arrow colouring.
`ax.view_init(elev=elev, azim=azim)` sets the camera and writes back to `ax.elev` / `ax.azim`.

- [ ] **Step 11: Update `__all__` in `src/em3d/vis.py`**

Replace:
```python
__all__ = ["plot_rcs", "plot_rcs_polar", "plot_field_slice"]
```
With:
```python
__all__ = ["plot_rcs", "plot_rcs_polar", "plot_field_slice", "plot_field_volume"]
```

- [ ] **Step 12: Append `plot_field_volume` to `src/em3d/vis.py`**

Add after `plot_field_slice`:

```python
def plot_field_volume(
    u,
    grid,
    *,
    part: str = "real",
    stride: int = 2,
    elev: float = 30.0,
    azim: float = -60.0,
    cmap: str = "RdBu_r",
    title: str | None = None,
    filename: str | None = None,
) -> tuple:
    """3-D quiver plot of the complex vector field u.

    Parameters
    ----------
    u        : array (3, Nx, Ny, Nz) complex — field from solver result.u
    grid     : em3d.Grid
    part     : "real" | "imag" | "abs" — which part of u to display
    stride   : show every stride-th point per axis (default 2)
    elev     : elevation angle in degrees for 3-D view (default 30)
    azim     : azimuth angle in degrees for 3-D view (default -60)
    cmap     : matplotlib colormap name (default "RdBu_r")
    title    : optional figure title
    filename : if given, save figure to this path at dpi=150

    Returns
    -------
    (fig, ax) — matplotlib Figure and Axes3D

    Notes
    -----
    Arrow colours encode the local norm ‖(U, V, W)‖ through `cmap`.
    If the decimated grid has more than 2 000 arrows a UserWarning is emitted.

    Examples
    --------
    >>> fig, ax = plot_field_volume(u, grid, elev=45, azim=30)   # isometric
    >>> fig, ax = plot_field_volume(u, grid, elev=90, azim=0)    # top-down
    >>> fig, ax = plot_field_volume(u, grid, elev=0,  azim=0)    # front view
    """
    import warnings
    plt = _require_matplotlib()
    import matplotlib.colors as mcolors

    u = _validate_u(u)
    if part not in ("real", "imag", "abs"):
        raise ValueError(f"part must be 'real', 'imag', or 'abs', got {part!r}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    F = _extract_part(u, part)          # (3, Nx, Ny, Nz), float64

    sl = (slice(None, None, stride),) * 3
    X, Y, Z = grid.coords()
    Xs = np.asarray(X[sl])
    Ys = np.asarray(Y[sl])
    Zs = np.asarray(Z[sl])
    U  = np.asarray(F[0][sl])
    V  = np.asarray(F[1][sl])
    W  = np.asarray(F[2][sl])

    n_arrows = Xs.size
    if n_arrows > 2_000:
        warnings.warn(
            f"plot_field_volume: {n_arrows} arrows after decimation; "
            f"consider increasing stride",
            UserWarning,
            stacklevel=2,
        )

    # Arrow length: longest arrow spans 90 % of one decimated cell
    dv13     = float(grid.dv ** (1.0 / 3.0))
    norm_max = float(np.sqrt(U ** 2 + V ** 2 + W ** 2).max())
    if norm_max > 0:
        length = stride * dv13 * 0.9 / norm_max
    else:
        length = stride * dv13

    # Per-arrow colour via local norm mapped through cmap
    arrow_norms = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=norm_max if norm_max > 0 else 1.0),
    )
    colors_rgba = sm.to_rgba(arrow_norms.ravel())   # (N, 4)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.quiver(
        Xs.ravel(), Ys.ravel(), Zs.ravel(),
        U.ravel(),  V.ravel(),  W.ravel(),
        length=length, normalize=False, colors=colors_rgba,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    return fig, ax
```

- [ ] **Step 13: Run full test suite — verify all 15 vis tests pass**

```
cd C:\Users\user\Documents\ClaudeProjects\ClaudeProject && py -m pytest tests/test_vis.py -v
```

Expected:
```
tests/test_vis.py::test_plot_rcs_returns_fig_ax              PASSED
tests/test_vis.py::test_plot_rcs_polar_linear                PASSED
tests/test_vis.py::test_plot_rcs_polar_db                    PASSED
tests/test_vis.py::test_plot_rcs_saves_file                  PASSED
tests/test_vis.py::test_missing_matplotlib_raises            PASSED
tests/test_vis.py::test_plot_field_slice_returns_fig_ax      PASSED
tests/test_vis.py::test_plot_field_slice_all_planes          PASSED
tests/test_vis.py::test_plot_field_slice_all_parts           PASSED
tests/test_vis.py::test_plot_field_slice_stride              PASSED
tests/test_vis.py::test_plot_field_slice_saves_file          PASSED
tests/test_vis.py::test_plot_field_slice_invalid_inputs      PASSED
tests/test_vis.py::test_plot_field_volume_returns_fig_ax3d   PASSED
tests/test_vis.py::test_plot_field_volume_warns_large_grid   PASSED
tests/test_vis.py::test_plot_field_volume_all_parts          PASSED
tests/test_vis.py::test_plot_field_volume_view_angles        PASSED
```

- [ ] **Step 14: Run full suite — verify nothing else broke**

```
cd C:\Users\user\Documents\ClaudeProjects\ClaudeProject && py -m pytest -v --tb=short
```

Expected: all previously passing tests still pass (75 + 10 new = 85 passed, 1 skipped).

- [ ] **Step 15: Commit**

```
git add src/em3d/vis.py tests/test_vis.py
git commit -m "feat(vis): add plot_field_volume with 3D quiver, stride, and elev/azim rotation"
```
