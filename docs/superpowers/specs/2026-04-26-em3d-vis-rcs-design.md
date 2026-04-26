# em3d.vis — RCS Visualization Design

## Goal

Add `em3d.vis` module with two public functions for visualizing radar cross-section (RCS / ЭПР)
curves produced by `em3d.farfield.rcs_plane`. Supports both publication-quality figures and
interactive Jupyter use. `matplotlib` is an optional dependency.

---

## Scope

**In scope:**
- Cartesian σ(φ) plot
- Polar σ(φ) plot (linear and dB scales)
- Optional file saving
- Lazy matplotlib import with a clear error if not installed

**Out of scope (future):**
- E-field slice visualization
- Permittivity geometry visualization
- Interactive/animated plots

---

## Architecture

### Files

| File | Action |
|---|---|
| `src/em3d/vis.py` | Create — public visualization API |
| `src/em3d/__init__.py` | Add `from . import vis` and `"vis"` to `__all__` |
| `pyproject.toml` | Add `[project.optional-dependencies] vis = ["matplotlib>=3.7"]` |
| `tests/test_vis.py` | Create — headless tests with Agg backend |

### Dependency strategy

`matplotlib` is imported lazily inside each public function via `_require_matplotlib()`.
If not installed, raises `ImportError` with the message:
```
Visualization requires matplotlib. Install with: pip install em3d[vis]
```
This keeps `em3d` importable without matplotlib for users who only need the solver.

---

## Public API

### `plot_rcs`

```python
def plot_rcs(
    phi: np.ndarray,
    sigma: np.ndarray,
    *,
    title: str | None = None,
    filename: str | None = None,
) -> tuple:   # (matplotlib.figure.Figure, matplotlib.axes.Axes)
```

**Parameters:**
- `phi` — `(n,)` float array, angles in radians `[0, 2π)`
- `sigma` — `(n,)` float array, RCS values ≥ 0
- `title` — optional plot title string
- `filename` — if given, saves figure with `fig.savefig(filename, dpi=150, bbox_inches="tight")`

**Behaviour:**
- Creates a new Figure and Axes
- Plots `phi` on X axis (label: `"φ (rad)"`), `sigma` on Y axis (label: `"σ"`)
- Adds grid
- Applies `title` if provided
- Returns `(fig, ax)`

---

### `plot_rcs_polar`

```python
def plot_rcs_polar(
    phi: np.ndarray,
    sigma: np.ndarray,
    *,
    db: bool = False,
    title: str | None = None,
    filename: str | None = None,
) -> tuple:   # (matplotlib.figure.Figure, matplotlib.axes.Axes)
```

**Parameters:**
- `phi` — `(n,)` float array, angles in radians
- `sigma` — `(n,)` float array, RCS values ≥ 0
- `db` — if `False` (default): plot `sigma` directly as radius;
          if `True`: plot `10 * log10(sigma / sigma.max() + 1e-30)` (relative dB, max = 0 dB)
- `title` — optional title
- `filename` — optional save path

**Behaviour:**
- Creates Figure with `projection="polar"` axes
- `db=False`: radius = `sigma`; theta label absent (angles on ring)
- `db=True`: radius = `sigma_db`; adds ylabel `"σ (dB rel. max)"`
- Returns `(fig, ax)`

---

### Private helper

```python
def _require_matplotlib():
    """Import and return matplotlib.pyplot, or raise ImportError with install hint."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install em3d[vis]"
        ) from None
```

---

## `pyproject.toml` change

```toml
[project.optional-dependencies]
gpu = ["cupy-cuda12x>=13"]
vis = ["matplotlib>=3.7"]
dev = ["pytest>=8", "pytest-xdist>=3"]
```

---

## `src/em3d/__init__.py` change

Add to existing imports:
```python
from . import vis
```
Add `"vis"` to `__all__`.

---

## Tests (`tests/test_vis.py`)

All tests use `matplotlib.use("Agg")` at module level to avoid display requirements.

### `test_plot_rcs_returns_fig_ax`
Call `plot_rcs(phi, sigma)` with synthetic data. Assert return is `(Figure, Axes)`.
Assert `ax.lines` is non-empty.

### `test_plot_rcs_polar_linear`
Call `plot_rcs_polar(phi, sigma, db=False)`. Assert return is `(Figure, Axes)`.
Assert `ax.name == "polar"`.

### `test_plot_rcs_polar_db`
Call `plot_rcs_polar(phi, sigma, db=True)`. Assert plotted radius values are all ≤ 0.0
(since dB is normalized to max = 0 dB).

### `test_plot_rcs_saves_file`
Call `plot_rcs(phi, sigma, filename=tmp_path / "out.png")`. Assert file exists and size > 0.

### `test_missing_matplotlib_raises`
Patch `builtins.__import__` to raise `ImportError` for `matplotlib`. Call `plot_rcs(...)`.
Assert `ImportError` is raised and message contains `"pip install em3d[vis]"`.

---

## Usage examples

### Jupyter notebook

```python
from em3d.farfield import rcs_plane
from em3d.vis import plot_rcs, plot_rcs_polar
import matplotlib.pyplot as plt

phi, sigma = rcs_plane(u, problem, n_phi=180, plane="xy")

fig, ax = plot_rcs(phi, sigma, title="RCS xy-plane")
plt.show()

fig, ax = plot_rcs_polar(phi, sigma, db=True, title="RCS polar (dB)")
plt.show()
```

### Script with file output

```python
plot_rcs(phi, sigma, filename="rcs_cartesian.pdf")
plot_rcs_polar(phi, sigma, db=False, filename="rcs_polar.png")
```
