import matplotlib
matplotlib.use("Agg")   # headless backend — must precede any pyplot import

import sys
import em3d
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from em3d.vis import plot_rcs, plot_rcs_polar, plot_field_slice


def _synthetic_data(n: int = 24):
    """Return (phi, sigma) with n points. sigma is always positive."""
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    sigma = 1.0 + 0.5 * np.cos(2.0 * phi)   # values in [0.5, 1.5]
    return phi, sigma


def test_plot_rcs_returns_fig_ax():
    """plot_rcs returns (Figure, Axes) and draws a line."""
    phi, sigma = _synthetic_data()
    fig, ax = plot_rcs(phi, sigma)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) > 0, "Expected at least one line on axes"


def test_plot_rcs_polar_linear():
    """plot_rcs_polar(db=False) returns polar axes with a line."""
    phi, sigma = _synthetic_data()
    fig, ax = plot_rcs_polar(phi, sigma, db=False)
    assert isinstance(fig, Figure)
    assert ax.name == "polar"
    assert len(ax.lines) > 0, "Expected at least one line on polar axes"


def test_plot_rcs_polar_db():
    """plot_rcs_polar(db=True) plots dB values: all radii <= 0."""
    phi, sigma = _synthetic_data()
    fig, ax = plot_rcs_polar(phi, sigma, db=True)
    assert ax.name == "polar"
    r_data = ax.lines[0].get_ydata()
    assert np.all(r_data <= 0.0), (
        f"dB values should be <= 0 (normalized to max), got max={r_data.max():.4f}"
    )


def test_plot_rcs_saves_file(tmp_path):
    """plot_rcs with filename= saves a non-empty PNG."""
    phi, sigma = _synthetic_data()
    out = tmp_path / "rcs.png"
    plot_rcs(phi, sigma, filename=str(out))
    assert out.exists(), f"Expected file at {out}"
    assert out.stat().st_size > 0, "Saved file is empty"


def test_missing_matplotlib_raises(monkeypatch):
    """When matplotlib.pyplot is unavailable, ImportError mentions pip install."""
    phi, sigma = _synthetic_data()
    # Setting sys.modules entry to None makes Python raise ImportError on import
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"pip install em3d\[vis\]"):
        plot_rcs(phi, sigma)


# ── Field visualization helpers ───────────────────────────────────────────

def _field_data(nx=8, ny=8, nz=8, seed=0):
    """Return (3, nx, ny, nz) complex128 field with random values."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, nx, ny, nz))
            + 1j * rng.standard_normal((3, nx, ny, nz)))


def _field_grid(n=8):
    """Return an n³ double-precision Grid."""
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
