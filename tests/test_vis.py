import matplotlib
matplotlib.use("Agg")   # headless backend — must precede any pyplot import

import sys
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from em3d.vis import plot_rcs, plot_rcs_polar


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
