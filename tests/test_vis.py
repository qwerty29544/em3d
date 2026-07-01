import matplotlib
matplotlib.use("Agg")   # headless backend — must precede any pyplot import

import sys
import em3d
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from em3d.vis import (
    plot_gamma0_spectrum,
    plot_field_scalar_slice,
    plot_field_scalar_volume,
    plot_field_slice,
    plot_field_vector_slice,
    plot_field_vector_volume,
    plot_field_volume,
    plot_rcs,
    plot_rcs_comparison,
    plot_rcs_comparison_polar,
    plot_rcs_polar,
)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Keep visualization tests isolated from pyplot's global figure registry."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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


def test_plot_rcs_polar_db_zero_sigma_is_finite():
    """All-zero RCS has no relative dB scale, but must not produce NaN."""
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    sigma = np.zeros_like(phi)
    fig, ax = plot_rcs_polar(phi, sigma, db=True)
    assert ax.name == "polar"
    r_data = ax.lines[0].get_ydata()
    assert np.all(np.isfinite(r_data))


def test_plot_rcs_comparison_draws_two_cartesian_curves():
    """plot_rcs_comparison draws numerical and Mie curves on one Cartesian axis."""
    phi, sigma_mie = _synthetic_data()
    sigma_num = sigma_mie * 1.2
    fig, ax = plot_rcs_comparison(phi, sigma_num, sigma_mie)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "em3d"
    assert ax.lines[1].get_label() == "Mie"


def test_plot_rcs_comparison_polar_draws_two_curves():
    """plot_rcs_comparison_polar draws numerical and Mie curves on polar axes."""
    phi, sigma_mie = _synthetic_data()
    sigma_num = sigma_mie * 1.2
    fig, ax = plot_rcs_comparison_polar(phi, sigma_num, sigma_mie)
    assert isinstance(fig, Figure)
    assert ax.name == "polar"
    assert len(ax.lines) == 2


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

def test_plot_gamma0_spectrum_draws_spectrum_hull_and_circle():
    """plot_gamma0_spectrum draws spectrum, convex hull, origin, centre, and circle."""
    analysis = em3d.gamma0.analyze_spectrum(
        np.array([2.0 + 0.0j, 3.0 + 1.0j, 4.0 + 0.0j, 3.0 + 0.25j])
    )
    fig, ax = plot_gamma0_spectrum(analysis)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.collections) >= 2
    assert len(ax.lines) >= 1
    assert len(ax.patches) == 1
    assert ax.get_aspect() in ("equal", 1.0)


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


def test_plot_field_scalar_slice_returns_heatmap_without_quiver():
    """plot_field_scalar_slice draws scalar heatmap only."""
    import matplotlib.quiver as mquiver
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_scalar_slice(u, grid, plane="xy", part="abs")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    quivers = [c for c in ax.collections if isinstance(c, mquiver.Quiver)]
    assert len(quivers) == 0
    assert len(ax.collections) >= 1
    assert len(fig.axes) == 2, "Expected main axes plus colorbar axes"


def test_plot_field_vector_slice_returns_quiver_without_scalar_background():
    """plot_field_vector_slice draws quiver only, without scalar pcolormesh/colorbar."""
    import matplotlib.quiver as mquiver
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_vector_slice(u, grid, plane="xy", part="real", stride=2)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    quivers = [c for c in ax.collections if isinstance(c, mquiver.Quiver)]
    assert len(quivers) == 1
    assert quivers[0].N == 4 * 4
    assert len(fig.axes) == 1, "Vector-only slice should not create a colorbar"


def test_plot_field_scalar_slice_component():
    """component=0/1/2 selects a single scalar component."""
    u = _field_data()
    grid = _field_grid()
    for component in [None, 0, 1, 2]:
        fig, ax = plot_field_scalar_slice(u, grid, component=component)
        assert isinstance(fig, Figure)
    with pytest.raises(ValueError, match="component"):
        plot_field_scalar_slice(u, grid, component=3)


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
    """stride=2 decimates quiver to 4×4 arrows on an 8×8 grid."""
    import matplotlib.quiver as mquiver
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_slice(u, grid, stride=2)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    quivers = [c for c in ax.collections if isinstance(c, mquiver.Quiver)]
    assert len(quivers) == 1, "Expected exactly one quiver collection"
    assert quivers[0].N == 4 * 4, (
        f"Expected 16 arrows (4×4 decimated from 8×8), got {quivers[0].N}"
    )


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

    # wrong first dimension (shape[0] != 3)
    with pytest.raises(ValueError, match=r"\(3, Nx, Ny, Nz\)"):
        plot_field_slice(np.zeros((2, 8, 8, 8)), grid)

    # wrong number of dimensions (ndim != 4)
    with pytest.raises(ValueError, match=r"\(3, Nx, Ny, Nz\)"):
        plot_field_slice(np.zeros((3, 8, 8)), grid)


# ── plot_field_volume tests ───────────────────────────────────────────────

def test_plot_field_volume_returns_fig_ax3d():
    """plot_field_volume returns (Figure, Axes3D) with ax.name == '3d'."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_volume(u, grid, stride=2)
    assert isinstance(fig, Figure)
    assert ax.name == "3d"


def test_plot_field_scalar_volume_returns_scatter3d():
    """plot_field_scalar_volume draws scalar 3D scatter only."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_scalar_volume(u, grid, stride=2)
    assert isinstance(fig, Figure)
    assert ax.name == "3d"
    assert len(ax.collections) >= 1


def test_plot_field_vector_volume_returns_quiver3d():
    """plot_field_vector_volume draws 3D vector quiver."""
    u = _field_data()
    grid = _field_grid()
    fig, ax = plot_field_vector_volume(u, grid, stride=2)
    assert isinstance(fig, Figure)
    assert ax.name == "3d"
    assert len(ax.collections) >= 1


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


def test_plot_field_volume_invalid_inputs():
    """Bad arguments raise ValueError with informative messages."""
    u = _field_data()
    grid = _field_grid()

    with pytest.raises(ValueError, match="part"):
        plot_field_volume(u, grid, part="magnitude", stride=2)

    with pytest.raises(ValueError, match="stride"):
        plot_field_volume(u, grid, stride=0)

    with pytest.raises(ValueError, match=r"\(3, Nx, Ny, Nz\)"):
        plot_field_volume(np.zeros((2, 8, 8, 8)), grid)

    with pytest.raises(ValueError, match=r"\(3, Nx, Ny, Nz\)"):
        plot_field_volume(np.zeros((3, 8, 8)), grid)
