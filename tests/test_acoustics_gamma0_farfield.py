import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import em3d
from em3d.acoustics import AcousticOperator, eta_sphere, make_acoustic_problem
from em3d.acoustics.farfield import farfield_amplitude, pattern_plane, scattering_pattern
from em3d.acoustics.gamma0 import coarse_operator_matrix, estimate_from_problem, find_params_from_problem
from em3d.acoustics.visualization import plot_pattern, plot_scalar_slice, plot_scalar_slices


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _grid(N=(5, 5, 5)):
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


def _problem(N=(5, 5, 5), k0=1.0):
    grid = _grid(N)
    eta = eta_sphere(
        grid,
        center=(0.0, 0.0, 0.0),
        radius=0.35,
        eta_inside=1.4 + 0.15j,
        eta_outside=1.0,
    )
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


def test_acoustic_scalar_slice_plot_returns_fig_ax():
    problem = _problem(N=(4, 4, 4), k0=1.0)
    fig, ax = plot_scalar_slice(problem.wave, problem.grid, plane="xy", part="abs")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.collections) >= 1
    assert len(fig.axes) == 2


def test_acoustic_scalar_slices_uses_expected_paths(monkeypatch):
    problem = _problem(N=(4, 4, 4), k0=1.0)
    import em3d.acoustics.visualization as acoustic_vis

    calls = []

    def fake_plot_scalar_slice(*args, **kwargs):
        calls.append(kwargs["filename"])
        return object(), object()

    monkeypatch.setattr(acoustic_vis, "plot_scalar_slice", fake_plot_scalar_slice)
    paths = plot_scalar_slices(
        problem.wave,
        problem.grid,
        output_dir=".",
        prefix="u",
        parts=("abs", "angle"),
    )
    assert len(paths) == 6
    assert len(calls) == 6
    assert str(paths[0]) == "u_abs_xy.png"
    assert str(paths[-1]) == "u_angle_yz.png"


def test_acoustic_pattern_plot_returns_cartesian_and_polar_axes():
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    sigma = 1.0 + 0.25 * np.cos(phi)
    fig, ax = plot_pattern(phi, sigma)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.name != "polar"

    polar_fig, polar_ax = plot_pattern(phi, sigma, polar=True)
    assert isinstance(polar_fig, Figure)
    assert polar_ax.name == "polar"
