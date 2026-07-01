import numpy as np
import pytest

import em3d
from em3d.acoustics import (
    AcousticProblem,
    eta_ellipsoid,
    eta_homogeneous,
    eta_slab,
    eta_sphere,
    make_acoustic_problem,
    plane_wave_scalar,
)


def _grid(N=(4, 4, 4)):
    be = em3d.Backend.numpy(em3d.Precision.DOUBLE)
    return em3d.Grid(N=N, L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=be)


def test_acoustic_problem_eta_convention_and_wave_shape():
    grid = _grid()
    eta = eta_homogeneous(grid, 1.0 + 0.0j)
    problem = make_acoustic_problem(
        grid,
        eta,
        k0=2.0,
        direction=(0.0, 0.0, 2.0),
        amplitude=2.0 + 0.5j,
    )
    assert isinstance(problem, AcousticProblem)
    assert problem.eta.shape == grid.N
    assert problem.wave.shape == grid.N
    assert problem.eta.dtype == grid.backend.complex_dtype
    assert problem.wave.dtype == grid.backend.complex_dtype
    assert np.allclose(problem.chi, 0.0)
    assert np.allclose(np.abs(problem.wave), abs(2.0 + 0.5j))


def test_plane_wave_scalar_uses_normalized_direction():
    grid = _grid()
    w1 = plane_wave_scalar(grid, k=3.0, direction=(0.0, 0.0, 1.0))
    w2 = plane_wave_scalar(grid, k=3.0, direction=(0.0, 0.0, 5.0))
    assert np.allclose(w1, w2)


def test_acoustic_problem_rejects_bad_eta_shape_and_zero_direction():
    grid = _grid()
    wave = plane_wave_scalar(grid, k=1.0, direction=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="eta.shape"):
        AcousticProblem(
            grid=grid,
            eta=np.ones((2, 2), dtype=np.complex128),
            wave=wave,
            k0=1.0,
            volume=1.0,
        )
    with pytest.raises(ValueError, match="direction"):
        plane_wave_scalar(grid, k=1.0, direction=(0.0, 0.0, 0.0))


def test_material_generators_return_eta_not_chi():
    grid = _grid()
    homogeneous = eta_homogeneous(grid, 2.5 + 0.25j)
    assert np.allclose(homogeneous, 2.5 + 0.25j)

    slab = eta_slab(
        grid,
        eta_inside=3.0 + 1.0j,
        eta_outside=1.0,
        axis=0,
        width_fraction=0.5,
    )
    assert set(np.unique(np.asarray(slab)).tolist()) == {1.0 + 0.0j, 3.0 + 1.0j}

    sphere = eta_sphere(
        grid,
        center=(0.0, 0.0, 0.0),
        radius=0.4,
        eta_inside=4.0,
        eta_outside=1.0,
    )
    assert np.max(np.asarray(sphere).real) == 4.0
    assert np.min(np.asarray(sphere).real) == 1.0

    ellipsoid = eta_ellipsoid(
        grid,
        center=(0.0, 0.0, 0.0),
        radii=(0.5, 0.25, 0.25),
        eta_inside=2.0,
        eta_outside=1.0,
    )
    assert ellipsoid.shape == grid.N
    assert np.count_nonzero(np.asarray(ellipsoid) == 2.0 + 0.0j) > 0
