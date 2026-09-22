import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.geometry import (
    AxisAlignedBox,
    Ellipsoid,
    FullDomain,
    SamplingMode,
    center_mask,
    geometry_diagnostics,
    sample_contrast_tensor,
    volume_fractions,
)
from em3d.grid import Grid


def _grid(N=(8, 8, 8)):
    return Grid(
        N=N,
        L=(1.0, 1.0, 1.0),
        center=(0.0, 0.0, 0.0),
        backend=Backend.numpy(Precision.DOUBLE),
    )


def test_axis_aligned_box_volume_fractions_preserve_exact_volume():
    grid = _grid()
    box = AxisAlignedBox(center=(0.03, -0.02, 0.01), size=(0.23, 0.31, 0.17))
    fractions = volume_fractions(box, grid)
    assert fractions.shape == grid.N
    assert np.all((0.0 <= fractions) & (fractions <= 1.0))
    represented = float(fractions.sum() * grid.dv)
    assert represented == pytest.approx(np.prod(box.size), abs=1e-14)


def test_center_and_volume_fraction_diagnostics_are_distinct():
    grid = _grid((6, 6, 6))
    box = AxisAlignedBox(center=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
    center = geometry_diagnostics(box, grid, SamplingMode.CELL_CENTER)
    averaged = geometry_diagnostics(box, grid, SamplingMode.VOLUME_FRACTION)
    assert center.cells_total > 0
    assert averaged.cells_total >= center.cells_total
    assert averaged.relative_volume_error == pytest.approx(0.0, abs=1e-14)


def test_sampling_contrast_uses_relative_permittivity_minus_identity():
    grid = _grid((4, 4, 4))
    box = AxisAlignedBox(center=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5))
    contrast = sample_contrast_tensor(
        grid,
        background_eps_r=np.eye(3),
        feature_eps_r=np.diag([2.0, 1.5, 1.25]),
        geometry=box,
        mode=SamplingMode.CELL_CENTER,
    )
    mask = center_mask(box, grid)
    np.testing.assert_allclose(contrast[:, :, ~mask], 0.0)
    for index, expected in enumerate([1.0, 0.5, 0.25]):
        np.testing.assert_allclose(contrast[index, index, mask], expected)


def test_ellipsoid_volume_fraction_requires_explicit_quadrature():
    grid = _grid()
    with pytest.raises(NotImplementedError):
        volume_fractions(
            Ellipsoid(center=(0, 0, 0), radii=(0.2, 0.2, 0.2)),
            grid,
        )
    assert center_mask(FullDomain(), grid).all()
