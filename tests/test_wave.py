import numpy as np
import pytest

from em3d.grid import Grid
from em3d.wave import flat_wave_vec


def test_flat_wave_shape_and_dtype(backend_numpy_double):
    grid = Grid(N=(4, 4, 4), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    wave = flat_wave_vec(grid, k=2.0, orient=(0.0, 0.0, 1.0), amplitude=(1.0, 0.0, 0.0))
    assert wave.shape == (3,) + grid.N
    assert wave.dtype == backend_numpy_double.complex_dtype


def test_flat_wave_plane_phase_along_z(backend_numpy_double):
    grid = Grid(N=(2, 2, 4), L=(1.0, 1.0, 4.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    k = 1.5
    wave = flat_wave_vec(grid, k=k, orient=(0.0, 0.0, 1.0), amplitude=(1.0, 0.0, 0.0))
    # x-component only; phase should be exp(i k z)
    z = grid.z
    expected_phase = np.exp(1j * k * np.asarray(z))
    np.testing.assert_allclose(wave[0, 0, 0, :], expected_phase, atol=1e-12)
    np.testing.assert_allclose(wave[1, 0, 0, :], np.zeros(4), atol=1e-12)


def test_flat_wave_requires_unit_orient(backend_numpy_double):
    grid = Grid(N=(2, 2, 2), L=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend=backend_numpy_double)
    with pytest.raises(ValueError):
        flat_wave_vec(grid, k=1.0, orient=(1.0, 1.0, 0.0), amplitude=(1.0, 0.0, 0.0))
