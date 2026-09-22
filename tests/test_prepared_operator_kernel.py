import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.grid import Grid
from em3d.operator import Operator, PreparedEMKernel
from em3d.problem import Problem


def _problem(N=(3, 3, 3), k0=1.0):
    backend = Backend.numpy(Precision.DOUBLE)
    grid = Grid(N=N, L=(1, 1, 1), center=(0, 0, 0), backend=backend)
    contrast = np.zeros((3, 3) + N, dtype=np.complex128)
    contrast[0, 0] = 0.1
    wave = np.ones((3,) + N, dtype=np.complex128)
    return Problem(grid=grid, eps_tensor=contrast, wave=wave, k0=k0, volume=1.0)


def test_prepared_kernel_reuse_preserves_operator_action():
    problem = _problem()
    prepared = PreparedEMKernel.build(problem.grid, k=problem.k0)
    direct = Operator(problem)
    reused = Operator(problem, prepared_kernel=prepared)
    np.testing.assert_allclose(
        direct.matvec(problem.wave),
        reused.matvec(problem.wave),
        rtol=1e-13,
        atol=1e-13,
    )


def test_prepared_kernel_rejects_incompatible_wave_number():
    problem = _problem(k0=1.0)
    prepared = PreparedEMKernel.build(problem.grid, k=2.0)
    with pytest.raises(ValueError, match="wave number"):
        Operator(problem, prepared_kernel=prepared)
