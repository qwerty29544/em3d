import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision
from em3d.spectral import (
    ArnoldiConfig,
    arnoldi,
    arnoldi_multistart,
    recover_operator_eigenvalue,
)


def test_arnoldi_matches_dense_spectral_radius():
    matrix = np.array(
        [[0.9, 0.4, 0.0], [0.0, 0.7, 0.2], [0.0, 0.0, -1.1]],
        dtype=np.complex128,
    )
    backend = Backend.numpy(Precision.DOUBLE)
    result = arnoldi(
        lambda vector: matrix @ vector,
        vector_size=3,
        backend=backend,
        config=ArnoldiConfig(krylov_dimension=3, milestones=(1, 2, 3)),
    )
    assert result.spectral_radius == pytest.approx(1.1, rel=1e-10)
    assert result.dominant_residual < 1e-10
    assert result.orthogonality_error < 1e-10


def test_multistart_and_eigenvalue_recovery():
    matrix = np.diag([0.5, -1.2]).astype(np.complex128)
    backend = Backend.numpy(Precision.DOUBLE)
    result = arnoldi_multistart(
        lambda vector: matrix @ vector,
        vector_size=2,
        backend=backend,
        config=ArnoldiConfig(krylov_dimension=2),
        seeds=(1, 2),
    )
    assert len(result.runs) == 2
    assert result.dominant_run.spectral_radius == pytest.approx(1.2)
    mu = 2.0 + 0.5j
    original = 3.0 - 0.25j
    iteration_value = 1.0 - original / mu
    assert recover_operator_eigenvalue(iteration_value, mu) == pytest.approx(original)
