import numpy as np

from em3d.backend import Backend
from em3d.dense import (
    dense_kernel_matrix,
    dense_kernel_matrix_reference,
    dense_operator_matrix,
)
from em3d.experiments.spectral_transfer import (
    build_spectral_case,
    local_inclusion_case,
)


def test_vectorized_dense_kernel_matches_scalar_reference():
    backend = Backend.numpy()
    built = build_spectral_case(
        local_inclusion_case(0.5, k0=1.25),
        grid_shape=2,
        backend=backend,
    )
    actual = dense_kernel_matrix(built.problem.grid, k=built.problem.k0)
    expected = dense_kernel_matrix_reference(
        built.problem.grid,
        k=built.problem.k0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_vectorized_full_operator_has_expected_action():
    backend = Backend.numpy()
    built = build_spectral_case(
        local_inclusion_case(0.5, k0=1.25),
        grid_shape=2,
        backend=backend,
    )
    matrix = dense_operator_matrix(built.problem)
    assert matrix.shape == (24, 24)
    assert np.isfinite(matrix).all()
