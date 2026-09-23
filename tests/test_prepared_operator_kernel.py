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


def test_derived_adjoint_matches_explicit_and_dense():
    problem = _problem(N=(4, 3, 3), k0=1.7)
    explicit = PreparedEMKernel.build(
        problem.grid,
        k=problem.k0,
        include_adjoint=True,
        adjoint_storage="explicit",
        build_strategy="standard",
    )
    derived = PreparedEMKernel.build(
        problem.grid,
        k=problem.k0,
        include_adjoint=True,
        adjoint_storage="derived",
        build_strategy="streamed",
    )
    random = np.random.default_rng(42)
    field = (
        random.standard_normal((3,) + problem.grid.N)
        + 1j * random.standard_normal((3,) + problem.grid.N)
    )
    explicit_result = Operator(problem, prepared_kernel=explicit).rmatvec(field)
    derived_result = Operator(problem, prepared_kernel=derived).rmatvec(field)
    np.testing.assert_allclose(derived_result, explicit_result, rtol=1e-12, atol=1e-12)

    matrix = Operator(problem).to_dense_operator()
    dense = (matrix.conj().T @ field.transpose(1, 2, 3, 0).reshape(-1)).reshape(
        *problem.grid.N, 3
    ).transpose(3, 0, 1, 2)
    np.testing.assert_allclose(derived_result, dense, rtol=1e-12, atol=1e-12)


def test_forward_only_kernel_rejects_adjoint_action():
    problem = _problem()
    prepared = PreparedEMKernel.build(
        problem.grid,
        k=problem.k0,
        include_adjoint=False,
        build_strategy="streamed",
    )
    operator = Operator(problem, prepared_kernel=prepared)
    with pytest.raises(RuntimeError, match="adjoint action is unavailable"):
        operator.rmatvec(problem.wave)


def test_streamed_and_standard_forward_kernels_match():
    problem = _problem(N=(4, 3, 2), k0=2.1)
    standard = PreparedEMKernel.build(
        problem.grid,
        k=problem.k0,
        include_adjoint=False,
        build_strategy="standard",
    )
    streamed = PreparedEMKernel.build(
        problem.grid,
        k=problem.k0,
        include_adjoint=False,
        build_strategy="streamed",
    )
    np.testing.assert_allclose(
        streamed.kernel_hat,
        standard.kernel_hat,
        rtol=1e-13,
        atol=1e-13,
    )
    assert streamed.persistent_nbytes == streamed.kernel_hat.nbytes
