from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from ..backend import Backend


@dataclass(frozen=True)
class ArnoldiConfig:
    krylov_dimension: int = 60
    milestones: tuple[int, ...] = (10, 20, 30, 40, 50, 60)
    seed: int = 20260909
    breakdown_tolerance: float = 1e-13
    reorthogonalization_passes: int = 2
    store_hessenberg: bool = True

    def __post_init__(self) -> None:
        if self.krylov_dimension <= 0:
            raise ValueError("krylov_dimension must be positive")
        if self.reorthogonalization_passes not in (1, 2):
            raise ValueError("reorthogonalization_passes must be 1 or 2")


@dataclass(frozen=True)
class ArnoldiResult:
    requested_dimension: int
    actual_dimension: int
    spectral_radius: float
    dominant_value: complex
    dominant_residual: float
    orthogonality_error: float
    elapsed_seconds: float
    radius_history: dict[int, float]
    residual_history: dict[int, float]
    hessenberg: np.ndarray | None
    seed: int


@dataclass(frozen=True)
class ArnoldiMultiStartResult:
    runs: tuple[ArnoldiResult, ...]
    dominant_run: ArnoldiResult



def _synchronize(backend: Backend) -> None:
    if backend.device == "cuda":
        backend.xp.cuda.Stream.null.synchronize()


def _ritz_diagnostic(hessenberg: np.ndarray, dimension: int) -> tuple[float, complex, float]:
    square = hessenberg[:dimension, :dimension]
    values, vectors = np.linalg.eig(square)
    index = int(np.argmax(np.abs(values)))
    dominant = complex(values[index])
    vector = vectors[:, index]
    if hessenberg.shape[0] > dimension:
        residual = float(abs(hessenberg[dimension, dimension - 1] * vector[-1]))
    else:
        residual = float("nan")
    return float(abs(dominant)), dominant, residual


def arnoldi(
    matvec: Callable[[Any], Any],
    *,
    vector_size: int,
    backend: Backend,
    config: ArnoldiConfig = ArnoldiConfig(),
) -> ArnoldiResult:
    """Matrix-free Arnoldi process with one or two CGS passes."""

    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    xp = backend.xp
    random = np.random.default_rng(config.seed)
    initial_host = (
        random.standard_normal(vector_size)
        + 1j * random.standard_normal(vector_size)
    ).astype(np.complex128)
    initial = backend.array(initial_host, dtype=backend.complex_dtype)
    initial /= xp.linalg.norm(initial)

    requested = min(int(config.krylov_dimension), int(vector_size))
    basis = xp.zeros((requested + 1, vector_size), dtype=backend.complex_dtype)
    hessenberg = xp.zeros((requested + 1, requested), dtype=backend.complex_dtype)
    basis[0] = initial

    _ = matvec(initial)
    _synchronize(backend)
    started = perf_counter()
    actual = requested

    for column in range(requested):
        work = matvec(basis[column])
        current = basis[: column + 1]
        coefficients = current.conj() @ work
        work = work - coefficients @ current
        if config.reorthogonalization_passes == 2:
            correction = current.conj() @ work
            work = work - correction @ current
            coefficients = coefficients + correction
        hessenberg[: column + 1, column] = coefficients
        beta = xp.linalg.norm(work)
        beta_host = float(backend.to_host(beta))
        hessenberg[column + 1, column] = beta
        if beta_host < config.breakdown_tolerance:
            actual = column + 1
            break
        basis[column + 1] = work / beta

    _synchronize(backend)
    elapsed = perf_counter() - started
    host_hessenberg = np.asarray(
        backend.to_host(hessenberg[: actual + 1, :actual]),
        dtype=np.complex128,
    )

    milestones = sorted(
        {
            int(value)
            for value in config.milestones
            if 1 <= int(value) <= actual
        }
        | {actual}
    )
    radius_history: dict[int, float] = {}
    residual_history: dict[int, float] = {}
    dominant = complex(np.nan, np.nan)
    dominant_residual = float("nan")
    for dimension in milestones:
        radius, value, residual = _ritz_diagnostic(host_hessenberg, dimension)
        radius_history[dimension] = radius
        residual_history[dimension] = residual
        if dimension == actual:
            dominant = value
            dominant_residual = residual

    used_basis = basis[:actual]
    gram = used_basis @ used_basis.conj().T
    identity = xp.eye(actual, dtype=backend.complex_dtype)
    orthogonality_error = float(
        backend.to_host(xp.linalg.norm(gram - identity) / max(actual, 1))
    )

    return ArnoldiResult(
        requested_dimension=requested,
        actual_dimension=actual,
        spectral_radius=float(radius_history[actual]),
        dominant_value=dominant,
        dominant_residual=float(dominant_residual),
        orthogonality_error=orthogonality_error,
        elapsed_seconds=float(elapsed),
        radius_history=radius_history,
        residual_history=residual_history,
        hessenberg=host_hessenberg if config.store_hessenberg else None,
        seed=int(config.seed),
    )


def arnoldi_multistart(
    matvec: Callable[[Any], Any],
    *,
    vector_size: int,
    backend: Backend,
    config: ArnoldiConfig = ArnoldiConfig(),
    seeds: Sequence[int],
) -> ArnoldiMultiStartResult:
    runs = tuple(
        arnoldi(
            matvec,
            vector_size=vector_size,
            backend=backend,
            config=replace(config, seed=int(seed)),
        )
        for seed in seeds
    )
    if not runs:
        raise ValueError("seeds must be non-empty")
    dominant = max(runs, key=lambda result: result.spectral_radius)
    return ArnoldiMultiStartResult(runs=runs, dominant_run=dominant)


def recover_operator_eigenvalue(iteration_value: complex, mu: complex) -> complex:
    return complex(mu) * (1.0 - complex(iteration_value))
