from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from time import perf_counter
from typing import Mapping

import numpy as np

from ..backend import Backend
from ..dense import dense_operator_matrix
from ..geometry import (
    AxisAlignedBox,
    Ellipsoid,
    FullDomain,
    Geometry,
    GeometryDiagnostics,
    SamplingMode,
    geometry_diagnostics,
    sample_contrast_tensor,
)
from ..grid import Grid
from ..layout import flatten_field
from ..operator import Operator, PreparedEMKernel
from ..problem import Problem
from ..solvers import SIM, SolverConfig, SolverResult
from ..spectral import (
    ArnoldiConfig,
    ArnoldiResult,
    CircleLocalization,
    ConvergenceDiagnostics,
    SpectrumLocalization,
    TransferAssessment,
    analyze_residual_history,
    analyze_spectrum,
    arnoldi,
    assess_transfer,
)
from ..wave import flat_wave_vec


@dataclass(frozen=True)
class SpectralCaseDefinition:
    key: str
    title: str
    domain_lengths: tuple[float, float, float]
    domain_center: tuple[float, float, float]
    k0: float
    background_eps_r: object
    feature_eps_r: object
    feature_geometry: Geometry
    wave_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    wave_amplitude: tuple[complex, complex, complex] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class BuiltSpectralCase:
    definition: SpectralCaseDefinition
    problem: Problem
    geometry: GeometryDiagnostics
    sampling_mode: SamplingMode


@dataclass(frozen=True)
class GridSpectrumRun:
    case_key: str
    grid_shape: tuple[int, int, int]
    matrix_dimension: int
    localization: SpectrumLocalization
    geometry: GeometryDiagnostics
    matrix_build_seconds: float
    eigenvalue_seconds: tuple[float, ...]

    @property
    def median_eigenvalue_seconds(self) -> float:
        return float(np.median(self.eigenvalue_seconds))


@dataclass(frozen=True)
class FineTransferRun:
    case_key: str
    parameter_label: str
    circle: CircleLocalization
    solver_result: SolverResult
    convergence: ConvergenceDiagnostics
    elapsed_seconds: float


@dataclass(frozen=True)
class ControlTransferRun:
    coarse: GridSpectrumRun
    target: GridSpectrumRun
    assessment: TransferAssessment | None


def _normalize_grid_shape(
    grid_shape: int | tuple[int, int, int],
) -> tuple[int, int, int]:
    if isinstance(grid_shape, int):
        shape = (grid_shape, grid_shape, grid_shape)
    else:
        shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"grid_shape must be a positive int or 3-tuple, got {grid_shape!r}")
    return shape


def build_spectral_case(
    definition: SpectralCaseDefinition,
    *,
    grid_shape: int | tuple[int, int, int],
    backend: Backend,
    sampling_mode: SamplingMode = SamplingMode.CELL_CENTER,
) -> BuiltSpectralCase:
    shape = _normalize_grid_shape(grid_shape)
    grid = Grid(
        N=shape,
        L=definition.domain_lengths,
        center=definition.domain_center,
        backend=backend,
    )
    contrast = sample_contrast_tensor(
        grid,
        background_eps_r=definition.background_eps_r,
        feature_eps_r=definition.feature_eps_r,
        geometry=definition.feature_geometry,
        mode=sampling_mode,
    )
    wave = flat_wave_vec(
        grid,
        k=definition.k0,
        orient=definition.wave_direction,
        amplitude=definition.wave_amplitude,
    )
    problem = Problem(
        grid=grid,
        eps_tensor=contrast,
        wave=wave,
        k0=float(definition.k0),
        volume=float(np.prod(definition.domain_lengths)),
    )
    diagnostics = geometry_diagnostics(
        definition.feature_geometry,
        grid,
        sampling_mode,
    )
    return BuiltSpectralCase(
        definition=definition,
        problem=problem,
        geometry=diagnostics,
        sampling_mode=SamplingMode(sampling_mode),
    )


def check_operator_consistency(
    built_case: BuiltSpectralCase,
    *,
    seed: int = 20260908,
    prepared_kernel: PreparedEMKernel | None = None,
) -> float:
    """Relative difference between the dense and FFT actions of the full operator."""

    problem = built_case.problem
    if problem.backend.device != "cpu":
        raise ValueError(
            "dense consistency checks require a NumPy-backed BuiltSpectralCase"
        )
    operator = Operator(problem, prepared_kernel=prepared_kernel)
    matrix = dense_operator_matrix(problem)
    random = np.random.default_rng(seed)
    field = (
        random.standard_normal((3,) + problem.grid.N)
        + 1j * random.standard_normal((3,) + problem.grid.N)
    ).astype(problem.backend.complex_dtype)
    dense_result = matrix @ flatten_field(field)
    fft_result = flatten_field(operator.matvec(field))
    denominator = np.linalg.norm(dense_result)
    if denominator == 0.0:
        return float(np.linalg.norm(fft_result - dense_result))
    return float(np.linalg.norm(fft_result - dense_result) / denominator)


def compute_grid_spectrum(
    built_case: BuiltSpectralCase,
    *,
    eigenvalue_repeats: int = 1,
) -> GridSpectrumRun:
    if built_case.problem.backend.device != "cpu":
        raise ValueError("full spectrum computation requires a NumPy backend")
    if eigenvalue_repeats <= 0:
        raise ValueError("eigenvalue_repeats must be positive")

    started = perf_counter()
    matrix = dense_operator_matrix(built_case.problem)
    build_seconds = perf_counter() - started

    samples: list[float] = []
    spectrum = None
    for _ in range(int(eigenvalue_repeats)):
        started = perf_counter()
        current = np.linalg.eigvals(matrix)
        samples.append(perf_counter() - started)
        if spectrum is None:
            spectrum = current
    assert spectrum is not None
    localization = analyze_spectrum(spectrum)
    return GridSpectrumRun(
        case_key=built_case.definition.key,
        grid_shape=tuple(built_case.problem.grid.N),
        matrix_dimension=int(matrix.shape[0]),
        localization=localization,
        geometry=built_case.geometry,
        matrix_build_seconds=float(build_seconds),
        eigenvalue_seconds=tuple(float(value) for value in samples),
    )


def assess_control_transfer(
    coarse: GridSpectrumRun,
    target: GridSpectrumRun,
) -> ControlTransferRun:
    if coarse.case_key != target.case_key:
        raise ValueError(
            f"case mismatch: {coarse.case_key!r} != {target.case_key!r}"
        )
    assessment = None
    if coarse.localization.circle is not None:
        assessment = assess_transfer(coarse.localization, target.localization)
    return ControlTransferRun(
        coarse=coarse,
        target=target,
        assessment=assessment,
    )


def run_parameter_transfer(
    built_case: BuiltSpectralCase,
    parameters: Mapping[str, CircleLocalization],
    *,
    solver_config: SolverConfig,
    prepared_kernel: PreparedEMKernel | None = None,
) -> tuple[FineTransferRun, ...]:
    operator = Operator(built_case.problem, prepared_kernel=prepared_kernel)
    results: list[FineTransferRun] = []
    for label, circle in parameters.items():
        configuration = replace(
            solver_config,
            mu=circle.mu,
            radius=circle.radius,
        )
        started = perf_counter()
        solver_result = SIM(configuration).solve(
            operator,
            built_case.problem.wave,
        )
        elapsed = perf_counter() - started
        diagnostics = analyze_residual_history(
            solver_result.residual_history,
            rtol=configuration.rtol,
        )
        results.append(
            FineTransferRun(
                case_key=built_case.definition.key,
                parameter_label=str(label),
                circle=circle,
                solver_result=solver_result,
                convergence=diagnostics,
                elapsed_seconds=float(elapsed),
            )
        )
    return tuple(results)


def diagnose_iteration_spectrum(
    built_case: BuiltSpectralCase,
    circle: CircleLocalization,
    *,
    config: ArnoldiConfig = ArnoldiConfig(),
    prepared_kernel: PreparedEMKernel | None = None,
) -> ArnoldiResult:
    operator = Operator(built_case.problem, prepared_kernel=prepared_kernel)
    return arnoldi(
        lambda vector: operator.iteration_matvec_flat(vector, mu=circle.mu),
        vector_size=3 * prod(built_case.problem.grid.N),
        backend=built_case.problem.backend,
        config=config,
    )


def default_article_cases() -> tuple[SpectralCaseDefinition, ...]:
    identity = np.eye(3, dtype=np.complex128)
    anisotropic = np.diag([2.0, 1.6, 1.3]).astype(np.complex128)
    base = np.diag([1.6, 1.4, 1.2]).astype(np.complex128)
    inclusion = np.diag(
        [3.0 + 0.03j, 2.4 + 0.02j, 1.8 + 0.01j]
    ).astype(np.complex128)
    common = {
        "domain_lengths": (1.0, 1.0, 1.0),
        "domain_center": (0.0, 0.0, 0.0),
        "k0": 4.0,
    }
    return (
        SpectralCaseDefinition(
            key="homogeneous_anisotropic",
            title="Однородная анизотропная область",
            background_eps_r=identity,
            feature_eps_r=anisotropic,
            feature_geometry=FullDomain(),
            **common,
        ),
        SpectralCaseDefinition(
            key="anisotropic_ellipsoid",
            title="Анизотропный эллипсоид",
            background_eps_r=identity,
            feature_eps_r=anisotropic,
            feature_geometry=Ellipsoid(
                center=(0.0, 0.0, 0.0),
                radii=(0.30, 0.25, 0.20),
            ),
            **common,
        ),
        SpectralCaseDefinition(
            key="local_inclusion",
            title="Область с локальной неоднородностью",
            background_eps_r=base,
            feature_eps_r=inclusion,
            feature_geometry=AxisAlignedBox(
                center=(0.0, 0.0, 0.0),
                size=(0.20, 0.20, 0.20),
            ),
            **common,
        ),
    )
