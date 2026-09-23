from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import prod
from time import perf_counter
from typing import Mapping

import numpy as np

from ..backend import Backend
from ..dense import dense_operator_matrix, dense_operator_matrix_backend
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
    ArnoldiMultiStartResult,
    ArnoldiResult,
    CircleLocalization,
    ConvergenceClass,
    ConvergenceDiagnostics,
    EnsembleLocalization,
    SpectrumLocalization,
    TransferAssessment,
    analyze_residual_history,
    analyze_spectrum,
    arnoldi,
    arnoldi_multistart,
    assess_transfer,
    build_ensemble_localization,
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
    backend_device: str = "cpu"

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


@dataclass(frozen=True)
class SamplingMetrics:
    wavelength_free: float
    material_index_scale: float
    wavelength_material: float
    ppw_free: float
    ppw_material: float
    kh_max: float
    nyquist_free_ok: bool
    nyquist_material_ok: bool
    feature_cells_min_axis: int
    feature_cells_total: int
    cells_across_feature_continuous: float
    geometry_two_cells_ok: bool
    required_cells_across_feature_free: float
    required_cells_across_feature_material: float


@dataclass(frozen=True)
class EnsembleParameterRun:
    case_key: str
    levels: tuple[int, ...]
    ensemble: EnsembleLocalization
    control_assessment: TransferAssessment | None
    delta_latest: float
    delta_max: float
    q_inflated_latest: float
    q_inflated_max: float
    inflated_latest_safe: bool
    inflated_max_safe: bool
    resolution_by_level: dict[int, int]
    maximum_resolution: int
    adequate_geometry_levels: tuple[int, ...]

    @property
    def label(self) -> str:
        return "+".join(str(level) for level in self.levels)

    @property
    def circle(self) -> CircleLocalization | None:
        return self.ensemble.localization.circle

    @property
    def geometry_two_cell_rule(self) -> bool:
        return self.maximum_resolution >= 2


class ParameterPhase(str, Enum):
    NO_PARAMETER = "no_parameter"
    CONVERGED = "converged"
    CONTRACTING_UNRESOLVED = "contracting_unresolved"
    UNSTABLE = "unstable"
    UNRESOLVED = "unresolved"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class ParameterPhaseAssessment:
    phase: ParameterPhase
    circle_exists: bool
    reached_rtol: bool
    asymptotic_ratio: float



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


def _synchronize_backend(backend: Backend) -> None:
    if backend.device == "cuda":
        backend.xp.cuda.Stream.null.synchronize()


def _free_backend_pool(backend: Backend) -> None:
    if backend.device == "cuda":
        backend.xp.get_default_memory_pool().free_all_blocks()


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
    """Build the full matrix and evaluate all eigenvalues.

    NumPy and CuPy backends are both supported. Only the final spectrum and
    localization are moved to host memory. Backend memory is released in a
    ``finally`` block so that a failed high-resolution eigensolve does not
    poison subsequent parameter points.
    """

    if eigenvalue_repeats <= 0:
        raise ValueError("eigenvalue_repeats must be positive")

    backend = built_case.problem.backend
    matrix = None
    try:
        _synchronize_backend(backend)
        started = perf_counter()
        matrix = dense_operator_matrix_backend(built_case.problem)
        _synchronize_backend(backend)
        build_seconds = perf_counter() - started

        samples: list[float] = []
        spectrum_host = None
        for _ in range(int(eigenvalue_repeats)):
            _synchronize_backend(backend)
            started = perf_counter()
            current = backend.xp.linalg.eigvals(matrix)
            _synchronize_backend(backend)
            samples.append(perf_counter() - started)
            if spectrum_host is None:
                spectrum_host = np.asarray(
                    backend.to_host(current), dtype=np.complex128
                )
            del current
        assert spectrum_host is not None
        localization = analyze_spectrum(spectrum_host)
        dimension = int(matrix.shape[0])
        return GridSpectrumRun(
            case_key=built_case.definition.key,
            grid_shape=tuple(built_case.problem.grid.N),
            matrix_dimension=dimension,
            localization=localization,
            geometry=built_case.geometry,
            matrix_build_seconds=float(build_seconds),
            eigenvalue_seconds=tuple(float(value) for value in samples),
            backend_device=str(backend.device),
        )
    finally:
        if matrix is not None:
            del matrix
        _free_backend_pool(backend)


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
    retain_solution: bool = True,
) -> tuple[FineTransferRun, ...]:
    operator = Operator(built_case.problem, prepared_kernel=prepared_kernel)
    results: list[FineTransferRun] = []
    for label, circle in parameters.items():
        configuration = replace(
            solver_config,
            mu=circle.mu,
            radius=circle.radius,
        )
        _synchronize_backend(built_case.problem.backend)
        started = perf_counter()
        solver_result = SIM(configuration).solve(
            operator,
            built_case.problem.wave,
        )
        _synchronize_backend(built_case.problem.backend)
        elapsed = perf_counter() - started
        diagnostics = analyze_residual_history(
            solver_result.residual_history,
            rtol=configuration.rtol,
        )
        if not retain_solution:
            solver_result = replace(solver_result, u=None)
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


def diagnose_iteration_spectrum_multistart(
    built_case: BuiltSpectralCase,
    circle: CircleLocalization,
    *,
    config: ArnoldiConfig = ArnoldiConfig(),
    seeds: tuple[int, ...],
    prepared_kernel: PreparedEMKernel | None = None,
) -> ArnoldiMultiStartResult:
    """Estimate the iteration spectral radius from several start vectors.

    Boundary classifications are sensitive to a missed dominant invariant
    subspace.  The multi-start variant keeps every run and selects the largest
    Ritz radius, rather than silently relying on one random vector.
    """

    operator = Operator(built_case.problem, prepared_kernel=prepared_kernel)
    return arnoldi_multistart(
        lambda vector: operator.iteration_matvec_flat(vector, mu=circle.mu),
        vector_size=3 * prod(built_case.problem.grid.N),
        backend=built_case.problem.backend,
        config=config,
        seeds=seeds,
    )


def local_inclusion_case(
    side: float,
    *,
    k0: float = 4.0,
    key_prefix: str = "local_inclusion",
    title_prefix: str = "Область с локальной неоднородностью",
) -> SpectralCaseDefinition:
    """Canonical local-inclusion case used by E3--E6."""

    identity = np.eye(3, dtype=np.complex128)
    base = np.diag([1.6, 1.4, 1.2]).astype(np.complex128)
    inclusion = np.diag(
        [3.0 + 0.03j, 2.4 + 0.02j, 1.8 + 0.01j]
    ).astype(np.complex128)
    token = f"{float(side):.6g}".replace(".", "p")
    wave_token = f"{float(k0):.6g}".replace(".", "p")
    return SpectralCaseDefinition(
        key=f"{key_prefix}_a{token}_k{wave_token}",
        title=f"{title_prefix}, a={float(side):g}, k0={float(k0):g}",
        domain_lengths=(1.0, 1.0, 1.0),
        domain_center=(0.0, 0.0, 0.0),
        k0=float(k0),
        background_eps_r=base,
        feature_eps_r=inclusion,
        feature_geometry=AxisAlignedBox(
            center=(0.0, 0.0, 0.0),
            size=(float(side), float(side), float(side)),
        ),
        wave_direction=(0.0, 0.0, 1.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )


def material_index_scale(definition: SpectralCaseDefinition) -> float:
    """Conservative refractive-index scale over both materials."""

    def eigenvalues(material) -> np.ndarray:
        array = np.asarray(material, dtype=np.complex128)
        if array.ndim == 0:
            return np.full(3, complex(array), dtype=np.complex128)
        if array.shape != (3, 3):
            raise ValueError(
                "material tensors must be scalars or (3, 3) arrays, "
                f"got {array.shape}"
            )
        return np.linalg.eigvals(array)

    values = np.concatenate(
        [
            eigenvalues(definition.background_eps_r),
            eigenvalues(definition.feature_eps_r),
        ]
    )
    return float(np.sqrt(np.max(np.abs(values))))


def sampling_metrics(
    definition: SpectralCaseDefinition,
    *,
    grid_shape: int | tuple[int, int, int],
    sampling_mode: SamplingMode = SamplingMode.CELL_CENTER,
) -> SamplingMetrics:
    shape = _normalize_grid_shape(grid_shape)
    steps = tuple(
        float(length) / int(count)
        for length, count in zip(definition.domain_lengths, shape, strict=True)
    )
    maximum_step = max(steps)
    wavelength_free = 2.0 * np.pi / float(definition.k0)
    index_scale = material_index_scale(definition)
    wavelength_material = wavelength_free / index_scale

    grid = Grid(
        N=shape,
        L=definition.domain_lengths,
        center=definition.domain_center,
        backend=Backend.numpy(),
    )
    geometry = geometry_diagnostics(
        definition.feature_geometry,
        grid,
        sampling_mode,
    )
    feature_cells_min_axis = int(min(geometry.cells_per_axis))
    if isinstance(definition.feature_geometry, AxisAlignedBox):
        feature_extent = min(definition.feature_geometry.size)
    elif isinstance(definition.feature_geometry, Ellipsoid):
        feature_extent = 2.0 * min(definition.feature_geometry.radii)
    else:
        feature_extent = min(definition.domain_lengths)

    return SamplingMetrics(
        wavelength_free=float(wavelength_free),
        material_index_scale=float(index_scale),
        wavelength_material=float(wavelength_material),
        ppw_free=float(wavelength_free / maximum_step),
        ppw_material=float(wavelength_material / maximum_step),
        kh_max=float(definition.k0 * maximum_step),
        nyquist_free_ok=bool(wavelength_free / maximum_step >= 2.0 - 1e-12),
        nyquist_material_ok=bool(
            wavelength_material / maximum_step >= 2.0 - 1e-12
        ),
        feature_cells_min_axis=feature_cells_min_axis,
        feature_cells_total=int(geometry.cells_total),
        cells_across_feature_continuous=float(feature_extent / maximum_step),
        geometry_two_cells_ok=bool(feature_cells_min_axis >= 2),
        required_cells_across_feature_free=float(
            2.0 * feature_extent / wavelength_free
        ),
        required_cells_across_feature_material=float(
            2.0 * feature_extent / wavelength_material
        ),
    )


def build_ensemble_parameter(
    spectra: Mapping[int, GridSpectrumRun],
    levels: tuple[int, ...],
    *,
    control: GridSpectrumRun | None = None,
) -> EnsembleParameterRun:
    selected = tuple(int(level) for level in levels)
    if not selected:
        raise ValueError("ensemble must contain at least one level")
    if any(level not in spectra for level in selected):
        missing = [level for level in selected if level not in spectra]
        raise KeyError(f"missing spectral levels: {missing}")

    ensemble = build_ensemble_localization(
        {level: spectra[level].localization for level in selected},
        levels=selected,
    )
    circle = ensemble.localization.circle
    control_assessment = None
    if control is not None and circle is not None:
        control_assessment = assess_transfer(
            ensemble.localization, control.localization
        )

    adjacent_hausdorff = ensemble.adjacent_hausdorff_distances
    delta_latest = float(adjacent_hausdorff[-1]) if adjacent_hausdorff else 0.0
    delta_max = float(max(adjacent_hausdorff)) if adjacent_hausdorff else 0.0
    if circle is None:
        q_latest = float("inf")
        q_max = float("inf")
        latest_safe = False
        maximum_safe = False
    else:
        q_latest = float((circle.radius + delta_latest) / abs(circle.mu))
        q_max = float((circle.radius + delta_max) / abs(circle.mu))
        latest_safe = bool(circle.radius + delta_latest < abs(circle.mu))
        maximum_safe = bool(circle.radius + delta_max < abs(circle.mu))

    resolution_by_level = {
        level: int(min(spectra[level].geometry.cells_per_axis))
        for level in selected
    }
    maximum_resolution = max(resolution_by_level.values())
    adequate = tuple(
        level
        for level, resolution in resolution_by_level.items()
        if resolution >= 2
    )
    case_keys = {spectra[level].case_key for level in selected}
    if len(case_keys) != 1:
        raise ValueError(f"ensemble contains multiple cases: {sorted(case_keys)}")

    return EnsembleParameterRun(
        case_key=next(iter(case_keys)),
        levels=selected,
        ensemble=ensemble,
        control_assessment=control_assessment,
        delta_latest=delta_latest,
        delta_max=delta_max,
        q_inflated_latest=q_latest,
        q_inflated_max=q_max,
        inflated_latest_safe=latest_safe,
        inflated_max_safe=maximum_safe,
        resolution_by_level=resolution_by_level,
        maximum_resolution=int(maximum_resolution),
        adequate_geometry_levels=adequate,
    )


def assess_parameter_phase(
    localization: SpectrumLocalization,
    fine_run: FineTransferRun | None,
) -> ParameterPhaseAssessment:
    if localization.circle is None:
        return ParameterPhaseAssessment(
            phase=ParameterPhase.NO_PARAMETER,
            circle_exists=False,
            reached_rtol=False,
            asymptotic_ratio=float("nan"),
        )
    if fine_run is None:
        return ParameterPhaseAssessment(
            phase=ParameterPhase.UNRESOLVED,
            circle_exists=True,
            reached_rtol=False,
            asymptotic_ratio=float("nan"),
        )

    mapping = {
        ConvergenceClass.CONVERGED: ParameterPhase.CONVERGED,
        ConvergenceClass.CONTRACTING_UNRESOLVED: ParameterPhase.CONTRACTING_UNRESOLVED,
        ConvergenceClass.UNSTABLE: ParameterPhase.UNSTABLE,
        ConvergenceClass.UNRESOLVED: ParameterPhase.UNRESOLVED,
        ConvergenceClass.NONFINITE: ParameterPhase.NONFINITE,
    }
    return ParameterPhaseAssessment(
        phase=mapping[fine_run.convergence.classification],
        circle_exists=True,
        reached_rtol=bool(fine_run.solver_result.converged),
        asymptotic_ratio=float(fine_run.convergence.asymptotic_ratio),
    )


def default_article_cases() -> tuple[SpectralCaseDefinition, ...]:
    identity = np.eye(3, dtype=np.complex128)
    anisotropic = np.diag([2.0, 1.6, 1.3]).astype(np.complex128)
    common = {
        "domain_lengths": (1.0, 1.0, 1.0),
        "domain_center": (0.0, 0.0, 0.0),
        "k0": 4.0,
    }
    local = local_inclusion_case(0.20, k0=4.0)
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
        replace(
            local,
            key="local_inclusion",
            title="Область с локальной неоднородностью",
        ),
    )
