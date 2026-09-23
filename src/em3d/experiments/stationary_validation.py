from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from ..backend import Backend
from ..farfield import rcs_plane
from ..geometry import Ellipsoid, SamplingMode, center_mask
from ..mie import mie_field, mie_rcs_plane
from ..operator import Operator, PreparedEMKernel
from ..solvers import BiCGStab, SIM, SolverConfig, SolverResult, TwoStep
from ..spectral import CircleLocalization
from .spectral_transfer import (
    BuiltSpectralCase,
    SpectralCaseDefinition,
    build_spectral_case,
    compute_grid_spectrum,
)


@dataclass(frozen=True)
class SolverExecution:
    case_key: str
    solver_name: str
    result: SolverResult
    elapsed_seconds: float
    true_relative_residual: float
    qualified: bool
    solution_host: np.ndarray | None


@dataclass(frozen=True)
class FieldComparisonMetrics:
    candidate_solver: str
    reference_solver: str
    relative_l2: float
    relative_linf: float
    component_relative_l2: tuple[float, float, float]


@dataclass(frozen=True)
class RCSCurve:
    solver_name: str
    plane: str
    phi: np.ndarray
    sigma: np.ndarray
    sigma_normalized: np.ndarray
    peak: float
    angular_integral: float


@dataclass(frozen=True)
class RCSComparisonMetrics:
    candidate_solver: str
    reference_solver: str
    plane: str
    normalized_l2: float
    absolute_l2: float
    absolute_linf: float
    peak_ratio: float
    integral_ratio: float
    peak_angle_error_degrees: float


@dataclass(frozen=True)
class MieReferenceComparison:
    radius_label: str
    radius: float
    field_relative_l2: float
    field_inside_relative_l2: float
    field_outside_relative_l2: float
    scattered_field_relative_l2: float
    scattered_field_outside_relative_l2: float
    rcs_normalized_l2: float
    rcs_absolute_l2: float
    rcs_absolute_linf: float
    rcs_peak_ratio: float
    rcs_integral_ratio: float
    rcs_peak_angle_error_degrees: float


@dataclass(frozen=True)
class MieValidationResult:
    case_key: str
    solver_name: str
    nominal_radius: float
    effective_radius: float
    represented_volume: float
    exact_volume: float
    relative_volume_error: float
    nominal: MieReferenceComparison
    effective: MieReferenceComparison
    farfield_backend_relative_l2: float | None


def _synchronize(backend: Backend) -> None:
    if backend.device == "cuda":
        backend.xp.cuda.Stream.null.synchronize()


def _relative_norm(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(candidate - reference))
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _relative_linf(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.max(np.abs(candidate - reference))) if candidate.size else 0.0
    denominator = float(np.max(np.abs(reference))) if reference.size else 0.0
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _solver_instance(
    name: str,
    *,
    base_config: SolverConfig,
    circle: CircleLocalization | None,
):
    normalized = str(name).strip().lower()
    if normalized == "sim":
        if circle is None:
            raise ValueError("SIM requires a spectral circle localization")
        return SIM(
            replace(
                base_config,
                mu=circle.mu,
                radius=circle.radius,
            )
        )
    if normalized == "bicgstab":
        return BiCGStab(replace(base_config, mu=None, radius=None))
    if normalized in {"twostep", "two_step", "two-step"}:
        return TwoStep(replace(base_config, mu=None, radius=None))
    raise ValueError(f"unknown solver {name!r}; expected SIM, BiCGStab, or TwoStep")


def estimate_parameter_circle(
    definition: SpectralCaseDefinition,
    *,
    coarse_grid_shape: int | tuple[int, int, int],
    eigenvalue_repeats: int = 1,
) -> CircleLocalization:
    """Estimate the stationary-iteration parameter on an independently sampled grid."""

    built = build_spectral_case(
        definition,
        grid_shape=coarse_grid_shape,
        backend=Backend.numpy(),
        sampling_mode=SamplingMode.CELL_CENTER,
    )
    spectrum = compute_grid_spectrum(
        built,
        eigenvalue_repeats=eigenvalue_repeats,
    )
    circle = spectrum.localization.circle
    if circle is None:
        raise ValueError(
            f"case {definition.key!r} has no admissible spectral circle on "
            f"grid {spectrum.grid_shape}"
        )
    return circle


def true_relative_residual(operator: Operator, solution, rhs) -> float:
    backend = operator.backend
    xp = backend.xp
    rhs_norm = float(backend.to_host(xp.linalg.norm(rhs)))
    residual = operator.matvec(solution) - rhs
    residual_norm = float(backend.to_host(xp.linalg.norm(residual)))
    if rhs_norm == 0.0:
        return residual_norm
    return residual_norm / rhs_norm


def run_solver_comparison(
    built_case: BuiltSpectralCase,
    *,
    solver_names: Sequence[str] = ("SIM", "BiCGStab", "TwoStep"),
    solver_config: SolverConfig = SolverConfig(),
    circle: CircleLocalization | None = None,
    prepared_kernel: PreparedEMKernel | None = None,
    retain_solutions: bool = True,
) -> tuple[SolverExecution, ...]:
    """Run all requested solvers with a common zero initial approximation.

    The solvers themselves report residuals against the same right-hand side.
    A separate true residual is recomputed for the returned iterate and is not
    included in the solver's operation count.
    """

    operator = Operator(built_case.problem, prepared_kernel=prepared_kernel)
    backend = built_case.problem.backend
    results: list[SolverExecution] = []
    tolerance = max(
        float(solver_config.rtol),
        float(solver_config.atol)
        / max(float(backend.to_host(backend.xp.linalg.norm(built_case.problem.wave))), 1e-300),
    )
    for solver_name in solver_names:
        solver = _solver_instance(
            solver_name,
            base_config=solver_config,
            circle=circle,
        )
        _synchronize(backend)
        started = perf_counter()
        result = solver.solve(operator, built_case.problem.wave)
        _synchronize(backend)
        elapsed = perf_counter() - started
        true_residual = true_relative_residual(
            operator,
            result.u,
            built_case.problem.wave,
        )
        qualified = bool(
            result.converged
            and np.isfinite(true_residual)
            and true_residual <= tolerance * (1.0 + 1e-10)
        )
        solution_host = None
        if retain_solutions:
            solution_host = np.asarray(backend.to_host(result.u), dtype=np.complex128)
        results.append(
            SolverExecution(
                case_key=built_case.definition.key,
                solver_name=str(solver_name),
                result=result,
                elapsed_seconds=float(elapsed),
                true_relative_residual=float(true_residual),
                qualified=qualified,
                solution_host=solution_host,
            )
        )
    return tuple(results)


def select_reference_solver(
    executions: Iterable[SolverExecution],
    *,
    preferred_order: Sequence[str] = ("BiCGStab", "TwoStep", "SIM"),
) -> SolverExecution:
    candidates = [execution for execution in executions if execution.qualified]
    if not candidates:
        raise ValueError("no solver reached the requested true-residual tolerance")
    order = {name: index for index, name in enumerate(preferred_order)}
    return min(
        candidates,
        key=lambda execution: (
            order.get(execution.solver_name, len(order)),
            execution.true_relative_residual,
            execution.elapsed_seconds,
        ),
    )


def compare_fields(
    candidate: SolverExecution,
    reference: SolverExecution,
) -> FieldComparisonMetrics:
    if candidate.solution_host is None or reference.solution_host is None:
        raise ValueError("field comparison requires retained host solutions")
    candidate_field = np.asarray(candidate.solution_host, dtype=np.complex128)
    reference_field = np.asarray(reference.solution_host, dtype=np.complex128)
    if candidate_field.shape != reference_field.shape:
        raise ValueError(
            f"field shapes differ: {candidate_field.shape} != {reference_field.shape}"
        )
    component_errors = tuple(
        _relative_norm(candidate_field[index], reference_field[index])
        for index in range(3)
    )
    return FieldComparisonMetrics(
        candidate_solver=candidate.solver_name,
        reference_solver=reference.solver_name,
        relative_l2=_relative_norm(candidate_field, reference_field),
        relative_linf=_relative_linf(candidate_field, reference_field),
        component_relative_l2=component_errors,
    )


def compute_rcs_curve(
    execution: SolverExecution,
    built_case: BuiltSpectralCase,
    *,
    n_phi: int = 180,
    plane: str = "xz",
    method: str = "direct",
    batch_size: int = 64,
) -> RCSCurve:
    backend = built_case.problem.backend
    if execution.result.u is not None:
        field = execution.result.u
    elif execution.solution_host is not None:
        field = backend.array(
            execution.solution_host,
            dtype=backend.complex_dtype,
        )
    else:
        raise ValueError(
            "RCS evaluation requires either a retained device solution or a host copy"
        )
    phi, sigma = rcs_plane(
        field,
        built_case.problem,
        n_phi=n_phi,
        plane=plane,
        method=method,
        batch_size=batch_size,
    )
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    peak = float(np.max(sigma)) if sigma.size else 0.0
    normalized = sigma / peak if peak > 0.0 else np.zeros_like(sigma)
    angular_integral = float(2.0 * np.pi * np.mean(sigma)) if sigma.size else 0.0
    return RCSCurve(
        solver_name=execution.solver_name,
        plane=str(plane),
        phi=phi,
        sigma=sigma,
        sigma_normalized=normalized,
        peak=peak,
        angular_integral=angular_integral,
    )


def _peak_angle_error_degrees(
    candidate_phi: np.ndarray,
    candidate_sigma: np.ndarray,
    reference_phi: np.ndarray,
    reference_sigma: np.ndarray,
    *,
    relative_peak_tolerance: float = 1e-6,
) -> float:
    if not candidate_sigma.size or not reference_sigma.size:
        return float("nan")
    candidate_peak = float(np.max(candidate_sigma))
    reference_peak = float(np.max(reference_sigma))
    candidate_threshold = max(1e-14, abs(candidate_peak) * relative_peak_tolerance)
    reference_threshold = max(1e-14, abs(reference_peak) * relative_peak_tolerance)
    candidate_angles = candidate_phi[
        candidate_sigma >= candidate_peak - candidate_threshold
    ]
    reference_angles = reference_phi[
        reference_sigma >= reference_peak - reference_threshold
    ]
    differences = np.angle(
        np.exp(
            1j
            * (
                candidate_angles[:, None]
                - reference_angles[None, :]
            )
        )
    )
    return float(np.rad2deg(np.min(np.abs(differences))))


def compare_rcs_curves(
    candidate: RCSCurve,
    reference: RCSCurve,
) -> RCSComparisonMetrics:
    if candidate.plane != reference.plane:
        raise ValueError("RCS planes differ")
    if candidate.phi.shape != reference.phi.shape or not np.allclose(
        candidate.phi,
        reference.phi,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("RCS angular grids differ")
    peak_ratio = (
        candidate.peak / reference.peak if reference.peak > 0.0 else float("nan")
    )
    integral_ratio = (
        candidate.angular_integral / reference.angular_integral
        if reference.angular_integral > 0.0
        else float("nan")
    )
    # Symmetric scattering diagrams can have several physically equivalent
    # global maxima.  The helper compares peak sets rather than a single
    # ``argmax`` index, avoiding spurious 180-degree discrepancies.
    peak_angle_error = _peak_angle_error_degrees(
        candidate.phi,
        candidate.sigma,
        reference.phi,
        reference.sigma,
    )
    return RCSComparisonMetrics(
        candidate_solver=candidate.solver_name,
        reference_solver=reference.solver_name,
        plane=candidate.plane,
        normalized_l2=_relative_norm(
            candidate.sigma_normalized,
            reference.sigma_normalized,
        ),
        absolute_l2=_relative_norm(candidate.sigma, reference.sigma),
        absolute_linf=_relative_linf(candidate.sigma, reference.sigma),
        peak_ratio=float(peak_ratio),
        integral_ratio=float(integral_ratio),
        peak_angle_error_degrees=peak_angle_error,
    )


def _masked_relative_norm(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    if mask.shape != candidate.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} != field shape {candidate.shape[1:]}")
    if not np.any(mask):
        return float("nan")
    return _relative_norm(candidate[:, mask], reference[:, mask])


def _reference_curve_metrics(
    numerical_curve: RCSCurve,
    *,
    radius_label: str,
    radius: float,
    eps_r: complex,
    k0: float,
    numerical_field: np.ndarray,
    analytic_field: np.ndarray,
    incident_field: np.ndarray,
    inside_mask: np.ndarray,
) -> MieReferenceComparison:
    phi_mie, sigma_mie = mie_rcs_plane(
        radius,
        eps_r,
        k0,
        n_phi=len(numerical_curve.phi),
        plane=numerical_curve.plane,
    )
    if not np.allclose(phi_mie, numerical_curve.phi, rtol=0.0, atol=1e-14):
        raise RuntimeError("numerical and Mie angular grids differ")
    sigma_mie = np.asarray(sigma_mie, dtype=np.float64)
    mie_peak = float(np.max(sigma_mie)) if sigma_mie.size else 0.0
    mie_normalized = (
        sigma_mie / mie_peak if mie_peak > 0.0 else np.zeros_like(sigma_mie)
    )
    mie_integral = float(2.0 * np.pi * np.mean(sigma_mie)) if sigma_mie.size else 0.0
    numerical_scattered = numerical_field - incident_field
    analytic_scattered = analytic_field - incident_field
    return MieReferenceComparison(
        radius_label=radius_label,
        radius=float(radius),
        field_relative_l2=_relative_norm(numerical_field, analytic_field),
        field_inside_relative_l2=_masked_relative_norm(
            numerical_field,
            analytic_field,
            inside_mask,
        ),
        field_outside_relative_l2=_masked_relative_norm(
            numerical_field,
            analytic_field,
            ~inside_mask,
        ),
        scattered_field_relative_l2=_relative_norm(
            numerical_scattered,
            analytic_scattered,
        ),
        scattered_field_outside_relative_l2=_masked_relative_norm(
            numerical_scattered,
            analytic_scattered,
            ~inside_mask,
        ),
        rcs_normalized_l2=_relative_norm(
            numerical_curve.sigma_normalized,
            mie_normalized,
        ),
        rcs_absolute_l2=_relative_norm(numerical_curve.sigma, sigma_mie),
        rcs_absolute_linf=_relative_linf(numerical_curve.sigma, sigma_mie),
        rcs_peak_ratio=(
            numerical_curve.peak / mie_peak if mie_peak > 0.0 else float("nan")
        ),
        rcs_integral_ratio=(
            numerical_curve.angular_integral / mie_integral
            if mie_integral > 0.0
            else float("nan")
        ),
        rcs_peak_angle_error_degrees=_peak_angle_error_degrees(
            numerical_curve.phi,
            numerical_curve.sigma,
            np.asarray(phi_mie, dtype=np.float64),
            sigma_mie,
        ),
    )


def evaluate_mie_solution(
    execution: SolverExecution,
    built_case: BuiltSpectralCase,
    *,
    eps_r: complex,
    nominal_radius: float,
    n_phi: int = 180,
    plane: str = "xz",
    compare_farfield_backends: bool = False,
    compute_field_reference: bool = True,
) -> MieValidationResult:
    if execution.solution_host is None:
        raise ValueError("Mie validation requires a retained host solution")
    geometry = built_case.definition.feature_geometry
    if not isinstance(geometry, Ellipsoid):
        raise TypeError("Mie validation requires an Ellipsoid geometry")
    if not np.allclose(
        geometry.radii,
        (nominal_radius, nominal_radius, nominal_radius),
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("Mie validation requires a spherical feature")

    numerical_field = np.asarray(execution.solution_host, dtype=np.complex128)
    incident_field = np.asarray(
        built_case.problem.backend.to_host(built_case.problem.wave),
        dtype=np.complex128,
    )
    numerical_curve = compute_rcs_curve(
        execution,
        built_case,
        n_phi=n_phi,
        plane=plane,
        method="direct",
    )
    represented_volume = float(built_case.geometry.represented_volume)
    effective_radius = float((3.0 * represented_volume / (4.0 * pi)) ** (1.0 / 3.0))
    grid = built_case.problem.grid
    nominal_inside_mask = center_mask(geometry, grid)
    effective_geometry = Ellipsoid(
        center=geometry.center,
        radii=(effective_radius, effective_radius, effective_radius),
    )
    effective_inside_mask = center_mask(effective_geometry, grid)

    if compute_field_reference:
        nominal_field = mie_field(
            grid,
            a=float(nominal_radius),
            eps_r=complex(eps_r),
            k0=float(built_case.problem.k0),
            amplitude=built_case.definition.wave_amplitude,
            orient=built_case.definition.wave_direction,
        )
        effective_field = mie_field(
            grid,
            a=effective_radius,
            eps_r=complex(eps_r),
            k0=float(built_case.problem.k0),
            amplitude=built_case.definition.wave_amplitude,
            orient=built_case.definition.wave_direction,
        )
    else:
        nominal_field = np.full_like(numerical_field, np.nan + 1j * np.nan)
        effective_field = nominal_field

    nominal = _reference_curve_metrics(
        numerical_curve,
        radius_label="nominal",
        radius=float(nominal_radius),
        eps_r=complex(eps_r),
        k0=float(built_case.problem.k0),
        numerical_field=numerical_field,
        analytic_field=nominal_field,
        incident_field=incident_field,
        inside_mask=nominal_inside_mask,
    )
    effective = _reference_curve_metrics(
        numerical_curve,
        radius_label="effective",
        radius=effective_radius,
        eps_r=complex(eps_r),
        k0=float(built_case.problem.k0),
        numerical_field=numerical_field,
        analytic_field=effective_field,
        incident_field=incident_field,
        inside_mask=effective_inside_mask,
    )

    backend_error = None
    if compare_farfield_backends:
        fft_curve = compute_rcs_curve(
            execution,
            built_case,
            n_phi=n_phi,
            plane=plane,
            method="fft",
        )
        backend_error = _relative_norm(fft_curve.sigma, numerical_curve.sigma)

    return MieValidationResult(
        case_key=built_case.definition.key,
        solver_name=execution.solver_name,
        nominal_radius=float(nominal_radius),
        effective_radius=effective_radius,
        represented_volume=represented_volume,
        exact_volume=float(built_case.geometry.exact_volume),
        relative_volume_error=float(built_case.geometry.relative_volume_error),
        nominal=nominal,
        effective=effective,
        farfield_backend_relative_l2=backend_error,
    )


__all__ = [
    "FieldComparisonMetrics",
    "MieReferenceComparison",
    "MieValidationResult",
    "RCSComparisonMetrics",
    "RCSCurve",
    "SolverExecution",
    "compare_fields",
    "compare_rcs_curves",
    "compute_rcs_curve",
    "estimate_parameter_circle",
    "evaluate_mie_solution",
    "run_solver_comparison",
    "select_reference_solver",
    "true_relative_residual",
]
