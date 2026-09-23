from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import em3d
from em3d.experiments.spectral_transfer import (
    BuiltSpectralCase,
    SpectralCaseDefinition,
    build_spectral_case,
)
from em3d.experiments.stationary_validation import (
    FieldComparisonMetrics,
    MieValidationResult,
    RCSComparisonMetrics,
    RCSCurve,
    SolverExecution,
    compare_fields,
    compare_rcs_curves,
    compute_rcs_curve,
    estimate_parameter_circle,
    evaluate_mie_solution,
    run_solver_comparison,
    select_reference_solver,
)
from em3d.geometry import SamplingMode

from experiments.em_spectral.artifacts import ArtifactStore

from .cases import mie_sphere_case, stationary_case_catalog
from .config import RuntimeConfig, ValidationStudyConfig


@dataclass(frozen=True)
class StationaryCaseStudyResult:
    definition: SpectralCaseDefinition
    built_case: BuiltSpectralCase
    circle: object | None
    circle_error: str | None
    executions: tuple[SolverExecution, ...]
    reference_solver: str | None
    field_metrics: tuple[FieldComparisonMetrics, ...]
    rcs_curves: tuple[RCSCurve, ...]
    rcs_metrics: tuple[RCSComparisonMetrics, ...]


@dataclass(frozen=True)
class MieFieldSliceSet:
    solver_name: str
    plane: str
    horizontal: np.ndarray
    vertical: np.ndarray
    numerical_total: np.ndarray
    analytic_nominal_total: np.ndarray
    analytic_effective_total: np.ndarray
    numerical_scattered: np.ndarray
    analytic_nominal_scattered: np.ndarray
    analytic_effective_scattered: np.ndarray


@dataclass(frozen=True)
class MieRCSCurveSet:
    solver_name: str
    phi: np.ndarray
    sigma_numerical: np.ndarray
    sigma_nominal: np.ndarray
    sigma_effective: np.ndarray


@dataclass(frozen=True)
class MieCaseStudyResult:
    definition: SpectralCaseDefinition
    grid_size: int
    circle_error: str | None
    executions: tuple[SolverExecution, ...]
    validations: tuple[MieValidationResult, ...]
    rcs_curves: tuple[MieRCSCurveSet, ...]
    field_slices: tuple[MieFieldSliceSet, ...]


@dataclass(frozen=True)
class ValidationSuiteResult:
    config: ValidationStudyConfig
    stationary_cases: tuple[StationaryCaseStudyResult, ...]
    mie_cases: tuple[MieCaseStudyResult, ...]
    output_root: str


def _backend(runtime: RuntimeConfig) -> em3d.Backend:
    precision = (
        em3d.Precision.SINGLE
        if runtime.precision == "single"
        else em3d.Precision.DOUBLE
    )
    if runtime.device == "cpu":
        return em3d.Backend.numpy(precision)
    if runtime.device == "cuda":
        return em3d.Backend.cupy(precision)
    return em3d.Backend.auto(precision)


def _central_magnitude_slice(field: np.ndarray, grid, *, plane: str):
    magnitude = np.sqrt(np.sum(np.abs(field) ** 2, axis=0))
    x = np.asarray(grid.backend.to_host(grid.x), dtype=np.float64)
    y = np.asarray(grid.backend.to_host(grid.y), dtype=np.float64)
    z = np.asarray(grid.backend.to_host(grid.z), dtype=np.float64)
    if plane == "xz":
        index = magnitude.shape[1] // 2
        return x, z, magnitude[:, index, :]
    if plane == "xy":
        index = magnitude.shape[2] // 2
        return x, y, magnitude[:, :, index]
    if plane == "yz":
        index = magnitude.shape[0] // 2
        return y, z, magnitude[index, :, :]
    raise ValueError(f"unsupported plane {plane!r}")


def _build_mie_field_slice(
    execution: SolverExecution,
    built: BuiltSpectralCase,
    validation: MieValidationResult,
    *,
    eps_r: complex,
    plane: str,
) -> MieFieldSliceSet:
    if execution.solution_host is None:
        raise ValueError("Mie field slice requires a retained solution")
    grid = built.problem.grid
    numerical = np.asarray(execution.solution_host, dtype=np.complex128)
    incident = np.asarray(
        grid.backend.to_host(built.problem.wave),
        dtype=np.complex128,
    )
    nominal = em3d.mie.mie_field(
        grid,
        a=validation.nominal_radius,
        eps_r=eps_r,
        k0=built.problem.k0,
        amplitude=built.definition.wave_amplitude,
        orient=built.definition.wave_direction,
    )
    effective = em3d.mie.mie_field(
        grid,
        a=validation.effective_radius,
        eps_r=eps_r,
        k0=built.problem.k0,
        amplitude=built.definition.wave_amplitude,
        orient=built.definition.wave_direction,
    )
    horizontal, vertical, numerical_total = _central_magnitude_slice(
        numerical, grid, plane=plane
    )
    _, _, nominal_total = _central_magnitude_slice(nominal, grid, plane=plane)
    _, _, effective_total = _central_magnitude_slice(effective, grid, plane=plane)
    _, _, numerical_scattered = _central_magnitude_slice(
        numerical - incident, grid, plane=plane
    )
    _, _, nominal_scattered = _central_magnitude_slice(
        nominal - incident, grid, plane=plane
    )
    _, _, effective_scattered = _central_magnitude_slice(
        effective - incident, grid, plane=plane
    )
    return MieFieldSliceSet(
        solver_name=execution.solver_name,
        plane=plane,
        horizontal=horizontal,
        vertical=vertical,
        numerical_total=numerical_total,
        analytic_nominal_total=nominal_total,
        analytic_effective_total=effective_total,
        numerical_scattered=numerical_scattered,
        analytic_nominal_scattered=nominal_scattered,
        analytic_effective_scattered=effective_scattered,
    )


def _circle_row(
    case_key: str,
    circle,
    error: str | None,
    *,
    coarse_grid_size: int,
) -> dict[str, Any]:
    if circle is None:
        return {
            "case_key": case_key,
            "status": "unavailable",
            "coarse_grid_size": int(coarse_grid_size),
            "error": error or "unknown error",
            "mu_real": np.nan,
            "mu_imag": np.nan,
            "radius": np.nan,
            "q": np.nan,
            "margin": np.nan,
        }
    return {
        "case_key": case_key,
        "status": "ok",
        "coarse_grid_size": int(coarse_grid_size),
        "error": "",
        "mu_real": float(np.real(circle.mu)),
        "mu_imag": float(np.imag(circle.mu)),
        "radius": float(circle.radius),
        "q": float(circle.q),
        "margin": float(circle.margin),
    }


def _solver_row(
    execution: SolverExecution,
    *,
    grid_size: int,
    case_title: str,
    circle,
    study: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = execution.result
    row = {
        "study": study,
        "case_key": execution.case_key,
        "case_title": case_title,
        "grid_size": int(grid_size),
        "dof": int(np.prod(result.u.shape)) if result.u is not None else 3 * grid_size**3,
        "solver_name": execution.solver_name,
        "status": result.status,
        "converged": bool(result.converged),
        "qualified": bool(execution.qualified),
        "iterations": int(result.iterations),
        "matvec_count": int(result.matvec_count),
        "rmatvec_count": int(result.rmatvec_count),
        "operator_action_count": int(result.operator_action_count),
        "reported_final_residual": float(result.residual_history[-1])
        if result.residual_history
        else np.nan,
        "true_final_residual": float(execution.true_relative_residual),
        "elapsed_seconds": float(execution.elapsed_seconds),
        "mu_real": float(np.real(circle.mu)) if circle is not None else np.nan,
        "mu_imag": float(np.imag(circle.mu)) if circle is not None else np.nan,
        "circle_radius": float(circle.radius) if circle is not None else np.nan,
        "circle_q": float(circle.q) if circle is not None else np.nan,
    }
    if extra:
        row.update(extra)
    return row


def _missing_sim_row(
    *,
    study: str,
    case_key: str,
    case_title: str,
    grid_size: int,
    error: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "study": study,
        "case_key": case_key,
        "case_title": case_title,
        "grid_size": int(grid_size),
        "dof": int(3 * grid_size**3),
        "solver_name": "SIM",
        "status": "parameter_unavailable",
        "converged": False,
        "qualified": False,
        "iterations": 0,
        "matvec_count": 0,
        "rmatvec_count": 0,
        "operator_action_count": 0,
        "reported_final_residual": np.nan,
        "true_final_residual": np.nan,
        "elapsed_seconds": 0.0,
        "mu_real": np.nan,
        "mu_imag": np.nan,
        "circle_radius": np.nan,
        "circle_q": np.nan,
        "error": error,
    }
    if extra:
        row.update(extra)
    return row


def _field_row(case_key: str, metric: FieldComparisonMetrics) -> dict[str, Any]:
    return {
        "case_key": case_key,
        "candidate_solver": metric.candidate_solver,
        "reference_solver": metric.reference_solver,
        "relative_l2": metric.relative_l2,
        "relative_linf": metric.relative_linf,
        "component_0_relative_l2": metric.component_relative_l2[0],
        "component_1_relative_l2": metric.component_relative_l2[1],
        "component_2_relative_l2": metric.component_relative_l2[2],
    }


def _rcs_row(case_key: str, metric: RCSComparisonMetrics) -> dict[str, Any]:
    return {
        "case_key": case_key,
        **asdict(metric),
    }


def _mie_row(
    validation: MieValidationResult,
    *,
    grid_size: int,
    eps_r: complex,
    k0a: float,
    true_residual: float,
    qualified: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_key": validation.case_key,
        "grid_size": int(grid_size),
        "eps_real": float(np.real(eps_r)),
        "eps_imag": float(np.imag(eps_r)),
        "k0a": float(k0a),
        "solver_name": validation.solver_name,
        "qualified": bool(qualified),
        "true_final_residual": float(true_residual),
        "nominal_radius": validation.nominal_radius,
        "effective_radius": validation.effective_radius,
        "represented_volume": validation.represented_volume,
        "exact_volume": validation.exact_volume,
        "relative_volume_error": validation.relative_volume_error,
        "farfield_backend_relative_l2": validation.farfield_backend_relative_l2,
    }
    for prefix, comparison in (
        ("nominal", validation.nominal),
        ("effective", validation.effective),
    ):
        for key, value in asdict(comparison).items():
            if key not in {"radius_label", "radius"}:
                row[f"{prefix}_{key}"] = value
    return row


def _write_execution_raw(
    store: ArtifactStore,
    execution: SolverExecution,
    *,
    stem: str,
    save_solution: bool,
) -> None:
    result = execution.result
    store.write_json(
        f"raw/residual_histories/{stem}.json",
        {
            "solver_name": execution.solver_name,
            "status": result.status,
            "converged": result.converged,
            "qualified": execution.qualified,
            "iterations": result.iterations,
            "matvec_count": result.matvec_count,
            "rmatvec_count": result.rmatvec_count,
            "operator_action_count": result.operator_action_count,
            "true_final_residual": execution.true_relative_residual,
            "residual_action_counts": result.residual_action_counts,
            "residual_history": result.residual_history,
        },
    )
    if save_solution and execution.solution_host is not None:
        store.write_npz(
            f"raw/solutions/{stem}.npz",
            field=execution.solution_host,
        )


def _run_stationary_cases(
    config: ValidationStudyConfig,
    *,
    store: ArtifactStore,
) -> tuple[StationaryCaseStudyResult, ...]:
    catalog = stationary_case_catalog()
    backend = _backend(config.runtime)
    solver_rows: list[dict[str, Any]] = []
    circle_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    rcs_rows: list[dict[str, Any]] = []
    studies: list[StationaryCaseStudyResult] = []

    for key in config.solver.case_keys:
        if key not in catalog:
            raise KeyError(f"unknown stationary validation case {key!r}")
        definition = catalog[key]
        circle = None
        circle_error = None
        coarse_size = config.solver.coarse_size_for(key)
        if "SIM" in config.solver.solver_names:
            try:
                circle = estimate_parameter_circle(
                    definition,
                    coarse_grid_shape=coarse_size,
                )
            except Exception as exc:  # preserve non-SIM runs
                circle_error = str(exc)
        circle_rows.append(
            _circle_row(
                definition.key,
                circle,
                circle_error,
                coarse_grid_size=coarse_size,
            )
        )

        built = build_spectral_case(
            definition,
            grid_shape=config.solver.grid_size,
            backend=backend,
            sampling_mode=SamplingMode.CELL_CENTER,
        )
        active_solvers = tuple(
            name
            for name in config.solver.solver_names
            if name != "SIM" or circle is not None
        )
        executions = run_solver_comparison(
            built,
            solver_names=active_solvers,
            solver_config=em3d.SolverConfig(
                max_iter=config.solver.max_iter,
                rtol=config.solver.rtol,
                divergence_guard=config.solver.divergence_guard,
            ),
            circle=circle,
            retain_solutions=True,
        )
        if "SIM" in config.solver.solver_names and circle is None:
            solver_rows.append(
                _missing_sim_row(
                    study="solver_field_rcs",
                    case_key=definition.key,
                    case_title=definition.title,
                    grid_size=config.solver.grid_size,
                    error=circle_error or "spectral parameter unavailable",
                    extra={"sim_coarse_size": int(coarse_size)},
                )
            )
        for execution in executions:
            solver_rows.append(
                _solver_row(
                    execution,
                    grid_size=config.solver.grid_size,
                    case_title=definition.title,
                    circle=circle,
                    study="solver_field_rcs",
                    extra={"sim_coarse_size": int(coarse_size)},
                )
            )
            _write_execution_raw(
                store,
                execution,
                stem=f"solver/{definition.key}_{execution.solver_name}",
                save_solution=config.solver.save_solutions,
            )

        reference_solver = None
        field_metrics: list[FieldComparisonMetrics] = []
        curves: list[RCSCurve] = []
        curve_metrics: list[RCSComparisonMetrics] = []
        try:
            reference = select_reference_solver(executions)
            reference_solver = reference.solver_name
            for execution in executions:
                if execution.qualified:
                    metric = compare_fields(execution, reference)
                    field_metrics.append(metric)
                    field_rows.append(_field_row(definition.key, metric))
            for plane in config.solver.rcs_planes:
                plane_curves = [
                    compute_rcs_curve(
                        execution,
                        built,
                        n_phi=config.solver.rcs_n_phi,
                        plane=plane,
                    )
                    for execution in executions
                    if execution.qualified
                ]
                curves.extend(plane_curves)
                reference_curve = next(
                    curve
                    for curve in plane_curves
                    if curve.solver_name == reference.solver_name
                )
                for curve in plane_curves:
                    metric = compare_rcs_curves(curve, reference_curve)
                    curve_metrics.append(metric)
                    rcs_rows.append(_rcs_row(definition.key, metric))
                    store.write_npz(
                        f"raw/rcs/solver/{definition.key}_{plane}_{curve.solver_name}.npz",
                        phi=curve.phi,
                        sigma=curve.sigma,
                        sigma_normalized=curve.sigma_normalized,
                    )
        except ValueError:
            pass

        studies.append(
            StationaryCaseStudyResult(
                definition=definition,
                built_case=built,
                circle=circle,
                circle_error=circle_error,
                executions=executions,
                reference_solver=reference_solver,
                field_metrics=tuple(field_metrics),
                rcs_curves=tuple(curves),
                rcs_metrics=tuple(curve_metrics),
            )
        )

    store.write_rows("tables/solver_parameter_circles.csv", circle_rows)
    store.write_rows("tables/solver_runs.csv", solver_rows)
    store.write_rows("tables/field_comparisons.csv", field_rows)
    store.write_rows("tables/rcs_solver_comparisons.csv", rcs_rows)
    return tuple(studies)


def _run_mie_cases(
    config: ValidationStudyConfig,
    *,
    store: ArtifactStore,
) -> tuple[MieCaseStudyResult, ...]:
    backend = _backend(config.runtime)
    solver_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    selected_validation_rows: list[dict[str, Any]] = []
    studies: list[MieCaseStudyResult] = []
    circle_cache: dict[tuple[complex, float], tuple[object | None, str | None]] = {}
    smallest_grid = min(config.mie.grid_sizes)

    for eps_r in config.mie.eps_values:
        for k0a in config.mie.k0a_values:
            definition = mie_sphere_case(
                eps_r=eps_r,
                k0a=k0a,
                radius=config.mie.radius,
                domain_length=config.mie.domain_length,
            )
            cache_key = (complex(eps_r), float(k0a))
            circle = None
            circle_error = None
            if "SIM" in config.mie.solver_names:
                if cache_key not in circle_cache:
                    try:
                        circle_cache[cache_key] = (
                            estimate_parameter_circle(
                                definition,
                                coarse_grid_shape=config.mie.sim_coarse_size,
                            ),
                            None,
                        )
                    except Exception as exc:
                        circle_cache[cache_key] = (None, str(exc))
                circle, circle_error = circle_cache[cache_key]

            for grid_size in config.mie.grid_sizes:
                built = build_spectral_case(
                    definition,
                    grid_shape=grid_size,
                    backend=backend,
                    sampling_mode=SamplingMode.CELL_CENTER,
                )
                active_solvers = tuple(
                    name
                    for name in config.mie.solver_names
                    if name != "SIM" or circle is not None
                )
                executions = run_solver_comparison(
                    built,
                    solver_names=active_solvers,
                    solver_config=em3d.SolverConfig(
                        max_iter=config.mie.max_iter,
                        rtol=config.mie.rtol,
                        divergence_guard=config.mie.divergence_guard,
                    ),
                    circle=circle,
                    retain_solutions=True,
                )
                extra = {
                    "eps_real": float(np.real(eps_r)),
                    "eps_imag": float(np.imag(eps_r)),
                    "k0a": float(k0a),
                    "radius": float(config.mie.radius),
                }
                if "SIM" in config.mie.solver_names and circle is None:
                    solver_rows.append(
                        _missing_sim_row(
                            study="mie",
                            case_key=definition.key,
                            case_title=definition.title,
                            grid_size=grid_size,
                            error=circle_error or "spectral parameter unavailable",
                            extra=extra,
                        )
                    )

                validations: list[MieValidationResult] = []
                curve_sets: list[MieRCSCurveSet] = []
                field_slices: list[MieFieldSliceSet] = []
                for execution in executions:
                    solver_rows.append(
                        _solver_row(
                            execution,
                            grid_size=grid_size,
                            case_title=definition.title,
                            circle=circle,
                            study="mie",
                            extra=extra,
                        )
                    )
                    stem = (
                        f"mie/{definition.key}_N{grid_size}_{execution.solver_name}"
                    )
                    _write_execution_raw(
                        store,
                        execution,
                        stem=stem,
                        save_solution=config.mie.save_solutions,
                    )
                    if not execution.qualified:
                        continue
                    validation = evaluate_mie_solution(
                        execution,
                        built,
                        eps_r=eps_r,
                        nominal_radius=config.mie.radius,
                        n_phi=config.mie.rcs_n_phi,
                        plane=config.mie.rcs_plane,
                        compare_farfield_backends=(
                            config.mie.compare_farfield_backends_on_smallest_grid
                            and grid_size == smallest_grid
                        ),
                        compute_field_reference=(
                            grid_size in config.mie.field_reference_grids
                        ),
                    )
                    validations.append(validation)
                    validation_rows.append(
                        _mie_row(
                            validation,
                            grid_size=grid_size,
                            eps_r=eps_r,
                            k0a=k0a,
                            true_residual=execution.true_relative_residual,
                            qualified=execution.qualified,
                        )
                    )
                    numerical_curve = compute_rcs_curve(
                        execution,
                        built,
                        n_phi=config.mie.rcs_n_phi,
                        plane=config.mie.rcs_plane,
                    )
                    phi_nominal, sigma_nominal = em3d.mie.mie_rcs_plane(
                        config.mie.radius,
                        eps_r,
                        built.problem.k0,
                        n_phi=config.mie.rcs_n_phi,
                        plane=config.mie.rcs_plane,
                    )
                    phi_effective, sigma_effective = em3d.mie.mie_rcs_plane(
                        validation.effective_radius,
                        eps_r,
                        built.problem.k0,
                        n_phi=config.mie.rcs_n_phi,
                        plane=config.mie.rcs_plane,
                    )
                    curve_sets.append(
                        MieRCSCurveSet(
                            solver_name=execution.solver_name,
                            phi=np.asarray(numerical_curve.phi, dtype=np.float64),
                            sigma_numerical=np.asarray(numerical_curve.sigma, dtype=np.float64),
                            sigma_nominal=np.asarray(sigma_nominal, dtype=np.float64),
                            sigma_effective=np.asarray(sigma_effective, dtype=np.float64),
                        )
                    )
                    store.write_npz(
                        f"raw/rcs/{stem}.npz",
                        phi=numerical_curve.phi,
                        sigma_numerical=numerical_curve.sigma,
                        sigma_nominal=sigma_nominal,
                        sigma_effective=sigma_effective,
                        phi_nominal=phi_nominal,
                        phi_effective=phi_effective,
                    )

                if validations:
                    validation_by_solver = {
                        item.solver_name: item for item in validations
                    }
                    qualified_candidates = [
                        execution
                        for execution in executions
                        if execution.solver_name in validation_by_solver
                    ]
                    selected_execution = min(
                        qualified_candidates,
                        key=lambda execution: (
                            0 if execution.solver_name == "BiCGStab" else 1,
                            execution.true_relative_residual,
                        ),
                    )
                    selected_row = _mie_row(
                        validation_by_solver[selected_execution.solver_name],
                        grid_size=grid_size,
                        eps_r=eps_r,
                        k0a=k0a,
                        true_residual=selected_execution.true_relative_residual,
                        qualified=selected_execution.qualified,
                    )
                    selected_row["selection_basis"] = (
                        "preferred_bicgstab"
                        if selected_execution.solver_name == "BiCGStab"
                        else "smallest_true_residual"
                    )
                    selected_validation_rows.append(selected_row)
                    if grid_size in config.mie.field_reference_grids:
                        field_slice = _build_mie_field_slice(
                            selected_execution,
                            built,
                            validation_by_solver[selected_execution.solver_name],
                            eps_r=eps_r,
                            plane=config.mie.rcs_plane,
                        )
                        field_slices.append(field_slice)
                        store.write_npz(
                            f"raw/field_slices/{definition.key}_N{grid_size}_{selected_execution.solver_name}.npz",
                            horizontal=field_slice.horizontal,
                            vertical=field_slice.vertical,
                            numerical_total=field_slice.numerical_total,
                            analytic_nominal_total=field_slice.analytic_nominal_total,
                            analytic_effective_total=field_slice.analytic_effective_total,
                            numerical_scattered=field_slice.numerical_scattered,
                            analytic_nominal_scattered=field_slice.analytic_nominal_scattered,
                            analytic_effective_scattered=field_slice.analytic_effective_scattered,
                        )

                compact_executions = tuple(
                    replace(
                        execution,
                        result=replace(execution.result, u=None),
                        solution_host=None,
                    )
                    for execution in executions
                )
                studies.append(
                    MieCaseStudyResult(
                        definition=definition,
                        grid_size=int(grid_size),
                        circle_error=circle_error,
                        executions=compact_executions,
                        validations=tuple(validations),
                        rcs_curves=tuple(curve_sets),
                        field_slices=tuple(field_slices),
                    )
                )

    store.write_rows("tables/mie_solver_runs.csv", solver_rows)
    store.write_rows("tables/mie_validation.csv", validation_rows)
    store.write_rows(
        "tables/mie_validation_selected.csv",
        selected_validation_rows,
    )
    return tuple(studies)


def run_validation_suite(
    config: ValidationStudyConfig,
    *,
    include_solver_study: bool = True,
    include_mie_study: bool = True,
    render_figures: bool = True,
) -> ValidationSuiteResult:
    store = ArtifactStore(
        config.output_root,
        schema="em3d-stationary-validation-artifacts-v1",
    )
    store.write_json("config.json", config)

    stationary = (
        _run_stationary_cases(config, store=store)
        if include_solver_study
        else ()
    )
    mie = _run_mie_cases(config, store=store) if include_mie_study else ()

    protocol = [
        {
            "experiment": "V1",
            "purpose": "solver comparison by true residual and operator actions",
            "executed": bool(include_solver_study),
        },
        {
            "experiment": "V2",
            "purpose": "field agreement and shared-scale slice diagnostics",
            "executed": bool(include_solver_study),
        },
        {
            "experiment": "V3",
            "purpose": "absolute and normalized RCS agreement between solvers",
            "executed": bool(include_solver_study),
        },
        {
            "experiment": "V4",
            "purpose": "Mie verification with nominal and effective sphere radii",
            "executed": bool(include_mie_study),
        },
    ]
    store.write_rows("tables/FINAL_experimental_protocol.csv", protocol)

    if render_figures:
        from .plots import render_validation_figures

        render_validation_figures(
            stationary,
            mie,
            store=store,
        )

    store.finalize(config=config)
    return ValidationSuiteResult(
        config=config,
        stationary_cases=tuple(stationary),
        mie_cases=tuple(mie),
        output_root=str(Path(config.output_root)),
    )


__all__ = [
    "MieCaseStudyResult",
    "MieFieldSliceSet",
    "MieRCSCurveSet",
    "StationaryCaseStudyResult",
    "ValidationSuiteResult",
    "run_validation_suite",
]
