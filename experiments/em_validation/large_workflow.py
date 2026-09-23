from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
import gc
import json
import math
import shutil

import numpy as np

import em3d
from em3d.experiments.spectral_transfer import (
    BuiltSpectralCase,
    GridSpectrumRun,
    SpectralCaseDefinition,
    build_ensemble_parameter,
    build_spectral_case,
    compute_grid_spectrum,
)
from em3d.experiments.stationary_validation import (
    RCSCurve,
    SolverExecution,
    compare_fields,
    compare_rcs_curves,
    compute_rcs_curve,
    run_solver_execution,
)
from em3d.geometry import Ellipsoid, SamplingMode, center_mask
from em3d.operator import Operator, PreparedEMKernel
from em3d.spectral import CircleLocalization

from experiments.em_spectral.artifacts import ArtifactStore

from .cases import mie_sphere_case, stationary_case_catalog
from .config import LargeGridStudyConfig, MieJobSpec, StationaryGridJobSpec
from .large_plots import (
    plot_grid_metric,
    plot_mie_error_phase_panel,
    plot_mie_field_panel,
    plot_rcs_four_panel,
    plot_solver_residuals,
    plot_stationary_field_panel,
)
from .memory import (
    MemoryCheckpoint,
    clear_cuda_runtime_caches,
    estimate_large_grid_memory,
    memory_checkpoint,
    memory_decision,
    run_operator_memory_probe,
)


@dataclass(frozen=True)
class LargeGridSuiteResult:
    profile: str
    output_root: str
    manifest_path: str
    job_rows: tuple[dict[str, Any], ...]
    solver_rows: tuple[dict[str, Any], ...]
    common_solvability_rows: tuple[dict[str, Any], ...]


def _backend(config: LargeGridStudyConfig) -> em3d.Backend:
    precision = (
        em3d.Precision.SINGLE
        if config.runtime.precision == "single"
        else em3d.Precision.DOUBLE
    )
    if config.runtime.device == "cpu":
        return em3d.Backend.numpy(precision)
    if config.runtime.device == "cuda":
        return em3d.Backend.cupy(precision)
    return em3d.Backend.auto(precision)


def _progress(config: LargeGridStudyConfig, message: str) -> None:
    if config.runtime.progress:
        print(message, flush=True)


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(candidate) - np.asarray(reference)))
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _relative_linf(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    numerator = float(np.max(np.abs(candidate - reference))) if candidate.size else 0.0
    denominator = float(np.max(np.abs(reference))) if reference.size else 0.0
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _masked_relative_l2(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return float("nan")
    return _relative_l2(candidate[:, mask], reference[:, mask])


def _solver_config(
    config: LargeGridStudyConfig,
    solver_name: str,
    *,
    rtol: float,
) -> em3d.SolverConfig:
    suite = config.solver_suite
    return em3d.SolverConfig(
        max_iter=int(suite.max_iter),
        rtol=float(rtol),
        divergence_guard=float(suite.divergence_guard),
        max_operator_actions=suite.action_budget(solver_name),
    )


def _solver_row(
    execution: SolverExecution,
    *,
    job_key: str,
    case_title: str,
    grid_size: int,
    tier: str,
    requested_rtol: float,
    circle: CircleLocalization | None,
    study: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = execution.result
    row: dict[str, Any] = {
        "study": study,
        "job_key": job_key,
        "case_key": execution.case_key,
        "case_title": case_title,
        "tier": tier,
        "grid_size": int(grid_size),
        "dof": int(3 * int(grid_size) ** 3),
        "solver_name": execution.solver_name,
        "status": result.status,
        "converged": bool(result.converged),
        "qualified": bool(execution.qualified),
        "requested_true_rtol": float(requested_rtol),
        "iterations": int(result.iterations),
        "matvec_count": int(result.matvec_count),
        "rmatvec_count": int(result.rmatvec_count),
        "operator_action_count": int(result.operator_action_count),
        "reported_final_residual": (
            float(result.residual_history[-1]) if result.residual_history else np.nan
        ),
        "true_final_residual": float(execution.true_relative_residual),
        "elapsed_seconds": float(execution.elapsed_seconds),
        "mu_real": float(np.real(circle.mu)) if circle is not None else np.nan,
        "mu_imag": float(np.imag(circle.mu)) if circle is not None else np.nan,
        "circle_radius": float(circle.radius) if circle is not None else np.nan,
        "circle_q": float(circle.q) if circle is not None else np.nan,
        "circle_margin": float(circle.margin) if circle is not None else np.nan,
    }
    if extra:
        row.update(dict(extra))
    return row


def _missing_solver_row(
    *,
    solver_name: str,
    status: str,
    job_key: str,
    case_key: str,
    case_title: str,
    grid_size: int,
    tier: str,
    study: str,
    requested_rtol: float,
    error: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "study": study,
        "job_key": job_key,
        "case_key": case_key,
        "case_title": case_title,
        "tier": tier,
        "grid_size": int(grid_size),
        "dof": int(3 * int(grid_size) ** 3),
        "solver_name": solver_name,
        "status": status,
        "converged": False,
        "qualified": False,
        "requested_true_rtol": float(requested_rtol),
        "iterations": 0,
        "matvec_count": 0,
        "rmatvec_count": 0,
        "operator_action_count": 0,
        "reported_final_residual": np.nan,
        "true_final_residual": np.nan,
        "elapsed_seconds": 0.0,
        "error": str(error),
        "mu_real": np.nan,
        "mu_imag": np.nan,
        "circle_radius": np.nan,
        "circle_q": np.nan,
        "circle_margin": np.nan,
    }
    if extra:
        row.update(dict(extra))
    return row


def _analytic_rcs_curve(
    radius: float,
    eps_r: complex,
    k0: float,
    *,
    n_phi: int,
    plane: str,
    label: str,
) -> RCSCurve:
    phi, sigma = em3d.mie.mie_rcs_plane(
        radius, eps_r, k0, n_phi=n_phi, plane=plane
    )
    phi = np.asarray(phi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    peak = float(np.max(sigma)) if sigma.size else 0.0
    normalized = sigma / peak if peak > 0.0 else np.zeros_like(sigma)
    return RCSCurve(
        solver_name=label,
        plane=plane,
        phi=phi,
        sigma=sigma,
        sigma_normalized=normalized,
        peak=peak,
        angular_integral=float(2.0 * np.pi * np.mean(sigma)) if sigma.size else 0.0,
    )


def _field_reference_rows(
    *,
    job: MieJobSpec,
    execution: SolverExecution,
    numerical: np.ndarray,
    incident: np.ndarray,
    analytic_nominal: np.ndarray,
    analytic_effective: np.ndarray,
    nominal_mask: np.ndarray,
    effective_mask: np.ndarray,
    nominal_radius: float,
    effective_radius: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, radius, analytic, mask in (
        ("nominal", nominal_radius, analytic_nominal, nominal_mask),
        ("effective", effective_radius, analytic_effective, effective_mask),
    ):
        numerical_scattered = numerical - incident
        analytic_scattered = analytic - incident
        rows.append(
            {
                "job_key": job.key,
                "case_key": execution.case_key,
                "grid_size": job.grid_size,
                "eps_real": float(np.real(job.eps_r)),
                "eps_imag": float(np.imag(job.eps_r)),
                "k0a": float(job.k0a),
                "solver_name": execution.solver_name,
                "radius_label": label,
                "reference_radius": float(radius),
                "field_relative_l2": _relative_l2(numerical, analytic),
                "field_inside_relative_l2": _masked_relative_l2(
                    numerical, analytic, mask
                ),
                "field_outside_relative_l2": _masked_relative_l2(
                    numerical, analytic, ~mask
                ),
                "scattered_field_relative_l2": _relative_l2(
                    numerical_scattered, analytic_scattered
                ),
                "scattered_field_outside_relative_l2": _masked_relative_l2(
                    numerical_scattered, analytic_scattered, ~mask
                ),
            }
        )
    return rows


def _rcs_reference_row(
    *,
    job_key: str,
    case_key: str,
    grid_size: int,
    eps_r: complex,
    k0a: float,
    solver_curve: RCSCurve,
    reference_curve: RCSCurve,
    radius_label: str,
) -> dict[str, Any]:
    metrics = compare_rcs_curves(solver_curve, reference_curve)
    return {
        "job_key": job_key,
        "case_key": case_key,
        "grid_size": int(grid_size),
        "eps_real": float(np.real(eps_r)),
        "eps_imag": float(np.imag(eps_r)),
        "k0a": float(k0a),
        "solver_name": solver_curve.solver_name,
        "plane": solver_curve.plane,
        "radius_label": radius_label,
        **asdict(metrics),
    }


def _pairwise_field_rows(
    job_key: str,
    grid_size: int,
    executions: Sequence[SolverExecution],
) -> list[dict[str, Any]]:
    qualified = [item for item in executions if item.qualified and item.solution_host is not None]
    rows: list[dict[str, Any]] = []
    for i, candidate in enumerate(qualified):
        for reference in qualified[i + 1 :]:
            metrics = compare_fields(candidate, reference)
            rows.append(
                {
                    "job_key": job_key,
                    "grid_size": int(grid_size),
                    **asdict(metrics),
                }
            )
    return rows


def _pairwise_rcs_rows(
    job_key: str,
    grid_size: int,
    curves: Sequence[RCSCurve],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_plane: dict[str, list[RCSCurve]] = {}
    for curve in curves:
        by_plane.setdefault(curve.plane, []).append(curve)
    for plane, plane_curves in by_plane.items():
        for i, candidate in enumerate(plane_curves):
            for reference in plane_curves[i + 1 :]:
                rows.append(
                    {
                        "job_key": job_key,
                        "grid_size": int(grid_size),
                        **asdict(compare_rcs_curves(candidate, reference)),
                    }
                )
    return rows


def _slice_coordinates(grid, plane: str):
    x = np.asarray(grid.backend.to_host(grid.x), dtype=float)
    y = np.asarray(grid.backend.to_host(grid.y), dtype=float)
    z = np.asarray(grid.backend.to_host(grid.z), dtype=float)
    if plane == "xy":
        H, V = np.meshgrid(x, y, indexing="ij")
        fixed = float(z[len(z) // 2])
        xyz = np.column_stack([H.ravel(), V.ravel(), np.full(H.size, fixed)])
        return x, y, xyz
    if plane == "xz":
        H, V = np.meshgrid(x, z, indexing="ij")
        fixed = float(y[len(y) // 2])
        xyz = np.column_stack([H.ravel(), np.full(H.size, fixed), V.ravel()])
        return x, z, xyz
    if plane == "yz":
        H, V = np.meshgrid(y, z, indexing="ij")
        fixed = float(x[len(x) // 2])
        xyz = np.column_stack([np.full(H.size, fixed), H.ravel(), V.ravel()])
        return y, z, xyz
    raise ValueError(f"unsupported plane {plane!r}")


def _slice_field(field: np.ndarray, plane: str) -> np.ndarray:
    if plane == "xy":
        return np.asarray(field[:, :, :, field.shape[3] // 2])
    if plane == "xz":
        return np.asarray(field[:, :, field.shape[2] // 2, :])
    if plane == "yz":
        return np.asarray(field[:, field.shape[1] // 2, :, :])
    raise ValueError(f"unsupported plane {plane!r}")


def _slice_payload(
    total: np.ndarray,
    incident: np.ndarray,
    analytic_total: np.ndarray | None,
    *,
    component: int,
    phase_mask_fraction: float,
) -> dict[str, np.ndarray]:
    scattered = total - incident
    total_magnitude = np.sqrt(np.sum(np.abs(total) ** 2, axis=0))
    scattered_magnitude = np.sqrt(np.sum(np.abs(scattered) ** 2, axis=0))
    component_values = scattered[component]
    component_abs = np.abs(component_values)
    phase = np.angle(component_values)
    threshold = phase_mask_fraction * max(float(np.max(component_abs)), 1e-300)
    phase = np.where(component_abs >= threshold, phase, np.nan)
    payload = {
        "total_magnitude": total_magnitude,
        "scattered_magnitude": scattered_magnitude,
        "scattered_component_abs": component_abs,
        "scattered_component_phase": phase,
    }
    if analytic_total is not None:
        analytic_scattered = analytic_total - incident
        payload["scattered_error_magnitude"] = np.sqrt(
            np.sum(np.abs(scattered - analytic_scattered) ** 2, axis=0)
        )
    return payload


def _stationary_slice_payload(
    total: np.ndarray,
    *,
    component: int,
    phase_mask_fraction: float,
) -> dict[str, np.ndarray]:
    component_values = total[component]
    component_abs = np.abs(component_values)
    phase = np.angle(component_values)
    threshold = phase_mask_fraction * max(float(np.max(component_abs)), 1e-300)
    phase = np.where(component_abs >= threshold, phase, np.nan)
    return {
        "total_magnitude": np.sqrt(np.sum(np.abs(total) ** 2, axis=0)),
        "component_abs": component_abs,
        "component_phase": phase,
    }


def _estimate_ensemble_circle(
    definition: SpectralCaseDefinition,
    *,
    levels: tuple[int, ...],
) -> tuple[CircleLocalization | None, list[dict[str, Any]], str | None]:
    spectra: dict[int, GridSpectrumRun] = {}
    rows: list[dict[str, Any]] = []
    try:
        for level in levels:
            built = build_spectral_case(
                definition,
                grid_shape=level,
                backend=em3d.Backend.numpy(),
                sampling_mode=SamplingMode.CELL_CENTER,
            )
            run = compute_grid_spectrum(built)
            spectra[int(level)] = run
            circle = run.localization.circle
            rows.append(
                {
                    "case_key": definition.key,
                    "parameter_strategy": "coarse_ensemble",
                    "coarse_level": int(level),
                    "localization_status": run.localization.status.value,
                    "mu_real": float(np.real(circle.mu)) if circle else np.nan,
                    "mu_imag": float(np.imag(circle.mu)) if circle else np.nan,
                    "radius": float(circle.radius) if circle else np.nan,
                    "q": float(circle.q) if circle else np.nan,
                    "margin": float(circle.margin) if circle else np.nan,
                }
            )
        ensemble = build_ensemble_parameter(spectra, levels)
        circle = ensemble.circle
        rows.append(
            {
                "case_key": definition.key,
                "parameter_strategy": "coarse_ensemble_result",
                "coarse_level": "+".join(str(v) for v in levels),
                "localization_status": ensemble.ensemble.localization.status.value,
                "mu_real": float(np.real(circle.mu)) if circle else np.nan,
                "mu_imag": float(np.imag(circle.mu)) if circle else np.nan,
                "radius": float(circle.radius) if circle else np.nan,
                "q": float(circle.q) if circle else np.nan,
                "margin": float(circle.margin) if circle else np.nan,
                "delta_latest": float(ensemble.delta_latest),
                "delta_max": float(ensemble.delta_max),
                "q_inflated_latest": float(ensemble.q_inflated_latest),
                "q_inflated_max": float(ensemble.q_inflated_max),
                "inflated_latest_safe": bool(ensemble.inflated_latest_safe),
                "inflated_max_safe": bool(ensemble.inflated_max_safe),
            }
        )
        if circle is None:
            return None, rows, "ensemble localization has no admissible circle"
        return circle, rows, None
    except Exception as exc:
        return None, rows, f"{type(exc).__name__}: {exc}"


def _run_mie_nearfield_gate(store: ArtifactStore) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xyz = np.array(
        [[0.1, 0.0, 0.2], [-0.2, 0.05, 0.3], [0.3, -0.1, -0.2]],
        dtype=float,
    )
    total = em3d.mie.mie_field_at(xyz, 0.25, 1.0 + 0.0j, 2.0)
    incident = np.zeros_like(total)
    incident[:, 0] = np.exp(1j * 2.0 * xyz[:, 2])
    error = _relative_l2(total, incident)
    rows.append(
        {
            "gate": "eps_r_equal_one",
            "plane": "all",
            "relative_l2": error,
            "threshold": 1e-12,
            "passed": bool(error <= 1e-12),
        }
    )
    for plane in ("xy", "xz", "yz"):
        result = em3d.mie.mie_nearfield_farfield_consistency(
            0.25,
            1.5 + 0.0j,
            2.0,
            plane=plane,
            n_phi=180,
            observation_radius=250.0,
        )
        value = float(result["relative_l2"])
        rows.append(
            {
                "gate": "nearfield_to_farfield",
                "plane": plane,
                "relative_l2": value,
                "threshold": 5e-4,
                "peak_ratio": float(result["peak_ratio"]),
                "passed": bool(value <= 5e-4),
            }
        )
        store.write_npz(
            f"raw/mie_gate/nearfield_farfield_{plane}.npz",
            phi=result["phi"],
            sigma_nearfield=result["sigma_nearfield"],
            sigma_reference=result["sigma_reference"],
        )
    store.write_rows("tables/mie_nearfield_gate.csv", rows)
    return rows


def _adaptive_farfield_batch(grid_size: int, *, precision: str) -> int:
    itemsize = 8 if precision == "single" else 16
    cells = int(grid_size) ** 3
    target = 192 * 1024**2
    return max(1, min(64, target // max(cells * itemsize, 1)))


def _save_figure(
    store: ArtifactStore,
    figure,
    *,
    relative_stem: str,
    config: LargeGridStudyConfig,
    figure_rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    paths = store.save_figure(
        relative_stem,
        figure,
        formats=config.visualization.save_formats,
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    for path in paths:
        figure_rows.append(
            {
                **dict(metadata),
                "format": path.suffix.lstrip("."),
                "path": str(path.relative_to(store.root)),
            }
        )


def _archive_subtree(store: ArtifactStore, relative_dir: str, archive_name: str) -> Path | None:
    source = store.root / relative_dir
    if not source.is_dir() or not any(source.rglob("*")):
        return None
    archives = store.root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    destination = archives / archive_name
    base = destination.with_suffix("")
    generated = Path(
        shutil.make_archive(
            str(base),
            "zip",
            root_dir=source.parent,
            base_dir=source.name,
        )
    )
    store.record_existing(generated.relative_to(store.root), kind="zip")
    return generated


def _observed_order_rows(rows: Sequence[Mapping[str, Any]], metric: str):
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("eps_real"),
            row.get("eps_imag"),
            row.get("k0a"),
            row.get("solver_name"),
            row.get("plane"),
            row.get("radius_label"),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in grouped.items():
        values = sorted(values, key=lambda item: int(item["grid_size"]))
        for lower, upper in zip(values, values[1:]):
            e1 = float(lower.get(metric, np.nan))
            e2 = float(upper.get(metric, np.nan))
            n1, n2 = int(lower["grid_size"]), int(upper["grid_size"])
            if not (np.isfinite(e1) and np.isfinite(e2) and e1 > 0.0 and e2 > 0.0):
                order = np.nan
            else:
                order = math.log(e1 / e2) / math.log(n2 / n1)
            output.append(
                {
                    "eps_real": key[0],
                    "eps_imag": key[1],
                    "k0a": key[2],
                    "solver_name": key[3],
                    "plane": key[4],
                    "radius_label": key[5],
                    "metric": metric,
                    "grid_size_lower": n1,
                    "grid_size_upper": n2,
                    "error_lower": e1,
                    "error_upper": e2,
                    "observed_order": order,
                }
            )
    return output


def _build_or_reuse_kernel(
    built: BuiltSpectralCase,
    *,
    config: LargeGridStudyConfig,
    cache: dict[str, Any],
    memory_rows: list[dict[str, Any]],
    job_key: str,
) -> PreparedEMKernel:
    backend = built.problem.backend
    key = (
        tuple(built.problem.grid.N),
        tuple(float(v) for v in built.problem.grid.L),
        float(built.problem.k0),
        backend.device,
        backend.precision.value,
        config.memory_policy.adjoint_storage,
        config.memory_policy.kernel_build_strategy,
    )
    if cache.get("key") == key and cache.get("kernel") is not None:
        checkpoint = memory_checkpoint(backend, "reused_kernel", started_at=perf_counter())
        memory_rows.append({"job_key": job_key, **checkpoint.to_row()})
        return cache["kernel"]

    if cache.get("kernel") is not None:
        cache.clear()
        gc.collect()
        clear_cuda_runtime_caches(backend)

    prepared, checkpoints = run_operator_memory_probe(
        built,
        policy=config.memory_policy,
        include_adjoint=True,
    )
    cache["key"] = key
    cache["kernel"] = prepared
    for checkpoint in checkpoints:
        memory_rows.append({"job_key": job_key, **checkpoint.to_row()})
    return prepared


def _process_mie_job(
    job: MieJobSpec,
    *,
    config: LargeGridStudyConfig,
    backend: em3d.Backend,
    store: ArtifactStore,
    circle_cache: dict[str, tuple[CircleLocalization | None, str | None]],
    kernel_cache: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    figure_rows: list[dict[str, Any]],
) -> None:
    started = perf_counter()
    definition = mie_sphere_case(
        eps_r=job.eps_r,
        k0a=job.k0a,
        radius=config.radius,
        domain_length=config.domain_length,
    )
    extra = {
        "eps_real": float(np.real(job.eps_r)),
        "eps_imag": float(np.imag(job.eps_r)),
        "k0a": float(job.k0a),
        "nominal_radius": float(config.radius),
    }
    _progress(
        config,
        f"[Mie] start {job.key}: eps={job.eps_r}, k0a={job.k0a:g}, N={job.grid_size}",
    )

    if definition.key not in circle_cache:
        circle, parameter_rows, error = _estimate_ensemble_circle(
            definition,
            levels=config.solver_suite.sim_coarse_sizes,
        )
        circle_cache[definition.key] = (circle, error)
        for row in parameter_rows:
            row.update(extra)
            tables["mie_sim_parameters"].append(row)
    circle, circle_error = circle_cache[definition.key]

    built = build_spectral_case(
        definition,
        grid_shape=job.grid_size,
        backend=backend,
        sampling_mode=SamplingMode.CELL_CENTER,
    )
    estimate = estimate_large_grid_memory(
        job.grid_size,
        precision=config.runtime.precision,
        include_adjoint=True,
        adjoint_storage=config.memory_policy.adjoint_storage,
        rcs_batch_size=_adaptive_farfield_batch(
            job.grid_size, precision=config.runtime.precision
        ),
    )
    decision = memory_decision(estimate, backend, config.memory_policy)
    tables["gpu_memory_probe"].append(
        {
            "job_key": job.key,
            **estimate.to_row(),
            **{f"decision_{key}": value for key, value in decision.to_row().items()},
        }
    )
    if not decision.allowed:
        for solver_name in config.solver_suite.solver_names:
            tables["mie_solver_runs"].append(
                _missing_solver_row(
                    solver_name=solver_name,
                    status="skipped_memory",
                    job_key=job.key,
                    case_key=definition.key,
                    case_title=definition.title,
                    grid_size=job.grid_size,
                    tier=job.tier,
                    study="mie",
                    requested_rtol=job.true_rtol,
                    error=decision.reason,
                    extra=extra,
                )
            )
        tables["job_status"].append(
            {
                "job_key": job.key,
                "study": "mie",
                "status": "skipped_memory",
                "error": decision.reason,
                "elapsed_seconds": perf_counter() - started,
                **extra,
                "grid_size": job.grid_size,
            }
        )
        return

    try:
        prepared = _build_or_reuse_kernel(
            built,
            config=config,
            cache=kernel_cache,
            memory_rows=tables["gpu_memory_checkpoints"],
            job_key=job.key,
        )
        operator = Operator(built.problem, prepared_kernel=prepared)
        executions: list[SolverExecution] = []
        curves: list[RCSCurve] = []
        batch_size = _adaptive_farfield_batch(
            job.grid_size, precision=config.runtime.precision
        )

        for solver_name in config.solver_suite.solver_names:
            if solver_name == "SIM" and circle is None:
                tables["mie_solver_runs"].append(
                    _missing_solver_row(
                        solver_name="SIM",
                        status="no_parameter",
                        job_key=job.key,
                        case_key=definition.key,
                        case_title=definition.title,
                        grid_size=job.grid_size,
                        tier=job.tier,
                        study="mie",
                        requested_rtol=job.true_rtol,
                        error=circle_error or "no admissible ensemble circle",
                        extra=extra,
                    )
                )
                continue
            execution = run_solver_execution(
                built,
                solver_name=solver_name,
                solver_config=_solver_config(
                    config, solver_name, rtol=job.true_rtol
                ),
                circle=circle,
                operator=operator,
                qualification_rtol=job.true_rtol,
                retain_solution=True,
            )
            executions.append(execution)
            tables["mie_solver_runs"].append(
                _solver_row(
                    execution,
                    job_key=job.key,
                    case_title=definition.title,
                    grid_size=job.grid_size,
                    tier=job.tier,
                    requested_rtol=job.true_rtol,
                    circle=circle,
                    study="mie",
                    extra=extra,
                )
            )
            store.write_npz(
                f"raw/residual_histories/{job.key}_{solver_name}.npz",
                residual=np.asarray(execution.result.residual_history, dtype=float),
                operator_actions=np.asarray(
                    execution.result.residual_action_counts, dtype=int
                ),
            )
            if execution.qualified:
                for plane in config.visualization.rcs_planes:
                    curve = compute_rcs_curve(
                        execution,
                        built,
                        n_phi=config.rcs_n_phi,
                        plane=plane,
                        method="direct",
                        batch_size=batch_size,
                    )
                    curves.append(curve)
                    tables["mie_rcs_by_plane"].append(
                        {
                            "job_key": job.key,
                            "case_key": definition.key,
                            "grid_size": job.grid_size,
                            "solver_name": solver_name,
                            "plane": plane,
                            "peak": curve.peak,
                            "angular_integral": curve.angular_integral,
                            **extra,
                        }
                    )
                    store.write_npz(
                        f"raw/rcs/mie/{job.key}_{solver_name}_{plane}.npz",
                        phi=curve.phi,
                        sigma=curve.sigma,
                        sigma_normalized=curve.sigma_normalized,
                    )
            # The host copy is retained for pairwise and Mie field metrics.  The
            # device solution is no longer needed after RCS post-processing.
            execution.result.u = None
            gc.collect()
            tables["gpu_memory_checkpoints"].append(
                {
                    "job_key": job.key,
                    **memory_checkpoint(
                        backend,
                        f"after_solver_{solver_name}",
                        started_at=started,
                    ).to_row(),
                }
            )

        tables["mie_solver_pairwise_fields"].extend(
            {
                **row,
                **extra,
            }
            for row in _pairwise_field_rows(job.key, job.grid_size, executions)
        )
        tables["mie_solver_pairwise_rcs"].extend(
            {
                **row,
                **extra,
            }
            for row in _pairwise_rcs_rows(job.key, job.grid_size, curves)
        )

        execution_by_name = {item.solver_name: item for item in executions}
        status_by_name = {
            name: (
                "no_parameter"
                if name == "SIM" and circle is None
                else execution_by_name[name].result.status
                if name in execution_by_name
                else "not_run"
            )
            for name in config.solver_suite.solver_names
        }
        qualified_by_name = {
            name: bool(execution_by_name.get(name) and execution_by_name[name].qualified)
            for name in config.solver_suite.solver_names
        }
        tables["mie_common_solvability"].append(
            {
                "job_key": job.key,
                "case_key": definition.key,
                "grid_size": job.grid_size,
                **extra,
                **{f"{name.lower()}_status": status_by_name[name] for name in status_by_name},
                **{f"{name.lower()}_qualified": qualified_by_name[name] for name in qualified_by_name},
                "all_solvers_qualified": all(qualified_by_name.values()),
            }
        )

        represented_volume = float(built.geometry.represented_volume)
        effective_radius = float((3.0 * represented_volume / (4.0 * np.pi)) ** (1.0 / 3.0))
        nominal_radius = float(config.radius)
        tables["mie_geometry"].append(
            {
                "job_key": job.key,
                "grid_size": job.grid_size,
                "nominal_radius": nominal_radius,
                "effective_radius": effective_radius,
                "represented_volume": represented_volume,
                "exact_volume": float(built.geometry.exact_volume),
                "relative_volume_error": float(built.geometry.relative_volume_error),
                **extra,
            }
        )

        qualified_executions = [
            execution
            for execution in executions
            if execution.qualified and execution.solution_host is not None
        ]
        incident_host = np.asarray(
            backend.to_host(built.problem.wave), dtype=np.complex128
        )
        analytic_nominal = analytic_effective = None
        nominal_mask = effective_mask = None
        if job.compute_full_field_metrics and qualified_executions:
            analytic_nominal = em3d.mie.mie_field(
                built.problem.grid,
                a=nominal_radius,
                eps_r=job.eps_r,
                k0=built.problem.k0,
                amplitude=definition.wave_amplitude,
                orient=definition.wave_direction,
            )
            analytic_effective = em3d.mie.mie_field(
                built.problem.grid,
                a=effective_radius,
                eps_r=job.eps_r,
                k0=built.problem.k0,
                amplitude=definition.wave_amplitude,
                orient=definition.wave_direction,
            )
            x = np.asarray(backend.to_host(built.problem.grid.x))
            y = np.asarray(backend.to_host(built.problem.grid.y))
            z = np.asarray(backend.to_host(built.problem.grid.z))
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            nominal_mask = X**2 + Y**2 + Z**2 <= nominal_radius**2
            effective_mask = X**2 + Y**2 + Z**2 <= effective_radius**2
            for execution in qualified_executions:
                tables["mie_solver_vs_mie_fields"].extend(
                    _field_reference_rows(
                        job=job,
                        execution=execution,
                        numerical=np.asarray(execution.solution_host),
                        incident=incident_host,
                        analytic_nominal=analytic_nominal,
                        analytic_effective=analytic_effective,
                        nominal_mask=nominal_mask,
                        effective_mask=effective_mask,
                        nominal_radius=nominal_radius,
                        effective_radius=effective_radius,
                    )
                )

        curves_by_plane: dict[str, dict[str, RCSCurve]] = {}
        for curve in curves:
            curves_by_plane.setdefault(curve.plane, {})[curve.solver_name] = curve
        for plane in config.visualization.rcs_planes:
            nominal_curve = _analytic_rcs_curve(
                nominal_radius,
                job.eps_r,
                built.problem.k0,
                n_phi=config.rcs_n_phi,
                plane=plane,
                label="Mie-nominal",
            )
            effective_curve = _analytic_rcs_curve(
                effective_radius,
                job.eps_r,
                built.problem.k0,
                n_phi=config.rcs_n_phi,
                plane=plane,
                label="Mie-effective",
            )
            for curve in curves_by_plane.get(plane, {}).values():
                tables["mie_solver_vs_mie_rcs"].append(
                    _rcs_reference_row(
                        job_key=job.key,
                        case_key=definition.key,
                        grid_size=job.grid_size,
                        eps_r=job.eps_r,
                        k0a=job.k0a,
                        solver_curve=curve,
                        reference_curve=nominal_curve,
                        radius_label="nominal",
                    )
                )
                tables["mie_solver_vs_mie_rcs"].append(
                    _rcs_reference_row(
                        job_key=job.key,
                        case_key=definition.key,
                        grid_size=job.grid_size,
                        eps_r=job.eps_r,
                        k0a=job.k0a,
                        solver_curve=curve,
                        reference_curve=effective_curve,
                        radius_label="effective",
                    )
                )
            if job.render_rcs and curves_by_plane.get(plane):
                figure = plot_rcs_four_panel(
                    phi=nominal_curve.phi,
                    numerical_curves={
                        name: curve.sigma
                        for name, curve in curves_by_plane[plane].items()
                    },
                    nominal=nominal_curve.sigma,
                    effective=effective_curve.sigma,
                    title=(
                        f"Задача Ми: eps={job.eps_r}, k0a={job.k0a:g}, "
                        f"N={job.grid_size}, плоскость {plane}"
                    ),
                )
                _save_figure(
                    store,
                    figure,
                    relative_stem=f"figures/mie_rcs/{job.key}_{plane}",
                    config=config,
                    figure_rows=figure_rows,
                    metadata={
                        "job_key": job.key,
                        "case_key": definition.key,
                        "grid_size": job.grid_size,
                        "plane": plane,
                        "quantity": "rcs_all_solvers_vs_mie",
                        "coordinate_system": "cartesian_and_polar",
                        "normalized": "absolute_and_normalized",
                    },
                )

        if job.render_field_slices and qualified_executions:
            for plane in config.visualization.field_planes:
                horizontal, vertical, xyz = _slice_coordinates(
                    built.problem.grid, plane
                )
                analytic_effective_flat = em3d.mie.mie_field_at(
                    xyz,
                    effective_radius,
                    job.eps_r,
                    built.problem.k0,
                    amplitude=definition.wave_amplitude,
                    orient=definition.wave_direction,
                )
                analytic_nominal_flat = em3d.mie.mie_field_at(
                    xyz,
                    nominal_radius,
                    job.eps_r,
                    built.problem.k0,
                    amplitude=definition.wave_amplitude,
                    orient=definition.wave_direction,
                )
                shape = (len(horizontal), len(vertical))
                analytic_effective_slice = analytic_effective_flat.T.reshape(3, *shape)
                analytic_nominal_slice = analytic_nominal_flat.T.reshape(3, *shape)
                incident_slice = _slice_field(incident_host, plane)
                analytic_payload = _slice_payload(
                    analytic_effective_slice,
                    incident_slice,
                    None,
                    component=config.visualization.field_component,
                    phase_mask_fraction=config.visualization.phase_mask_fraction,
                )
                solver_payloads: dict[str, dict[str, np.ndarray]] = {}
                raw_arrays: dict[str, np.ndarray] = {
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "analytic_effective_total": analytic_effective_slice,
                    "analytic_nominal_total": analytic_nominal_slice,
                    "incident": incident_slice,
                }
                for execution in qualified_executions:
                    numerical_slice = _slice_field(
                        np.asarray(execution.solution_host), plane
                    )
                    payload = _slice_payload(
                        numerical_slice,
                        incident_slice,
                        analytic_effective_slice,
                        component=config.visualization.field_component,
                        phase_mask_fraction=config.visualization.phase_mask_fraction,
                    )
                    solver_payloads[execution.solver_name] = payload
                    raw_arrays[f"{execution.solver_name}_total"] = numerical_slice
                store.write_npz(
                    f"raw/field_slices/mie/{job.key}_{plane}.npz",
                    **raw_arrays,
                )
                figure = plot_mie_field_panel(
                    horizontal=horizontal,
                    vertical=vertical,
                    solver_slices=solver_payloads,
                    analytic_slice=analytic_payload,
                    title=(
                        f"Поля задачи Ми: eps={job.eps_r}, k0a={job.k0a:g}, "
                        f"N={job.grid_size}, {plane}"
                    ),
                    plane=plane,
                )
                _save_figure(
                    store,
                    figure,
                    relative_stem=f"figures/mie_fields/{job.key}_{plane}_overview",
                    config=config,
                    figure_rows=figure_rows,
                    metadata={
                        "job_key": job.key,
                        "case_key": definition.key,
                        "grid_size": job.grid_size,
                        "plane": plane,
                        "quantity": "field_overview_all_solvers_vs_mie",
                        "coordinate_system": "cartesian_slice",
                        "normalized": False,
                    },
                )
                figure = plot_mie_error_phase_panel(
                    horizontal=horizontal,
                    vertical=vertical,
                    solver_slices=solver_payloads,
                    title=(
                        f"Фаза и ошибка рассеянного поля: eps={job.eps_r}, "
                        f"k0a={job.k0a:g}, N={job.grid_size}, {plane}"
                    ),
                    plane=plane,
                )
                _save_figure(
                    store,
                    figure,
                    relative_stem=f"figures/mie_fields/{job.key}_{plane}_phase_error",
                    config=config,
                    figure_rows=figure_rows,
                    metadata={
                        "job_key": job.key,
                        "case_key": definition.key,
                        "grid_size": job.grid_size,
                        "plane": plane,
                        "quantity": "scattered_field_phase_and_error",
                        "coordinate_system": "cartesian_slice",
                        "normalized": False,
                    },
                )

        figure = plot_solver_residuals(
            executions,
            title=(
                f"Сходимость методов, задача Ми: eps={job.eps_r}, "
                f"k0a={job.k0a:g}, N={job.grid_size}"
            ),
        )
        _save_figure(
            store,
            figure,
            relative_stem=f"figures/mie_residuals/{job.key}",
            config=config,
            figure_rows=figure_rows,
            metadata={
                "job_key": job.key,
                "case_key": definition.key,
                "grid_size": job.grid_size,
                "plane": "none",
                "quantity": "solver_residuals",
                "coordinate_system": "operator_actions",
                "normalized": True,
            },
        )

        tables["job_status"].append(
            {
                "job_key": job.key,
                "study": "mie",
                "status": "complete",
                "error": "",
                "elapsed_seconds": perf_counter() - started,
                "grid_size": job.grid_size,
                **extra,
            }
        )
        _progress(
            config,
            f"[Mie] finish {job.key}: qualified "
            f"{sum(item.qualified for item in executions)}/{len(config.solver_suite.solver_names)}",
        )
    except Exception as exc:
        # CUDA OOM and numerical failures are recorded per job.  The caller may
        # still choose to raise by configuring oom_behavior='raise'.
        error = f"{type(exc).__name__}: {exc}"
        is_oom = "out of memory" in error.lower() or "OutOfMemory" in type(exc).__name__
        if is_oom and config.memory_policy.oom_behavior == "skip_and_record":
            status = "oom"
        else:
            status = "failed"
        tables["job_status"].append(
            {
                "job_key": job.key,
                "study": "mie",
                "status": status,
                "error": error,
                "elapsed_seconds": perf_counter() - started,
                "grid_size": job.grid_size,
                **extra,
            }
        )
        if not is_oom or config.memory_policy.oom_behavior == "raise":
            raise
    finally:
        gc.collect()
        if config.memory_policy.clear_pool_before_case:
            backend.clear_memory_pool()


def _process_stationary_job(
    job: StationaryGridJobSpec,
    *,
    config: LargeGridStudyConfig,
    backend: em3d.Backend,
    store: ArtifactStore,
    circle_cache: dict[str, tuple[CircleLocalization | None, str | None]],
    tables: dict[str, list[dict[str, Any]]],
    figure_rows: list[dict[str, Any]],
) -> None:
    started = perf_counter()
    catalog = stationary_case_catalog()
    if job.case_key not in catalog:
        raise KeyError(f"unknown stationary case {job.case_key!r}")
    definition = catalog[job.case_key]
    if definition.key not in circle_cache:
        circle, parameter_rows, error = _estimate_ensemble_circle(
            definition,
            levels=config.solver_suite.sim_coarse_sizes,
        )
        circle_cache[definition.key] = (circle, error)
        for row in parameter_rows:
            row["case_key_requested"] = job.case_key
            tables["stationary_sim_parameters"].append(row)
    circle, circle_error = circle_cache[definition.key]

    built = build_spectral_case(
        definition,
        grid_shape=job.grid_size,
        backend=backend,
        sampling_mode=SamplingMode.CELL_CENTER,
    )
    estimate = estimate_large_grid_memory(
        job.grid_size,
        precision=config.runtime.precision,
        include_adjoint=True,
        adjoint_storage=config.memory_policy.adjoint_storage,
        rcs_batch_size=_adaptive_farfield_batch(job.grid_size, precision=config.runtime.precision),
    )
    decision = memory_decision(estimate, backend, config.memory_policy)
    tables["gpu_memory_probe"].append(
        {
            "job_key": job.key,
            **estimate.to_row(),
            **{f"decision_{key}": value for key, value in decision.to_row().items()},
        }
    )
    if not decision.allowed:
        for name in config.solver_suite.solver_names:
            tables["stationary_solver_runs"].append(
                _missing_solver_row(
                    solver_name=name,
                    status="skipped_memory",
                    job_key=job.key,
                    case_key=definition.key,
                    case_title=definition.title,
                    grid_size=job.grid_size,
                    tier=job.tier,
                    study="stationary",
                    requested_rtol=job.true_rtol,
                    error=decision.reason,
                )
            )
        tables["job_status"].append(
            {
                "job_key": job.key,
                "study": "stationary",
                "status": "skipped_memory",
                "error": decision.reason,
                "elapsed_seconds": perf_counter() - started,
                "grid_size": job.grid_size,
            }
        )
        return

    try:
        prepared, checkpoints = run_operator_memory_probe(
            built,
            policy=config.memory_policy,
            include_adjoint=True,
        )
        tables["gpu_memory_checkpoints"].extend(
            {"job_key": job.key, **item.to_row()} for item in checkpoints
        )
        operator = Operator(built.problem, prepared_kernel=prepared)
        executions: list[SolverExecution] = []
        curves: list[RCSCurve] = []
        batch_size = _adaptive_farfield_batch(job.grid_size, precision=config.runtime.precision)
        for solver_name in config.solver_suite.solver_names:
            if solver_name == "SIM" and circle is None:
                tables["stationary_solver_runs"].append(
                    _missing_solver_row(
                        solver_name="SIM",
                        status="no_parameter",
                        job_key=job.key,
                        case_key=definition.key,
                        case_title=definition.title,
                        grid_size=job.grid_size,
                        tier=job.tier,
                        study="stationary",
                        requested_rtol=job.true_rtol,
                        error=circle_error or "no admissible ensemble circle",
                    )
                )
                continue
            execution = run_solver_execution(
                built,
                solver_name=solver_name,
                solver_config=_solver_config(config, solver_name, rtol=job.true_rtol),
                circle=circle,
                operator=operator,
                qualification_rtol=job.true_rtol,
                retain_solution=True,
            )
            executions.append(execution)
            tables["stationary_solver_runs"].append(
                _solver_row(
                    execution,
                    job_key=job.key,
                    case_title=definition.title,
                    grid_size=job.grid_size,
                    tier=job.tier,
                    requested_rtol=job.true_rtol,
                    circle=circle,
                    study="stationary",
                )
            )
            store.write_npz(
                f"raw/residual_histories/{job.key}_{solver_name}.npz",
                residual=np.asarray(execution.result.residual_history, dtype=float),
                operator_actions=np.asarray(execution.result.residual_action_counts, dtype=int),
            )
            if execution.qualified:
                for plane in config.visualization.rcs_planes:
                    curve = compute_rcs_curve(
                        execution,
                        built,
                        n_phi=config.rcs_n_phi,
                        plane=plane,
                        method="direct",
                        batch_size=batch_size,
                    )
                    curves.append(curve)
                    store.write_npz(
                        f"raw/rcs/stationary/{job.key}_{solver_name}_{plane}.npz",
                        phi=curve.phi,
                        sigma=curve.sigma,
                        sigma_normalized=curve.sigma_normalized,
                    )
            execution.result.u = None
            gc.collect()

        tables["stationary_solver_pairwise_fields"].extend(
            _pairwise_field_rows(job.key, job.grid_size, executions)
        )
        tables["stationary_solver_pairwise_rcs"].extend(
            _pairwise_rcs_rows(job.key, job.grid_size, curves)
        )
        qualified = [
            item for item in executions if item.qualified and item.solution_host is not None
        ]
        if job.render_field_slices and qualified:
            for plane in config.visualization.field_planes:
                horizontal, vertical, _xyz = _slice_coordinates(built.problem.grid, plane)
                payloads: dict[str, dict[str, np.ndarray]] = {}
                raw_arrays: dict[str, np.ndarray] = {
                    "horizontal": horizontal,
                    "vertical": vertical,
                }
                for execution in qualified:
                    field_slice = _slice_field(np.asarray(execution.solution_host), plane)
                    payloads[execution.solver_name] = _stationary_slice_payload(
                        field_slice,
                        component=config.visualization.field_component,
                        phase_mask_fraction=config.visualization.phase_mask_fraction,
                    )
                    raw_arrays[f"{execution.solver_name}_total"] = field_slice
                store.write_npz(
                    f"raw/field_slices/stationary/{job.key}_{plane}.npz",
                    **raw_arrays,
                )
                figure = plot_stationary_field_panel(
                    horizontal=horizontal,
                    vertical=vertical,
                    solver_slices=payloads,
                    title=f"{definition.title}, N={job.grid_size}, {plane}",
                    plane=plane,
                )
                _save_figure(
                    store,
                    figure,
                    relative_stem=f"figures/stationary_fields/{job.key}_{plane}",
                    config=config,
                    figure_rows=figure_rows,
                    metadata={
                        "job_key": job.key,
                        "case_key": definition.key,
                        "grid_size": job.grid_size,
                        "plane": plane,
                        "quantity": "field_all_solvers",
                        "coordinate_system": "cartesian_slice",
                        "normalized": False,
                    },
                )
        curves_by_plane: dict[str, dict[str, RCSCurve]] = {}
        for curve in curves:
            curves_by_plane.setdefault(curve.plane, {})[curve.solver_name] = curve
        for plane, values in curves_by_plane.items():
            if job.render_rcs:
                first = next(iter(values.values()))
                figure = plot_rcs_four_panel(
                    phi=first.phi,
                    numerical_curves={name: curve.sigma for name, curve in values.items()},
                    title=f"{definition.title}, N={job.grid_size}, {plane}",
                )
                _save_figure(
                    store,
                    figure,
                    relative_stem=f"figures/stationary_rcs/{job.key}_{plane}",
                    config=config,
                    figure_rows=figure_rows,
                    metadata={
                        "job_key": job.key,
                        "case_key": definition.key,
                        "grid_size": job.grid_size,
                        "plane": plane,
                        "quantity": "rcs_all_solvers",
                        "coordinate_system": "cartesian_and_polar",
                        "normalized": "absolute_and_normalized",
                    },
                )
        figure = plot_solver_residuals(
            executions,
            title=f"Сходимость методов: {definition.title}, N={job.grid_size}",
        )
        _save_figure(
            store,
            figure,
            relative_stem=f"figures/stationary_residuals/{job.key}",
            config=config,
            figure_rows=figure_rows,
            metadata={
                "job_key": job.key,
                "case_key": definition.key,
                "grid_size": job.grid_size,
                "plane": "none",
                "quantity": "solver_residuals",
                "coordinate_system": "operator_actions",
                "normalized": True,
            },
        )
        tables["job_status"].append(
            {
                "job_key": job.key,
                "study": "stationary",
                "status": "complete",
                "error": "",
                "elapsed_seconds": perf_counter() - started,
                "grid_size": job.grid_size,
            }
        )
    finally:
        gc.collect()
        backend.clear_memory_pool()


def _initial_tables() -> dict[str, list[dict[str, Any]]]:
    names = (
        "job_status",
        "gpu_memory_probe",
        "gpu_memory_checkpoints",
        "mie_sim_parameters",
        "mie_solver_runs",
        "mie_common_solvability",
        "mie_geometry",
        "mie_solver_pairwise_fields",
        "mie_solver_pairwise_rcs",
        "mie_solver_vs_mie_fields",
        "mie_solver_vs_mie_rcs",
        "mie_rcs_by_plane",
        "mie_grid_convergence_by_solver",
        "mie_observed_orders_by_solver",
        "stationary_sim_parameters",
        "stationary_solver_runs",
        "stationary_solver_pairwise_fields",
        "stationary_solver_pairwise_rcs",
        "stationary_grid_convergence",
        "figure_index",
    )
    return {name: [] for name in names}


def _write_tables(store: ArtifactStore, tables: dict[str, list[dict[str, Any]]]) -> None:
    # The two convergence tables are aliases of the detailed reference rows in
    # a single-run profile.  Merge mode later combines them across sessions.
    tables["mie_grid_convergence_by_solver"] = list(tables["mie_solver_vs_mie_rcs"])
    observed: list[dict[str, Any]] = []
    for metric in ("normalized_l2", "absolute_l2", "absolute_linf"):
        observed.extend(_observed_order_rows(tables["mie_solver_vs_mie_rcs"], metric))
    tables["mie_observed_orders_by_solver"] = observed
    for name, rows in tables.items():
        store.write_rows(f"tables/{name}.csv", rows)
        store.write_json(f"raw/tables/{name}.json", rows)


def _create_archives(store: ArtifactStore) -> None:
    for relative_dir, archive_name in (
        ("figures/mie_fields", "mie_field_figures.zip"),
        ("figures/mie_rcs", "mie_rcs_figures.zip"),
        ("figures/stationary_fields", "stationary_field_figures.zip"),
        ("figures/stationary_rcs", "stationary_rcs_figures.zip"),
    ):
        _archive_subtree(store, relative_dir, archive_name)
    selected = store.root / "publication_selected"
    selected.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "figures/mie_rcs/*.pdf",
        "figures/mie_fields/*overview.pdf",
        "figures/stationary_rcs/*.pdf",
        "figures/stationary_fields/*.pdf",
    ):
        for path in store.root.glob(pattern):
            target = selected / path.name
            if not target.exists():
                shutil.copy2(path, target)
    _archive_subtree(store, "publication_selected", "publication_selected_figures.zip")


def run_large_grid_suite(config: LargeGridStudyConfig) -> LargeGridSuiteResult:
    """Run the all-solver Chapter 4 validation suite.

    Every Mie job attempts SIM, BiCGStab and TwoStep under one true-residual
    qualification protocol.  A missing SIM circle or a solver failure is a
    recorded scientific outcome rather than a reason to discard the job.
    """

    store = ArtifactStore(
        config.output_root,
        schema="em3d-chapter4-all-solvers-large-grid-v1",
    )
    tables = _initial_tables()
    figure_rows = tables["figure_index"]
    backend = _backend(config)
    store.write_json("config.json", config)

    if config.require_mie_nearfield_gate:
        gate_rows = _run_mie_nearfield_gate(store)
        if not all(bool(row["passed"]) for row in gate_rows):
            store.finalize(config=config, status="failed_mie_nearfield_gate")
            raise RuntimeError("Mie near-field consistency gate failed")

    circle_cache: dict[str, tuple[CircleLocalization | None, str | None]] = {}
    kernel_cache: dict[str, Any] = {}

    # Ordering by N and k0 maximises reuse of the expensive FFT kernel across
    # the three permittivity values of the main tensor-product series.
    mie_jobs = sorted(
        config.mie_jobs,
        key=lambda job: (job.grid_size, job.k0a, float(np.real(job.eps_r)), float(np.imag(job.eps_r))),
    )
    for job in mie_jobs:
        _process_mie_job(
            job,
            config=config,
            backend=backend,
            store=store,
            circle_cache=circle_cache,
            kernel_cache=kernel_cache,
            tables=tables,
            figure_rows=figure_rows,
        )
    kernel_cache.clear()
    gc.collect()
    clear_cuda_runtime_caches(backend)

    stationary_circle_cache: dict[str, tuple[CircleLocalization | None, str | None]] = {}
    for job in config.stationary_jobs:
        _progress(config, f"[stationary] start {job.key}")
        _process_stationary_job(
            job,
            config=config,
            backend=backend,
            store=store,
            circle_cache=stationary_circle_cache,
            tables=tables,
            figure_rows=figure_rows,
        )
        _progress(config, f"[stationary] finish {job.key}")

    # Summary convergence plots are useful whenever at least two grid levels
    # are present in one batch.
    if tables["mie_solver_vs_mie_rcs"]:
        grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in tables["mie_solver_vs_mie_rcs"]:
            if row.get("plane") == "xz" and row.get("radius_label") == "effective":
                grouped.setdefault((float(row["eps_real"]), float(row["k0a"])), []).append(row)
        for (eps_real, k0a), rows in grouped.items():
            if len({int(row["grid_size"]) for row in rows}) < 2:
                continue
            figure = plot_grid_metric(
                rows,
                metric="normalized_l2",
                title=f"Сеточная сходимость ЭПР: eps={eps_real:g}, k0a={k0a:g}",
                ylabel="Относительная ошибка формы ЭПР",
            )
            token = f"eps{eps_real:g}_k0a{k0a:g}".replace(".", "p")
            _save_figure(
                store,
                figure,
                relative_stem=f"figures/mie_convergence/{token}",
                config=config,
                figure_rows=figure_rows,
                metadata={
                    "job_key": token,
                    "case_key": "mie_grid_convergence",
                    "grid_size": "multiple",
                    "plane": "xz",
                    "quantity": "rcs_grid_convergence_by_solver",
                    "coordinate_system": "loglog",
                    "normalized": True,
                },
            )

    _write_tables(store, tables)
    if config.visualization.create_subtree_archives:
        _create_archives(store)
    manifest = store.finalize(config=config, status="complete")
    return LargeGridSuiteResult(
        profile=config.profile,
        output_root=str(store.root),
        manifest_path=str(manifest),
        job_rows=tuple(tables["job_status"]),
        solver_rows=tuple(tables["mie_solver_runs"] + tables["stationary_solver_runs"]),
        common_solvability_rows=tuple(tables["mie_common_solvability"]),
    )


__all__ = ["LargeGridSuiteResult", "run_large_grid_suite"]
