from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np

from em3d.experiments.spectral_transfer import (
    FineTransferRun,
    GridSpectrumRun,
    SpectralCaseDefinition,
    build_spectral_case,
    compute_grid_spectrum,
    run_parameter_transfer,
)
from em3d.geometry import FullDomain
from em3d.operator import PreparedEMKernel
from em3d.solvers import SolverConfig

from .artifacts import ArtifactStore
from .common import make_backend, write_spectrum
from .config import FixedSpectrumScanConfig, SpectralStudyConfig


@dataclass(frozen=True)
class FixedSpectrumPoint:
    material_key: str
    material_title: str
    eps_r_diagonal: tuple[complex, complex, complex]
    k0: float
    spectrum: GridSpectrumRun
    iteration: FineTransferRun | None


@dataclass(frozen=True)
class FixedSpectrumStudyResult:
    points: tuple[FixedSpectrumPoint, ...]
    output_root: str

    def for_material(self, material_key: str) -> tuple[FixedSpectrumPoint, ...]:
        return tuple(point for point in self.points if point.material_key == material_key)


def _definitions(
    scan: FixedSpectrumScanConfig,
) -> tuple[tuple[str, str, np.ndarray, tuple[float, ...]], ...]:
    """Return the two material families retained from dissertation Section 4.2.

    The historical text contains an inconsistent statement ``N=8`` and
    ``N_Q=216``.  The final stand uses ``grid_size=6`` by default, matching
    ``N_Q=6^3=216`` and records the chosen value explicitly in every table.
    """

    # The previous text contains diag(2,1,0), while the lossy comparison
    # uses real parts diag(2,1,3) and reports a valid SIM circle.  The final
    # controlled experiment keeps the real parts identical and varies only
    # the loss term; this removes the degenerate zero-permittivity direction.
    real_tensor = np.diag([2.0, 1.0, 3.0]).astype(np.complex128)
    lossy_tensor = np.diag([2.0 + 1.0j, 1.0 + 1.0j, 3.0 + 1.0j]).astype(
        np.complex128
    )
    return (
        (
            "real_anisotropy",
            "Вещественная анизотропия diag(2,1,3)",
            real_tensor,
            scan.real_wave_numbers,
        ),
        (
            "lossy_anisotropy",
            "Комплексная анизотропия diag(2+i,1+i,3+i)",
            lossy_tensor,
            scan.lossy_wave_numbers,
        ),
    )


def _definition(
    *,
    material_key: str,
    material_title: str,
    tensor: np.ndarray,
    k0: float,
) -> SpectralCaseDefinition:
    return SpectralCaseDefinition(
        key=f"chapter42_{material_key}_k{float(k0):g}",
        title=f"{material_title}, k0={float(k0):g}",
        domain_lengths=(1.0, 1.0, 1.0),
        domain_center=(0.0, 0.0, 0.0),
        k0=float(k0),
        background_eps_r=np.eye(3, dtype=np.complex128),
        feature_eps_r=np.asarray(tensor, dtype=np.complex128),
        feature_geometry=FullDomain(),
        wave_direction=(0.0, -1.0, 0.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )


def _point_row(point: FixedSpectrumPoint) -> dict[str, object]:
    localization = point.spectrum.localization
    circle = localization.circle
    run = point.iteration
    return {
        "material_key": point.material_key,
        "material_title": point.material_title,
        "eps11_real": float(point.eps_r_diagonal[0].real),
        "eps11_imag": float(point.eps_r_diagonal[0].imag),
        "eps22_real": float(point.eps_r_diagonal[1].real),
        "eps22_imag": float(point.eps_r_diagonal[1].imag),
        "eps33_real": float(point.eps_r_diagonal[2].real),
        "eps33_imag": float(point.eps_r_diagonal[2].imag),
        "k0": point.k0,
        "N": point.spectrum.grid_shape[0],
        "cells": int(np.prod(point.spectrum.grid_shape)),
        "matrix_dimension": point.spectrum.matrix_dimension,
        "localization_status": localization.status.value,
        "origin_in_hull": localization.origin_in_hull,
        "origin_distance": localization.origin_distance,
        "mu_real": float(circle.mu.real) if circle is not None else np.nan,
        "mu_imag": float(circle.mu.imag) if circle is not None else np.nan,
        "radius": float(circle.radius) if circle is not None else np.nan,
        "q": float(circle.q) if circle is not None else np.nan,
        "margin": float(circle.margin) if circle is not None else np.nan,
        "matrix_build_seconds": point.spectrum.matrix_build_seconds,
        "eigenvalue_seconds_median": point.spectrum.median_eigenvalue_seconds,
        "sim_status": run.solver_result.status if run is not None else "no_parameter",
        "sim_converged": bool(run.solver_result.converged) if run is not None else False,
        "sim_iterations": int(run.solver_result.iterations) if run is not None else 0,
        "sim_operator_actions": int(run.solver_result.matvec_count)
        if run is not None
        else 0,
        "sim_final_residual": (
            float(run.solver_result.true_final_residual)
            if run is not None and run.solver_result.true_final_residual is not None
            else (
                float(run.solver_result.residual_history[-1])
                if run is not None and run.solver_result.residual_history
                else np.nan
            )
        ),
        "sim_observed_ratio": float(run.convergence.asymptotic_ratio)
        if run is not None
        else np.nan,
        "sim_classification": run.convergence.classification.value
        if run is not None
        else "no_parameter",
        "sim_elapsed_seconds": float(run.elapsed_seconds) if run is not None else np.nan,
    }


def run_fixed_spectrum_study(
    config: SpectralStudyConfig,
    *,
    store: ArtifactStore,
    render_figures: bool = False,
) -> FixedSpectrumStudyResult:
    """Reproduce and strengthen the fixed-grid spectral scans of Section 4.2."""

    scan = config.fixed_spectrum
    backend = make_backend(config)
    points: list[FixedSpectrumPoint] = []
    rows: list[dict[str, object]] = []

    store.log_event(
        "study_start",
        study="chapter42_fixed_spectrum",
        grid_size=scan.grid_size,
    )

    for material_key, material_title, tensor, wave_numbers in _definitions(scan):
        for k0 in wave_numbers:
            definition = _definition(
                material_key=material_key,
                material_title=material_title,
                tensor=tensor,
                k0=k0,
            )
            built = build_spectral_case(
                definition,
                grid_shape=scan.grid_size,
                backend=backend,
            )
            spectrum = compute_grid_spectrum(
                built,
                eigenvalue_repeats=config.eigenvalue_repeats,
            )
            write_spectrum(
                store,
                f"raw/chapter42/spectra/{material_key}_k{float(k0):g}_N{scan.grid_size}.npz",
                spectrum,
            )

            iteration: FineTransferRun | None = None
            circle = spectrum.localization.circle
            if circle is not None:
                prepared = PreparedEMKernel.build(
                    built.problem.grid,
                    k=built.problem.k0,
                )
                (iteration,) = run_parameter_transfer(
                    built,
                    {"same_grid": circle},
                    solver_config=SolverConfig(
                        max_iter=scan.max_iter,
                        rtol=scan.rtol,
                        divergence_guard=scan.divergence_guard,
                    ),
                    prepared_kernel=prepared,
                    retain_solution=False,
                )
                store.write_json(
                    (
                        "raw/chapter42/residual_histories/"
                        f"{material_key}_k{float(k0):g}_N{scan.grid_size}.json"
                    ),
                    iteration.solver_result.residual_history,
                )

            point = FixedSpectrumPoint(
                material_key=material_key,
                material_title=material_title,
                eps_r_diagonal=tuple(complex(value) for value in np.diag(tensor)),
                k0=float(k0),
                spectrum=spectrum,
                iteration=iteration,
            )
            points.append(point)
            rows.append(_point_row(point))
            store.log_event(
                "case_finish",
                study="chapter42_fixed_spectrum",
                material=material_key,
                k0=float(k0),
                localization_status=spectrum.localization.status.value,
                sim_status=(
                    iteration.solver_result.status
                    if iteration is not None
                    else "no_parameter"
                ),
            )

    store.write_rows("tables/S42_fixed_grid_spectrum_scan.csv", rows)

    result = FixedSpectrumStudyResult(
        points=tuple(points),
        output_root=str(store.root),
    )
    if render_figures:
        _render_figures(result, config, store)

    store.log_event(
        "study_finish",
        study="chapter42_fixed_spectrum",
        points=len(points),
    )
    return result


def _nearest_points(
    points: Iterable[FixedSpectrumPoint],
    selected: tuple[float, ...],
) -> tuple[FixedSpectrumPoint, ...]:
    materialized = tuple(points)
    if not materialized:
        return ()
    chosen: list[FixedSpectrumPoint] = []
    for target in selected:
        point = min(materialized, key=lambda item: abs(item.k0 - target))
        if point not in chosen:
            chosen.append(point)
    return tuple(chosen)


def _render_figures(
    result: FixedSpectrumStudyResult,
    config: SpectralStudyConfig,
    store: ArtifactStore,
) -> None:
    import matplotlib.pyplot as plt

    selections = {
        "real_anisotropy": config.fixed_spectrum.selected_real_wave_numbers,
        "lossy_anisotropy": config.fixed_spectrum.selected_lossy_wave_numbers,
    }
    for material_key, selected in selections.items():
        points = result.for_material(material_key)
        if not points:
            continue

        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        k0 = np.asarray([point.k0 for point in points], dtype=float)
        q = np.asarray(
            [
                point.spectrum.localization.circle.q
                if point.spectrum.localization.circle is not None
                else np.nan
                for point in points
            ],
            dtype=float,
        )
        radius = np.asarray(
            [
                point.spectrum.localization.circle.radius
                if point.spectrum.localization.circle is not None
                else np.nan
                for point in points
            ],
            dtype=float,
        )
        mu_abs = np.asarray(
            [
                abs(point.spectrum.localization.circle.mu)
                if point.spectrum.localization.circle is not None
                else np.nan
                for point in points
            ],
            dtype=float,
        )
        iterations = np.asarray(
            [
                point.iteration.solver_result.matvec_count
                if point.iteration is not None
                else np.nan
                for point in points
            ],
            dtype=float,
        )
        axes[0, 0].plot(k0, q, marker="o")
        axes[0, 0].axhline(1.0, linestyle="--", linewidth=1.0)
        axes[0, 0].set(
            xlabel="$k_0$",
            ylabel="$q=R/|\\mu|$",
            title="Спектральный коэффициент",
        )
        axes[0, 1].plot(k0, radius, marker="o", label="$R$")
        axes[0, 1].plot(k0, mu_abs, marker="s", label="$|\\mu|$")
        axes[0, 1].set(
            xlabel="$k_0$",
            ylabel="Модуль",
            title="Окружность и параметр",
        )
        axes[0, 1].legend()
        axes[1, 0].plot(k0, iterations, marker="o")
        axes[1, 0].set(
            xlabel="$k_0$",
            ylabel="Действия оператора",
            title="Стоимость SIM на той же сетке",
        )
        axes[1, 1].axis("off")
        statuses = [
            f"k0={point.k0:g}: "
            + (
                point.iteration.solver_result.status
                if point.iteration is not None
                else "no_parameter"
            )
            for point in points
        ]
        axes[1, 1].text(0.0, 1.0, "\n".join(statuses), va="top", family="monospace")
        figure.suptitle(points[0].material_title)
        store.save_figure(f"figures/S42/{material_key}_scan", figure)
        plt.close(figure)

        chosen = _nearest_points(points, selected)
        columns = min(3, len(chosen))
        rows = int(ceil(len(chosen) / columns))
        figure, axes_array = plt.subplots(
            rows,
            columns,
            figsize=(5.0 * columns, 4.2 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, point in zip(axes_array.flat, chosen, strict=False):
            spectrum = point.spectrum.localization.spectrum
            hull = point.spectrum.localization.hull
            axis.scatter(spectrum.real, spectrum.imag, s=8, alpha=0.65)
            if hull.size:
                closed = np.concatenate((hull, hull[:1]))
                axis.plot(closed.real, closed.imag, linestyle="--", linewidth=1.2)
            circle = point.spectrum.localization.circle
            if circle is not None:
                theta = np.linspace(0.0, 2.0 * np.pi, 361)
                boundary = circle.mu + circle.radius * np.exp(1j * theta)
                axis.plot(boundary.real, boundary.imag, linewidth=1.0)
                axis.scatter([circle.mu.real], [circle.mu.imag], marker="x", s=50)
            axis.scatter([0.0], [0.0], marker="+", s=50)
            axis.set(
                title=f"$k_0={point.k0:g}$",
                xlabel="Re $\\lambda$",
                ylabel="Im $\\lambda$",
            )
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(True, alpha=0.25)
        for axis in axes_array.flat[len(chosen) :]:
            axis.axis("off")
        figure.suptitle(points[0].material_title)
        store.save_figure(f"figures/S42/{material_key}_spectra", figure)
        plt.close(figure)

        residual_points = tuple(point for point in chosen if point.iteration is not None)
        if residual_points:
            figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
            for point in residual_points:
                history = point.iteration.solver_result.residual_history
                axis.semilogy(range(len(history)), history, label=f"$k_0={point.k0:g}$")
            axis.set(
                xlabel="Действия оператора",
                ylabel="Истинная относительная невязка",
                title=points[0].material_title,
            )
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
            store.save_figure(f"figures/S42/{material_key}_residuals", figure)
            plt.close(figure)


__all__ = [
    "FixedSpectrumPoint",
    "FixedSpectrumStudyResult",
    "run_fixed_spectrum_study",
]
