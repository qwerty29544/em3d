from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finite_limit(arrays: Sequence[np.ndarray], *, symmetric: bool = False):
    values = [np.asarray(value) for value in arrays if np.asarray(value).size]
    finite = [value[np.isfinite(value)] for value in values]
    finite = [value for value in finite if value.size]
    if not finite:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        limit = max(float(np.max(np.abs(value))) for value in finite)
        return -limit, limit
    return 0.0, max(float(np.max(value)) for value in finite)


def plot_solver_residuals(executions, *, title: str):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    for execution in executions:
        result = execution.result
        history = np.asarray(result.residual_history, dtype=float)
        if history.size == 0:
            continue
        counts = np.asarray(result.residual_action_counts, dtype=float)
        if counts.size != history.size:
            counts = np.arange(history.size, dtype=float)
        label = f"{execution.solver_name} [{result.status}]"
        axis.semilogy(counts, history, label=label)
    axis.set_xlabel(r"Число действий $A+A^*$")
    axis.set_ylabel("Относительная невязка")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _mesh(horizontal, vertical):
    return np.meshgrid(horizontal, vertical, indexing="ij")


def plot_mie_field_panel(
    *,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    solver_slices: Mapping[str, Mapping[str, np.ndarray]],
    analytic_slice: Mapping[str, np.ndarray],
    title: str,
    plane: str,
    solver_order: Sequence[str] = ("SIM", "BiCGStab", "TwoStep"),
):
    """Plot total/scattered/component magnitudes for all solvers and Mie."""

    plt = _plt()
    columns = list(solver_order) + ["Mie"]
    rows = (
        ("total_magnitude", r"$|E|$"),
        ("scattered_magnitude", r"$|E^{sc}|$"),
        ("scattered_component_abs", r"$|E_x^{sc}|$"),
    )
    arrays_by_row: dict[str, list[np.ndarray]] = {}
    for key, _ in rows:
        arrays_by_row[key] = [
            np.asarray(solver_slices[name][key])
            for name in solver_order
            if name in solver_slices
        ] + [np.asarray(analytic_slice[key])]
    limits = {key: _finite_limit(values) for key, values in arrays_by_row.items()}

    figure, axes = plt.subplots(
        len(rows), len(columns), figsize=(4.0 * len(columns), 3.5 * len(rows)), squeeze=False
    )
    H, V = _mesh(horizontal, vertical)
    images = {}
    for row_index, (key, label) in enumerate(rows):
        vmin, vmax = limits[key]
        for column_index, name in enumerate(columns):
            axis = axes[row_index, column_index]
            if name == "Mie":
                values = np.asarray(analytic_slice[key])
            elif name in solver_slices:
                values = np.asarray(solver_slices[name][key])
            else:
                axis.text(0.5, 0.5, "нет квалифицированного решения", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            image = axis.pcolormesh(H, V, values, shading="auto", vmin=vmin, vmax=vmax)
            images[row_index] = image
            if row_index == 0:
                axis.set_title(name)
            if column_index == 0:
                axis.set_ylabel(f"{plane[1]}\n{label}")
            else:
                axis.set_ylabel(plane[1])
            axis.set_xlabel(plane[0])
            axis.set_aspect("equal", adjustable="box")
        if row_index in images:
            figure.colorbar(images[row_index], ax=axes[row_index, :].tolist(), shrink=0.78, label=label)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.91, wspace=0.28, hspace=0.28)
    return figure


def plot_mie_error_phase_panel(
    *,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    solver_slices: Mapping[str, Mapping[str, np.ndarray]],
    title: str,
    plane: str,
    solver_order: Sequence[str] = ("SIM", "BiCGStab", "TwoStep"),
):
    plt = _plt()
    available = [name for name in solver_order if name in solver_slices]
    if not available:
        raise ValueError("no qualified solver slices")
    error_limit = _finite_limit(
        [solver_slices[name]["scattered_error_magnitude"] for name in available]
    )[1]
    figure, axes = plt.subplots(2, len(available), figsize=(4.2 * len(available), 7.0), squeeze=False)
    H, V = _mesh(horizontal, vertical)
    phase_image = None
    error_image = None
    for index, name in enumerate(available):
        phase = np.asarray(solver_slices[name]["scattered_component_phase"])
        error = np.asarray(solver_slices[name]["scattered_error_magnitude"])
        phase_image = axes[0, index].pcolormesh(
            H, V, phase, shading="auto", vmin=-np.pi, vmax=np.pi
        )
        error_image = axes[1, index].pcolormesh(
            H, V, error, shading="auto", vmin=0.0, vmax=error_limit
        )
        axes[0, index].set_title(name)
        for axis in axes[:, index]:
            axis.set_xlabel(plane[0])
            axis.set_ylabel(plane[1])
            axis.set_aspect("equal", adjustable="box")
    assert phase_image is not None and error_image is not None
    figure.colorbar(phase_image, ax=axes[0, :].tolist(), shrink=0.8, label=r"$\arg E_x^{sc}$")
    figure.colorbar(error_image, ax=axes[1, :].tolist(), shrink=0.8, label=r"$|E_h^{sc}-E_{Mie}^{sc}|$")
    figure.suptitle(title)
    figure.subplots_adjust(top=0.9, wspace=0.28, hspace=0.3)
    return figure


def plot_rcs_four_panel(
    *,
    phi: np.ndarray,
    numerical_curves: Mapping[str, np.ndarray],
    title: str,
    nominal: np.ndarray | None = None,
    effective: np.ndarray | None = None,
):
    """Absolute/normalised RCS in Cartesian and polar coordinates."""

    plt = _plt()
    figure = plt.figure(figsize=(12.5, 9.5))
    axes = (
        figure.add_subplot(2, 2, 1),
        figure.add_subplot(2, 2, 2),
        figure.add_subplot(2, 2, 3, projection="polar"),
        figure.add_subplot(2, 2, 4, projection="polar"),
    )
    curves: list[tuple[str, np.ndarray]] = [
        (name, np.asarray(values, dtype=float))
        for name, values in numerical_curves.items()
    ]
    if nominal is not None:
        curves.append(("Mie, nominal", np.asarray(nominal, dtype=float)))
    if effective is not None:
        curves.append(("Mie, effective", np.asarray(effective, dtype=float)))
    for label, values in curves:
        peak = float(np.max(values)) if values.size else 0.0
        normalized = values / peak if peak > 0.0 else np.zeros_like(values)
        axes[0].plot(np.rad2deg(phi), values, label=label)
        axes[1].plot(np.rad2deg(phi), normalized, label=label)
        axes[2].plot(phi, values, label=label)
        axes[3].plot(phi, normalized, label=label)
    axes[0].set_title("Абсолютная ЭПР, декартовы координаты")
    axes[1].set_title("Нормированная ЭПР, декартовы координаты")
    axes[2].set_title("Абсолютная ЭПР, полярные координаты")
    axes[3].set_title("Нормированная ЭПР, полярные координаты")
    for axis in axes[:2]:
        axis.set_xlabel("Угол, градусы")
        axis.set_xlim(0.0, 360.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.3)
    for axis in axes[2:]:
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("ЭПР")
    axes[1].set_ylabel("Нормированная ЭПР")
    axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def plot_stationary_field_panel(
    *,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    solver_slices: Mapping[str, Mapping[str, np.ndarray]],
    title: str,
    plane: str,
    solver_order: Sequence[str] = ("SIM", "BiCGStab", "TwoStep"),
):
    plt = _plt()
    available = [name for name in solver_order if name in solver_slices]
    if not available:
        raise ValueError("no qualified stationary fields")
    rows = (
        ("total_magnitude", r"$|E|$", False),
        ("component_abs", r"$|E_x|$", False),
        ("component_phase", r"$\arg E_x$", True),
    )
    limits = {}
    for key, _label, symmetric in rows:
        if key == "component_phase":
            limits[key] = (-np.pi, np.pi)
        else:
            limits[key] = _finite_limit([solver_slices[name][key] for name in available], symmetric=symmetric)
    H, V = _mesh(horizontal, vertical)
    figure, axes = plt.subplots(3, len(available), figsize=(4.1 * len(available), 10.0), squeeze=False)
    images = {}
    for row_index, (key, label, _symmetric) in enumerate(rows):
        vmin, vmax = limits[key]
        for column_index, name in enumerate(available):
            values = np.asarray(solver_slices[name][key])
            image = axes[row_index, column_index].pcolormesh(H, V, values, shading="auto", vmin=vmin, vmax=vmax)
            images[row_index] = image
            if row_index == 0:
                axes[row_index, column_index].set_title(name)
            axes[row_index, column_index].set_xlabel(plane[0])
            axes[row_index, column_index].set_ylabel(plane[1])
            axes[row_index, column_index].set_aspect("equal", adjustable="box")
        figure.colorbar(images[row_index], ax=axes[row_index, :].tolist(), shrink=0.78, label=label)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.91, wspace=0.28, hspace=0.28)
    return figure


def plot_grid_metric(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    title: str,
    ylabel: str,
):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        value_f = float(value)
        if not np.isfinite(value_f):
            continue
        grouped.setdefault(str(row["solver_name"]), []).append((int(row["grid_size"]), value_f))
    for name, values in sorted(grouped.items()):
        values.sort()
        axis.loglog([item[0] for item in values], [item[1] for item in values], marker="o", label=name)
    axis.set_xlabel("N")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


__all__ = [
    "plot_grid_metric",
    "plot_mie_error_phase_panel",
    "plot_mie_field_panel",
    "plot_rcs_four_panel",
    "plot_solver_residuals",
    "plot_stationary_field_panel",
]
