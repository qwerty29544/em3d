from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from em3d.geometry import center_mask

from experiments.em_spectral.artifacts import ArtifactStore

from .workflow import MieCaseStudyResult, MieRCSCurveSet, StationaryCaseStudyResult


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _safe_token(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(" ", "_")


def plot_residuals_by_operator_actions(study: StationaryCaseStudyResult):
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for execution in study.executions:
        result = execution.result
        x = result.residual_action_counts
        if len(x) != len(result.residual_history):
            x = list(range(len(result.residual_history)))
        axis.semilogy(x, result.residual_history, marker="o", markersize=3, label=execution.solver_name)
    axis.set_xlabel("Число действий оператора и сопряжённого оператора")
    axis.set_ylabel("Относительная норма невязки")
    axis.set_title(f"Сходимость: {study.definition.title}")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def _slice_data(field: np.ndarray, grid, plane: str = "xz"):
    magnitude = np.sqrt(np.sum(np.abs(field) ** 2, axis=0))
    x = np.asarray(grid.backend.to_host(grid.x), dtype=np.float64)
    y = np.asarray(grid.backend.to_host(grid.y), dtype=np.float64)
    z = np.asarray(grid.backend.to_host(grid.z), dtype=np.float64)
    if plane == "xz":
        index = magnitude.shape[1] // 2
        return x, z, magnitude[:, index, :], (0, index, 0)
    if plane == "xy":
        index = magnitude.shape[2] // 2
        return x, y, magnitude[:, :, index], (0, 0, index)
    if plane == "yz":
        index = magnitude.shape[0] // 2
        return y, z, magnitude[index, :, :], (index, 0, 0)
    raise ValueError(f"unsupported plane {plane!r}")


def _mask_slice(mask: np.ndarray, plane: str):
    if plane == "xz":
        return mask[:, mask.shape[1] // 2, :]
    if plane == "xy":
        return mask[:, :, mask.shape[2] // 2]
    if plane == "yz":
        return mask[mask.shape[0] // 2, :, :]
    raise ValueError(f"unsupported plane {plane!r}")


def _component_slice_data(
    field: np.ndarray,
    grid,
    *,
    plane: str,
    component: int,
    part: str,
    phase_floor: float | None = None,
):
    if component not in (0, 1, 2):
        raise ValueError("component must be 0, 1, or 2")
    component_field = np.asarray(field[component], dtype=np.complex128)
    if part == "abs":
        scalar = np.abs(component_field)
    elif part == "real":
        scalar = np.real(component_field)
    elif part == "imag":
        scalar = np.imag(component_field)
    elif part == "phase":
        scalar = np.angle(component_field).astype(np.float64)
        if phase_floor is not None:
            scalar = scalar.copy()
            scalar[np.abs(component_field) < float(phase_floor)] = np.nan
    else:
        raise ValueError("part must be abs, real, imag, or phase")

    x = np.asarray(grid.backend.to_host(grid.x), dtype=np.float64)
    y = np.asarray(grid.backend.to_host(grid.y), dtype=np.float64)
    z = np.asarray(grid.backend.to_host(grid.z), dtype=np.float64)
    if plane == "xz":
        index = scalar.shape[1] // 2
        return x, z, scalar[:, index, :]
    if plane == "xy":
        index = scalar.shape[2] // 2
        return x, y, scalar[:, :, index]
    if plane == "yz":
        index = scalar.shape[0] // 2
        return y, z, scalar[index, :, :]
    raise ValueError(f"unsupported plane {plane!r}")


def plot_field_component_slices(
    study: StationaryCaseStudyResult,
    *,
    plane: str = "xz",
    component: int = 0,
    part: str = "abs",
):
    qualified = [
        execution
        for execution in study.executions
        if execution.qualified and execution.solution_host is not None
    ]
    if not qualified:
        raise ValueError("no qualified fields available")
    grid = study.built_case.problem.grid
    global_component_peak = max(
        float(np.max(np.abs(execution.solution_host[component])))
        for execution in qualified
    )
    phase_floor = (
        max(global_component_peak * 1e-8, 1e-14)
        if part == "phase"
        else None
    )
    slices = []
    for execution in qualified:
        horizontal, vertical, values = _component_slice_data(
            execution.solution_host,
            grid,
            plane=plane,
            component=component,
            part=part,
            phase_floor=phase_floor,
        )
        slices.append((execution.solver_name, horizontal, vertical, values))

    finite_values = [
        values[np.isfinite(values)]
        for _, _, _, values in slices
        if np.any(np.isfinite(values))
    ]
    if not finite_values:
        raise ValueError("component slices contain no finite values")
    component_label = ("x", "y", "z")[component]
    if part == "phase":
        vmin, vmax = -np.pi, np.pi
        colorbar_label = fr"$\arg E_{component_label}$"
    elif part in {"real", "imag"}:
        limit = max(float(np.max(np.abs(values))) for values in finite_values)
        vmin, vmax = -limit, limit
        operator_label = "Re" if part == "real" else "Im"
        colorbar_label = fr"$\operatorname{{{operator_label}}} E_{component_label}$"
    else:
        vmin = 0.0
        vmax = max(float(np.max(values)) for values in finite_values)
        colorbar_label = fr"$|E_{component_label}|$"

    mask = center_mask(study.definition.feature_geometry, grid)
    mask2d = _mask_slice(mask, plane)
    plt = _plt()
    figure, axes = plt.subplots(
        1,
        len(slices),
        figsize=(5.0 * len(slices), 4.2),
        squeeze=False,
    )
    image = None
    for axis, (name, horizontal, vertical, values) in zip(
        axes[0], slices, strict=True
    ):
        H, V = np.meshgrid(horizontal, vertical, indexing="ij")
        image = axis.pcolormesh(
            H,
            V,
            values,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        if np.any(mask2d) and np.any(~mask2d):
            axis.contour(H, V, mask2d.astype(float), levels=[0.5], linewidths=1.0)
        axis.set_title(name)
        axis.set_xlabel(plane[0])
        axis.set_ylabel(plane[1])
        axis.set_aspect("equal", adjustable="box")
    assert image is not None
    description = {
        "abs": "модуль",
        "real": "действительная часть",
        "imag": "мнимая часть",
        "phase": "фаза",
    }[part]
    figure.suptitle(
        f"Компонента E_{component_label}, {description}: {study.definition.title}"
    )
    figure.subplots_adjust(top=0.82, right=0.88, wspace=0.28)
    colorbar_axis = figure.add_axes([0.90, 0.18, 0.015, 0.64])
    figure.colorbar(image, cax=colorbar_axis, label=colorbar_label)
    return figure, axes


def plot_field_slices(
    study: StationaryCaseStudyResult,
    *,
    plane: str = "xz",
):
    qualified = [
        execution
        for execution in study.executions
        if execution.qualified and execution.solution_host is not None
    ]
    if not qualified:
        raise ValueError("no qualified fields available")
    grid = study.built_case.problem.grid
    slices = []
    for execution in qualified:
        horizontal, vertical, values, _ = _slice_data(
            execution.solution_host,
            grid,
            plane=plane,
        )
        slices.append((execution.solver_name, horizontal, vertical, values))
    vmin = min(float(np.min(values)) for _, _, _, values in slices)
    vmax = max(float(np.max(values)) for _, _, _, values in slices)
    mask = center_mask(study.definition.feature_geometry, grid)
    mask2d = _mask_slice(mask, plane)

    plt = _plt()
    figure, axes = plt.subplots(
        1,
        len(slices),
        figsize=(5.0 * len(slices), 4.2),
        squeeze=False,
    )
    image = None
    for axis, (name, horizontal, vertical, values) in zip(axes[0], slices, strict=True):
        H, V = np.meshgrid(horizontal, vertical, indexing="ij")
        image = axis.pcolormesh(H, V, values, shading="auto", vmin=vmin, vmax=vmax)
        if np.any(mask2d) and np.any(~mask2d):
            axis.contour(H, V, mask2d.astype(float), levels=[0.5], linewidths=1.0)
        axis.set_title(name)
        axis.set_xlabel(plane[0])
        axis.set_ylabel(plane[1])
        axis.set_aspect("equal", adjustable="box")
    assert image is not None
    figure.suptitle(f"Поле с общей цветовой шкалой: {study.definition.title}")
    figure.subplots_adjust(top=0.82, right=0.88, wspace=0.28)
    colorbar_axis = figure.add_axes([0.90, 0.18, 0.015, 0.64])
    figure.colorbar(image, cax=colorbar_axis, label=r"$|E|$")
    return figure, axes


def plot_field_differences(
    study: StationaryCaseStudyResult,
    *,
    plane: str = "xz",
):
    if study.reference_solver is None:
        raise ValueError("reference solver is unavailable")
    reference = next(
        execution
        for execution in study.executions
        if execution.solver_name == study.reference_solver
    )
    candidates = [
        execution
        for execution in study.executions
        if execution.qualified
        and execution.solution_host is not None
        and execution.solver_name != study.reference_solver
    ]
    if not candidates:
        raise ValueError("no non-reference qualified fields available")
    grid = study.built_case.problem.grid
    slices = []
    for execution in candidates:
        difference = execution.solution_host - reference.solution_host
        horizontal, vertical, values, _ = _slice_data(difference, grid, plane=plane)
        slices.append((execution.solver_name, horizontal, vertical, values))
    vmax = max(float(np.max(values)) for _, _, _, values in slices)

    plt = _plt()
    figure, axes = plt.subplots(
        1,
        len(slices),
        figsize=(5.0 * len(slices), 4.2),
        squeeze=False,
    )
    image = None
    for axis, (name, horizontal, vertical, values) in zip(axes[0], slices, strict=True):
        H, V = np.meshgrid(horizontal, vertical, indexing="ij")
        image = axis.pcolormesh(H, V, values, shading="auto", vmin=0.0, vmax=vmax)
        axis.set_title(f"{name} – {study.reference_solver}")
        axis.set_xlabel(plane[0])
        axis.set_ylabel(plane[1])
        axis.set_aspect("equal", adjustable="box")
    assert image is not None
    figure.suptitle(f"Разность полей: {study.definition.title}")
    figure.subplots_adjust(top=0.82, right=0.88, wspace=0.28)
    colorbar_axis = figure.add_axes([0.90, 0.18, 0.015, 0.64])
    figure.colorbar(image, cax=colorbar_axis, label=r"$|E-E_{ref}|$")
    return figure, axes


def plot_rcs_curves(
    study: StationaryCaseStudyResult,
    *,
    plane: str,
    normalized: bool,
):
    curves = [curve for curve in study.rcs_curves if curve.plane == plane]
    if not curves:
        raise ValueError(f"no RCS curves for plane {plane!r}")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for curve in curves:
        values = curve.sigma_normalized if normalized else curve.sigma
        axis.plot(np.rad2deg(curve.phi), values, label=curve.solver_name)
    axis.set_xlabel("Угол, градусы")
    axis.set_ylabel("Нормированная ЭПР" if normalized else "ЭПР")
    axis.set_title(
        f"{'Нормированная' if normalized else 'Абсолютная'} ЭПР, плоскость {plane}"
    )
    axis.set_xlim(0.0, 360.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def _selected_mie_rows(mie_cases: Iterable[MieCaseStudyResult]):
    """Select one qualified solver per physical Mie case for discretization plots.

    BiCGStab is preferred because it is the publication default.  If it did not
    qualify, the converged result with the smallest independently recomputed
    true residual is used.  Solver-to-solver agreement remains available in the
    tabular artifacts and is deliberately not mixed with the grid/Mie error.
    """

    for case in mie_cases:
        candidates = []
        validation_by_solver = {item.solver_name: item for item in case.validations}
        for execution in case.executions:
            validation = validation_by_solver.get(execution.solver_name)
            if validation is not None:
                candidates.append((execution, validation))
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                0 if item[0].solver_name == "BiCGStab" else 1,
                item[0].true_relative_residual,
            )
        )
        execution, validation = candidates[0]
        yield case, execution, validation


def _selected_mie_curve(case: MieCaseStudyResult) -> MieRCSCurveSet:
    if not case.rcs_curves:
        raise ValueError("no Mie RCS curve data available")
    return min(
        case.rcs_curves,
        key=lambda curve: (0 if curve.solver_name == "BiCGStab" else 1, curve.solver_name),
    )


def plot_mie_shape_error(mie_cases: Iterable[MieCaseStudyResult]):
    grouped: dict[tuple[float, int], list[tuple[float, float]]] = defaultdict(list)
    for case, _execution, validation in _selected_mie_rows(mie_cases):
        eps = complex(case.definition.feature_eps_r)
        k0a = case.definition.k0 * validation.nominal_radius
        grouped[(eps.real, case.grid_size)].append(
            (float(k0a), float(validation.nominal.rcs_normalized_l2))
        )
    if not grouped:
        raise ValueError("no qualified Mie validations available")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for (eps_real, grid_size), values in sorted(grouped.items()):
        values.sort()
        axis.semilogy(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            label=fr"$\varepsilon_r={eps_real:g}$, $N={grid_size}$",
        )
    axis.set_xlabel(r"$k_0a$")
    axis.set_ylabel("Ошибка формы нормированной ЭПР")
    axis.set_title("Верификация по решению Ми")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return figure, axis


def plot_mie_field_error(mie_cases: Iterable[MieCaseStudyResult]):
    grouped: dict[tuple[float, int], list[tuple[float, float]]] = defaultdict(list)
    for case, _execution, validation in _selected_mie_rows(mie_cases):
        value = float(validation.nominal.field_relative_l2)
        if not np.isfinite(value):
            continue
        eps = complex(case.definition.feature_eps_r)
        k0a = case.definition.k0 * validation.nominal_radius
        grouped[(eps.real, case.grid_size)].append((float(k0a), value))
    if not grouped:
        raise ValueError("no Mie field-reference metrics available")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for (eps_real, grid_size), values in sorted(grouped.items()):
        values.sort()
        axis.semilogy(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            label=fr"$\varepsilon_r={eps_real:g}$, $N={grid_size}$",
        )
    axis.set_xlabel(r"$k_0a$")
    axis.set_ylabel("Относительная ошибка полного поля")
    axis.set_title("Сопоставление поля с аналитическим решением Ми")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return figure, axis


def plot_mie_field_slices(case: MieCaseStudyResult):
    if not case.field_slices:
        raise ValueError("no Mie field slice data available")
    data = case.field_slices[0]
    rows = (
        (
            "Полное поле",
            data.numerical_total,
            data.analytic_nominal_total,
            data.analytic_effective_total,
        ),
        (
            "Рассеянное поле",
            data.numerical_scattered,
            data.analytic_nominal_scattered,
            data.analytic_effective_scattered,
        ),
    )
    plt = _plt()
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), squeeze=False)
    titles = ("Численно", "Ми: номинальный радиус", "Ми: эффективный радиус")
    H, V = np.meshgrid(data.horizontal, data.vertical, indexing="ij")
    for row_index, (row_label, *arrays) in enumerate(rows):
        vmin = min(float(np.min(array)) for array in arrays)
        vmax = max(float(np.max(array)) for array in arrays)
        image = None
        for column_index, (title, values) in enumerate(zip(titles, arrays, strict=True)):
            axis = axes[row_index, column_index]
            image = axis.pcolormesh(H, V, values, shading="auto", vmin=vmin, vmax=vmax)
            axis.set_title(title if row_index == 0 else "")
            axis.set_xlabel(data.plane[0])
            axis.set_ylabel(data.plane[1] if column_index == 0 else "")
            axis.set_aspect("equal", adjustable="box")
        axes[row_index, 0].text(
            -0.28,
            0.5,
            row_label,
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row_index, 0].transAxes,
        )
        assert image is not None
        cax = figure.add_axes([0.91, 0.56 - 0.43 * row_index, 0.012, 0.30])
        figure.colorbar(image, cax=cax, label=r"$|E|$")
    eps = complex(case.definition.feature_eps_r)
    validation = case.validations[0]
    k0a = case.definition.k0 * validation.nominal_radius
    figure.suptitle(
        fr"Поле сферы: $\varepsilon_r={eps.real:g}$, $k_0a={k0a:g}$, "
        fr"$N={case.grid_size}$ ({data.solver_name})"
    )
    figure.subplots_adjust(left=0.09, right=0.89, top=0.88, hspace=0.32, wspace=0.24)
    return figure, axes


def plot_mie_rcs_comparison(case: MieCaseStudyResult):
    curve = _selected_mie_curve(case)
    degrees = np.rad2deg(curve.phi)

    def normalized(values):
        peak = float(np.max(values)) if len(values) else 0.0
        return values / peak if peak > 0.0 else np.zeros_like(values)

    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].plot(degrees, curve.sigma_numerical, label=f"Численно ({curve.solver_name})")
    axes[0].plot(degrees, curve.sigma_nominal, linestyle="--", label="Ми, номинальный радиус")
    axes[0].plot(degrees, curve.sigma_effective, linestyle=":", label="Ми, эффективный радиус")
    axes[0].set_ylabel("ЭПР")
    axes[0].set_title("Абсолютная диаграмма")

    axes[1].plot(degrees, normalized(curve.sigma_numerical), label="Численно")
    axes[1].plot(degrees, normalized(curve.sigma_nominal), linestyle="--", label="Ми, номинальный радиус")
    axes[1].plot(degrees, normalized(curve.sigma_effective), linestyle=":", label="Ми, эффективный радиус")
    axes[1].set_ylabel("Нормированная ЭПР")
    axes[1].set_title("Форма диаграммы")

    for axis in axes:
        axis.set_xlabel("Угол, градусы")
        axis.set_xlim(0.0, 360.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    eps = complex(case.definition.feature_eps_r)
    k0a = case.definition.k0 * case.validations[0].nominal_radius
    figure.suptitle(fr"Сфера: $\varepsilon_r={eps.real:g}$, $k_0a={k0a:g}$, $N={case.grid_size}$")
    figure.subplots_adjust(top=0.82, wspace=0.28)
    return figure, axes


def plot_mie_scattered_field_error(mie_cases: Iterable[MieCaseStudyResult]):
    grouped: dict[tuple[float, int], list[tuple[float, float]]] = defaultdict(list)
    for case, _execution, validation in _selected_mie_rows(mie_cases):
        value = float(validation.nominal.scattered_field_outside_relative_l2)
        if not np.isfinite(value):
            continue
        eps = complex(case.definition.feature_eps_r)
        k0a = case.definition.k0 * validation.nominal_radius
        grouped[(eps.real, case.grid_size)].append((float(k0a), value))
    if not grouped:
        raise ValueError("no Mie scattered-field metrics available")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for (eps_real, grid_size), values in sorted(grouped.items()):
        values.sort()
        axis.semilogy(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            label=fr"$\varepsilon_r={eps_real:g}$, $N={grid_size}$",
        )
    axis.set_xlabel(r"$k_0a$")
    axis.set_ylabel("Ошибка рассеянного поля вне сферы")
    axis.set_title("Верификация рассеянной компоненты поля")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return figure, axis


def plot_mie_geometry_error(mie_cases: Iterable[MieCaseStudyResult]):
    values: dict[int, tuple[float, float]] = {}
    for case in mie_cases:
        if not case.validations:
            continue
        validation = case.validations[0]
        # Geometry depends on N and radius, not on material, frequency, or solver.
        values[int(case.grid_size)] = (
            validation.relative_volume_error,
            validation.effective_radius / validation.nominal_radius,
        )
    if not values:
        raise ValueError("no Mie geometry metrics available")
    plt = _plt()
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    grid_sizes = sorted(values)
    axis.plot(
        grid_sizes,
        [100.0 * values[N][0] for N in grid_sizes],
        marker="o",
        label="Ошибка объёма, %",
    )
    axis.plot(
        grid_sizes,
        [100.0 * (values[N][1] - 1.0) for N in grid_sizes],
        marker="s",
        label="Ошибка эффективного радиуса, %",
    )
    axis.set_xlabel("N")
    axis.set_ylabel("Относительное отклонение, %")
    axis.set_title("Геометрическая ошибка воксельного представления сферы")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def render_validation_figures(
    stationary_cases: Iterable[StationaryCaseStudyResult],
    mie_cases: Iterable[MieCaseStudyResult],
    *,
    store: ArtifactStore,
) -> None:
    plt = _plt()
    for study in stationary_cases:
        token = _safe_token(study.definition.key)
        figures = []
        try:
            figure, _ = plot_residuals_by_operator_actions(study)
            store.save_figure(f"figures/solver/{token}_residuals", figure)
            figures.append(figure)
        except ValueError:
            pass
        try:
            figure, _ = plot_field_slices(study, plane="xz")
            store.save_figure(f"figures/field/{token}_xz", figure)
            figures.append(figure)
        except ValueError:
            pass
        for part in ("abs", "phase"):
            try:
                figure, _ = plot_field_component_slices(
                    study,
                    plane="xz",
                    component=0,
                    part=part,
                )
                store.save_figure(
                    f"figures/field/{token}_Ex_{part}_xz",
                    figure,
                )
                figures.append(figure)
            except ValueError:
                pass
        try:
            figure, _ = plot_field_differences(study, plane="xz")
            store.save_figure(f"figures/field/{token}_differences_xz", figure)
            figures.append(figure)
        except ValueError:
            pass
        for plane in sorted({curve.plane for curve in study.rcs_curves}):
            for normalized in (False, True):
                figure, _ = plot_rcs_curves(
                    study,
                    plane=plane,
                    normalized=normalized,
                )
                suffix = "normalized" if normalized else "absolute"
                store.save_figure(
                    f"figures/rcs/{token}_{plane}_{suffix}",
                    figure,
                )
                figures.append(figure)
        for figure in figures:
            plt.close(figure)

    mie_cases = tuple(mie_cases)
    if mie_cases:
        maximum_grid = max(case.grid_size for case in mie_cases)
        field_cases: dict[str, MieCaseStudyResult] = {}
        for case in mie_cases:
            if not case.field_slices:
                continue
            current = field_cases.get(case.definition.key)
            if current is None or case.grid_size > current.grid_size:
                field_cases[case.definition.key] = case
        for case in field_cases.values():
            try:
                figure, _ = plot_mie_field_slices(case)
            except ValueError:
                continue
            store.save_figure(
                f"figures/mie/fields/{_safe_token(case.definition.key)}_N{case.grid_size}",
                figure,
            )
            plt.close(figure)

        for case in mie_cases:
            if case.grid_size != maximum_grid or not case.rcs_curves:
                continue
            try:
                figure, _ = plot_mie_rcs_comparison(case)
            except ValueError:
                continue
            store.save_figure(
                f"figures/mie/curves/{_safe_token(case.definition.key)}_N{case.grid_size}",
                figure,
            )
            plt.close(figure)
        for name, factory in (
            ("mie_rcs_shape_error", plot_mie_shape_error),
            ("mie_field_error", plot_mie_field_error),
            ("mie_scattered_field_error", plot_mie_scattered_field_error),
            ("mie_geometry_error", plot_mie_geometry_error),
        ):
            try:
                figure, _ = factory(mie_cases)
            except ValueError:
                continue
            store.save_figure(f"figures/mie/{name}", figure)
            plt.close(figure)


__all__ = [
    "plot_field_component_slices",
    "plot_field_differences",
    "plot_field_slices",
    "plot_mie_field_error",
    "plot_mie_field_slices",
    "plot_mie_geometry_error",
    "plot_mie_rcs_comparison",
    "plot_mie_scattered_field_error",
    "plot_mie_shape_error",
    "plot_rcs_curves",
    "plot_residuals_by_operator_actions",
    "render_validation_figures",
]
