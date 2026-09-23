from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ensemble_transfer import EnsembleTransferStudyResult
from .geometry_resolution import GeometryResolutionStudyResult
from .volume_averaging import VolumeAveragingStudyResult
from .wave_number_phase import WaveNumberPhaseStudyResult
from .workflow import CaseStudyResult


def _save(figure, output: str | Path | None) -> None:
    if output is not None:
        figure.savefig(output, bbox_inches="tight")


def _rows_frame(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame([dict(row) for row in rows])


def _annotated_heatmap(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    annotation_format: str = ".2f",
    output: str | Path | None = None,
):
    if frame.empty:
        raise ValueError("cannot plot an empty table")
    pivot = frame.pivot(index=index, columns=columns, values=values).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    matrix = pivot.to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    image = axis.imshow(matrix, origin="lower", aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels([f"{value:g}" for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels([f"{value:g}" for value in pivot.index])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    format(value, annotation_format),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def plot_spectral_localizations(
    case: CaseStudyResult,
    *,
    output: str | Path | None = None,
):
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    for level, run in sorted(case.spectra.items()):
        axis.scatter(
            run.localization.spectrum.real,
            run.localization.spectrum.imag,
            s=8,
            alpha=0.45,
            label=f"N={level}",
        )
        hull = np.r_[run.localization.hull, run.localization.hull[:1]]
        axis.plot(hull.real, hull.imag, linewidth=1.0)
    axis.set_xlabel(r"$\operatorname{Re}\lambda$")
    axis.set_ylabel(r"$\operatorname{Im}\lambda$")
    axis.set_title(case.definition.title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def plot_transfer_bounds(
    case: CaseStudyResult,
    *,
    output: str | Path | None = None,
):
    valid = [run for run in case.control_transfers if run.assessment is not None]
    levels = [run.coarse.grid_shape[0] for run in valid]
    actual = [run.assessment.target_factor for run in valid]
    bounds = [run.assessment.target_factor_bound for run in valid]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(levels, actual, marker="o", label=r"$q_c(\mu_H)$")
    axis.plot(levels, bounds, marker="s", label="верхняя оценка")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"$N_H$")
    axis.set_ylabel("спектральный коэффициент")
    axis.set_title(case.definition.title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def plot_residual_histories(
    case: CaseStudyResult,
    *,
    output: str | Path | None = None,
):
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for run in case.fine_runs:
        axis.semilogy(
            run.solver_result.residual_history,
            label=run.parameter_label,
        )
    axis.set_xlabel("матрично-векторные умножения")
    axis.set_ylabel("относительная невязка")
    axis.set_title(case.definition.title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def plot_geometry_resolution_ratio(
    result: GeometryResolutionStudyResult,
    *,
    output: str | Path | None = None,
):
    return _annotated_heatmap(
        _rows_frame(result.summary_rows),
        index="inclusion_side",
        columns="N_H",
        values="observed_residual_ratio",
        title="E4: асимптотическое отношение невязок",
        xlabel=r"Размер грубой сетки $N_H$",
        ylabel="Сторона включения",
        colorbar_label="Наблюдаемое отношение",
        annotation_format=".3f",
        output=output,
    )


def plot_geometry_resolution_cells(
    result: GeometryResolutionStudyResult,
    *,
    output: str | Path | None = None,
):
    return _annotated_heatmap(
        _rows_frame(result.summary_rows),
        index="inclusion_side",
        columns="N_H",
        values="coarse_cells_min_axis",
        title="E4: геометрическое разрешение включения",
        xlabel=r"Размер грубой сетки $N_H$",
        ylabel="Сторона включения",
        colorbar_label="Минимальное число ячеек по оси",
        annotation_format=".0f",
        output=output,
    )


def plot_volume_averaging_improvement(
    result: VolumeAveragingStudyResult,
    *,
    output: str | Path | None = None,
):
    return _annotated_heatmap(
        _rows_frame(result.comparison_rows),
        index="inclusion_side",
        columns="N_H",
        values="ratio_improvement",
        title="E6: изменение коэффициента сходимости при объёмном усреднении",
        xlabel=r"Размер грубой сетки $N_H$",
        ylabel="Сторона включения",
        colorbar_label=r"$q_{obs}^{center}-q_{obs}^{avg}$",
        annotation_format=".3f",
        output=output,
    )


def plot_volume_averaging_ratios(
    result: VolumeAveragingStudyResult,
    *,
    output: str | Path | None = None,
):
    frame = _rows_frame(result.comparison_rows)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    grouped = frame.groupby("N_H", sort=True)
    levels = []
    center = []
    averaged = []
    for level, part in grouped:
        levels.append(level)
        center.append(float(np.nanmedian(part["center_ratio"])))
        averaged.append(float(np.nanmedian(part["avg_ratio"])))
    axis.plot(levels, center, marker="o", label="по центрам ячеек")
    axis.plot(levels, averaged, marker="s", label="объёмное усреднение")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"Размер грубой сетки $N_H$")
    axis.set_ylabel("Медианное отношение невязок")
    axis.set_title("E6: влияние способа представления границы")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def plot_ensemble_ratio(
    result: EnsembleTransferStudyResult,
    *,
    output: str | Path | None = None,
):
    frame = _rows_frame(result.comparison_rows)
    if frame.empty:
        raise ValueError("cannot plot an empty ensemble result")
    ensembles = list(dict.fromkeys(frame["ensemble"].tolist()))
    sides = sorted(frame["inclusion_side"].unique())
    pivot = frame.pivot(
        index="inclusion_side",
        columns="ensemble",
        values="ensemble_ratio",
    ).reindex(index=sides, columns=ensembles)
    matrix = pivot.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    image = axis.imshow(matrix, origin="lower", aspect="auto")
    axis.set_xticks(np.arange(len(ensembles)))
    axis.set_xticklabels(ensembles, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(sides)))
    axis.set_yticklabels([f"{value:g}" for value in sides])
    axis.set_xlabel("Ансамбль грубых сеток")
    axis.set_ylabel("Сторона включения")
    axis.set_title("E5: фактическая устойчивость ансамблевого параметра")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Наблюдаемое отношение невязок")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def _categorical_phase_heatmap(
    frame: pd.DataFrame,
    *,
    values: str,
    title: str,
    labels: tuple[str, ...],
    output: str | Path | None = None,
):
    if frame.empty:
        raise ValueError("cannot plot an empty phase table")
    pivot = (
        frame.pivot(index="k0", columns="N_H", values=values)
        .sort_index()
        .reindex(sorted(frame["N_H"].unique()), axis=1)
    )
    matrix = pivot.to_numpy(dtype=float)
    categories = len(labels)
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=-0.5,
        vmax=categories - 0.5,
        cmap=plt.get_cmap("viridis", categories),
    )
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels([f"{value:g}" for value in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels([f"{value:g}" for value in pivot.index])
    axis.set_xlabel(r"Размер грубой сетки $N_H$")
    axis.set_ylabel(r"Волновое число $k_0$")
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis, ticks=np.arange(categories))
    colorbar.ax.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    figure.tight_layout()
    _save(figure, output)
    return figure, axis


def plot_wave_phase_detailed(
    result: WaveNumberPhaseStudyResult,
    *,
    output: str | Path | None = None,
):
    return _categorical_phase_heatmap(
        _rows_frame(result.phase_rows),
        values="phase_code_detailed",
        title="E3/E3b: подробная фазовая классификация",
        labels=(
            "нет параметра",
            "неустойчив",
            "допуск достигнут",
            "затухает без допуска",
            "не разрешён",
            "nonfinite",
        ),
        output=output,
    )


def plot_wave_phase_article(
    result: WaveNumberPhaseStudyResult,
    *,
    output: str | Path | None = None,
):
    return _categorical_phase_heatmap(
        _rows_frame(result.phase_rows),
        values="article_phase_code",
        title="E3/E3b: трёхфазная проекция для статьи",
        labels=("параметра нет", "перенос неустойчив", "перенос устойчив"),
        output=output,
    )


def plot_wave_material_ppw(
    result: WaveNumberPhaseStudyResult,
    *,
    output: str | Path | None = None,
):
    return _annotated_heatmap(
        _rows_frame(result.phase_rows),
        index="k0",
        columns="N_H",
        values="coarse_ppw_material",
        title="E3: пространственное разрешение материальной длины волны",
        xlabel=r"Размер грубой сетки $N_H$",
        ylabel=r"Волновое число $k_0$",
        colorbar_label=r"$\mathrm{PPW}_{mat}$",
        annotation_format=".2f",
        output=output,
    )


def plot_wave_required_grid(
    result: WaveNumberPhaseStudyResult,
    *,
    output: str | Path | None = None,
):
    frame = _rows_frame(result.by_wave_number_rows).sort_values("k0")
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(frame["k0"], frame["N_nyquist_free"], marker="o", label="PPW=2, свободная длина")
    axis.plot(frame["k0"], frame["N_nyquist_material"], marker="s", label="PPW=2, материальная длина")
    axis.plot(frame["k0"], frame["min_N_parameter_exists"], marker="^", label="минимальный N с параметром")
    axis.plot(frame["k0"], frame["min_N_contracting"], marker="d", label="минимальный N с затуханием")
    axis.plot(frame["k0"], frame["min_N_reached_rtol"], marker="x", label="минимальный N с достижением допуска")
    axis.set_xlabel(r"Волновое число $k_0$")
    axis.set_ylabel(r"Размер грубой сетки $N_H$")
    axis.set_title("E3: требуемое пространственное разрешение")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, output)
    return figure, axis
