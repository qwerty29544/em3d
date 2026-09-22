from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .workflow import CaseStudyResult


def plot_spectral_localizations(case: CaseStudyResult, *, output: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    for level, run in sorted(case.spectra.items()):
        axis.scatter(run.localization.spectrum.real, run.localization.spectrum.imag, s=8, alpha=0.45, label=f"N={level}")
        hull = np.r_[run.localization.hull, run.localization.hull[:1]]
        axis.plot(hull.real, hull.imag, linewidth=1.0)
    axis.set_xlabel(r"$\operatorname{Re}\lambda$")
    axis.set_ylabel(r"$\operatorname{Im}\lambda$")
    axis.set_title(case.definition.title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    if output is not None:
        figure.savefig(output, bbox_inches="tight")
    return figure, axis


def plot_transfer_bounds(case: CaseStudyResult, *, output: str | Path | None = None):
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
    if output is not None:
        figure.savefig(output, bbox_inches="tight")
    return figure, axis


def plot_residual_histories(case: CaseStudyResult, *, output: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for run in case.fine_runs:
        axis.semilogy(run.solver_result.residual_history, label=run.parameter_label)
    axis.set_xlabel("матрично-векторные умножения")
    axis.set_ylabel("относительная невязка")
    axis.set_title(case.definition.title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    if output is not None:
        figure.savefig(output, bbox_inches="tight")
    return figure, axis
