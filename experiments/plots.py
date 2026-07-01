"""Notebook-friendly plot groups for chapter 6 experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import em3d


def plot_three_field_slices(
    u,
    grid,
    *,
    part: str = "abs",
    component: int | None = None,
    output_dir=None,
    prefix: str = "field",
) -> list[tuple]:
    """Plot scalar field slices in the xy, xz, and yz planes."""
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    figures = []
    for plane in ("xy", "xz", "yz"):
        filename = None
        if output_path is not None:
            suffix = "norm" if component is None else f"component{component}"
            filename = str(output_path / f"{prefix}_{suffix}_{part}_{plane}.png")
        figures.append(
            em3d.vis.plot_field_scalar_slice(
                u,
                grid,
                plane=plane,
                part=part,
                component=component,
                title=f"{part} field slice {plane}",
                filename=filename,
            )
        )
    return figures


def _grid_size_from_row(row: dict) -> int:
    value = row.get("N")
    if isinstance(value, str):
        return int(value.split("x")[0])
    if isinstance(value, (tuple, list)):
        return int(value[0])
    return int(value)


def plot_fft_vs_dense_timing(rows, *, output_dir=None, filename: str = "fft_vs_dense_timing.png"):
    """Plot FFT and dense matvec timings against grid size."""
    plt = em3d.vis._require_matplotlib()
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=_grid_size_from_row)
    n_values = [_grid_size_from_row(row) for row in sorted_rows]
    fft_times = [float(row["fft_avg_sec"]) for row in sorted_rows]
    dense_times = [float(row["dense_avg_sec"]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(n_values, fft_times, marker="o", label="FFT matvec")
    ax.plot(n_values, dense_times, marker="s", label="Dense NumPy")
    ax.set_xlabel("N")
    ax.set_ylabel("time, s")
    ax.set_title("FFT-vs-dense matvec timing")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()
    if output_path is not None:
        fig.savefig(str(output_path / filename), dpi=160, bbox_inches="tight")
    return fig, ax


def _k0a_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def plot_rcs_scan(rows, *, output_dir=None) -> dict[str, object]:
    """Plot Cartesian and polar normalized RCS curves for every scan row."""
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    directional = []
    for row in rows:
        k0a = float(row["k0a"])
        token = _k0a_token(k0a)
        cartesian_filename = None
        polar_filename = None
        if output_path is not None:
            cartesian_filename = str(output_path / f"rcs_k0a{token}_cartesian.png")
            polar_filename = str(output_path / f"rcs_k0a{token}_polar.png")
        directional.append(
            em3d.vis.plot_rcs_comparison(
                row["phi"],
                row["sigma_num_norm"],
                row["sigma_mie_norm"],
                title=f"Normalized RCS, k0a={k0a:g}",
                filename=cartesian_filename,
            )
        )
        directional.append(
            em3d.vis.plot_rcs_comparison_polar(
                row["phi"],
                row["sigma_num_norm"],
                row["sigma_mie_norm"],
                title=f"Normalized RCS, k0a={k0a:g}",
                filename=polar_filename,
            )
        )

    summary = None
    if rows:
        plt = em3d.vis._require_matplotlib()
        k0a_values = [float(row["k0a"]) for row in rows]
        shape_err = [float(row["shape_err"]) for row in rows]
        scale_ratio = [float(row["scale_ratio"]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(k0a_values, shape_err, marker="o", label="shape error")
        ax.plot(k0a_values, scale_ratio, marker="s", label="scale ratio")
        ax.set_xlabel("k0a")
        ax.set_ylabel("value")
        ax.set_title("RCS scan diagnostics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if hasattr(fig, "tight_layout"):
            fig.tight_layout()
        if output_path is not None:
            fig.savefig(str(output_path / "rcs_scan_summary.png"), dpi=160, bbox_inches="tight")
        summary = (fig, ax)
    return {"directional": directional, "summary": summary}


def _history_from_run(run) -> tuple[str, list[float]]:
    if isinstance(run, dict):
        label = run.get("solver_name") or run.get("case_name") or "run"
        return str(label), [float(x) for x in run.get("residual_history", [])]
    label = getattr(run, "solver_name", None) or getattr(run, "case_name", None) or "run"
    return str(label), [float(x) for x in getattr(run, "residual_history", [])]


def plot_residual_histories(runs_or_rows, *, output_dir=None, filename: str = "residual_histories.png"):
    """Plot residual histories for solver runs or row dictionaries."""
    plt = em3d.vis._require_matplotlib()
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for run in runs_or_rows:
        label, history = _history_from_run(run)
        if history:
            ax.semilogy(np.arange(len(history)), history, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("Solver convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()
    if output_path is not None:
        fig.savefig(str(output_path / filename), dpi=160, bbox_inches="tight")
    return fig, ax
