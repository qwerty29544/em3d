from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import pi
from pathlib import Path
from typing import Any, Iterable


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "да"}


def _float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: float, significant: int = 2) -> str:
    return f"{float(value):.{significant}g}"


def _best_solver_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "ok")) not in {"", "ok", "converged", "max_iter"}:
            continue
        key = (str(row["eps_real"]), str(row["k"]))
        grouped[key].append(row)

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for key, candidates in grouped.items():
        converged = [row for row in candidates if _truth(row.get("converged"))]
        if converged:
            selected[key] = min(
                converged,
                key=lambda row: (
                    int(float(row.get("iterations", 10**12))),
                    _float(row.get("final_residual")),
                    str(row.get("solver_name", "")),
                ),
            )
        else:
            selected[key] = min(
                candidates,
                key=lambda row: (
                    _float(row.get("final_residual")),
                    int(float(row.get("iterations", 10**12))),
                    str(row.get("solver_name", "")),
                ),
            )
    return selected


def build_best_metric_rows_by_eps(
    solver_rows: Iterable[dict[str, Any]],
    metric_rows: Iterable[dict[str, Any]],
) -> dict[str, list[list[str]]]:
    """Build compact dissertation rows using one solver per (eps, k).

    A converged run is selected by the smallest iteration count.  If no solver
    converged, the smallest final residual is used and the row remains marked
    as non-converged.  Metrics are joined by material, wave number, and solver.
    """

    selected = _best_solver_rows(solver_rows)
    metrics = {
        (str(row["eps_real"]), str(row["k"]), str(row["solver_name"])): row
        for row in metric_rows
    }
    tables: dict[str, list[list[str]]] = defaultdict(list)
    for (eps, k_text), solver_row in selected.items():
        solver = str(solver_row["solver_name"])
        metric = metrics.get((eps, k_text, solver))
        if metric is None:
            continue
        k = float(k_text)
        residual = _float(solver_row.get("final_residual"))
        row = [
            f"{k:g}",
            f"{2.0 * pi / k:.3f}",
            solver,
            "да" if _truth(solver_row.get("converged")) else "нет",
            str(int(float(solver_row.get("iterations", 0)))),
            _format_number(residual, 2),
            f"{100.0 * _float(metric.get('shape_err'), float('nan')):.2f}",
            f"{_float(metric.get('scale_ratio'), float('nan')):.3f}",
            f"{100.0 * _float(metric.get('abs_rel_err'), float('nan')):.2f}",
        ]
        tables[eps].append(row)

    for rows in tables.values():
        rows.sort(key=lambda row: float(row[0]))
    return dict(sorted(tables.items(), key=lambda item: float(item[0])))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("solver_csv", type=Path)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    tables = build_best_metric_rows_by_eps(
        _read_csv(args.solver_csv),
        _read_csv(args.metrics_csv),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "k",
        "lambda",
        "solver",
        "converged",
        "iterations",
        "residual",
        "shape_error_percent",
        "scale_ratio",
        "absolute_error_percent",
    ]
    for eps, rows in tables.items():
        path = args.output_dir / f"mie_discrepancy_eps_{eps}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
