from __future__ import annotations

import csv
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

from experiments.em_spectral.artifacts import ArtifactStore

from .large_workflow import _create_archives, _observed_order_rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() == path.read_bytes():
                continue
            target = target.with_name(f"{path.parent.name}_{target.name}")
        shutil.copy2(path, target)


def merge_large_grid_runs(
    inputs: Sequence[str | Path],
    *,
    output_root: str | Path,
) -> Path:
    """Merge independent Kaggle job/profile outputs into one manifest tree."""

    roots = [Path(value).resolve() for value in inputs]
    if not roots:
        raise ValueError("at least one input run is required")
    for root in roots:
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(root / "manifest.json")

    store = ArtifactStore(
        output_root,
        schema="em3d-chapter4-all-solvers-large-grid-merged-v1",
    )
    table_names: set[str] = set()
    for root in roots:
        table_names.update(path.stem for path in (root / "tables").glob("*.csv"))
        _copy_tree_contents(root / "figures", store.root / "figures")
        _copy_tree_contents(root / "raw", store.root / "raw")

    merged_tables: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(table_names):
        rows: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for root in roots:
            for row in _read_csv(root / "tables" / f"{name}.csv"):
                marker = tuple(sorted((str(key), str(value)) for key, value in row.items()))
                if marker in seen:
                    continue
                seen.add(marker)
                rows.append(row)
        merged_tables[name] = rows

    convergence = merged_tables.get("mie_solver_vs_mie_rcs", [])
    merged_tables["mie_grid_convergence_by_solver"] = list(convergence)
    observed: list[dict[str, Any]] = []
    for metric in ("normalized_l2", "absolute_l2", "absolute_linf"):
        observed.extend(_observed_order_rows(convergence, metric))
    merged_tables["mie_observed_orders_by_solver"] = observed

    for name, rows in merged_tables.items():
        store.write_rows(f"tables/{name}.csv", rows)
        store.write_json(f"raw/tables/{name}.json", rows)

    for path in sorted((store.root / "figures").rglob("*")):
        if path.is_file():
            store.record_existing(path.relative_to(store.root))
    _create_archives(store)
    manifest = store.finalize(
        config={
            "mode": "merge",
            "inputs": [str(root) for root in roots],
        },
        status="complete",
    )
    return manifest


__all__ = ["merge_large_grid_runs"]
