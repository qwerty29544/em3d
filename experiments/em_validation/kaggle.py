from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Literal, Sequence

from .config import RuntimeConfig, ValidationStudyConfig


BatchAxis = Literal["k0a", "eps", "grid"]


@dataclass(frozen=True)
class KaggleBatchPlan:
    """Description of one independently runnable Kaggle publication batch."""

    index: int
    count: int
    axis: BatchAxis
    selected_values: tuple[object, ...]

    @property
    def label(self) -> str:
        return f"{self.axis}-batch-{self.index + 1:02d}-of-{self.count:02d}"


def _partition(
    values: Sequence[object],
    *,
    index: int,
    count: int,
) -> tuple[object, ...]:
    if count <= 0:
        raise ValueError("batch count must be positive")
    if index < 0 or index >= count:
        raise ValueError(f"batch index must satisfy 0 <= index < {count}, got {index}")
    selected = tuple(value for position, value in enumerate(values) if position % count == index)
    if not selected:
        raise ValueError(
            f"batch {index}/{count} is empty for {len(values)} configured values"
        )
    return selected


def publication_config_for_kaggle(
    *,
    output_root: str | Path,
    precision: Literal["single", "double"] = "double",
    solver_grid_size: int | None = None,
    mie_grid_sizes: Sequence[int] | None = None,
    mie_eps_values: Sequence[complex] | None = None,
    mie_k0a_values: Sequence[float] | None = None,
    mie_batch_axis: BatchAxis = "k0a",
    mie_batch_index: int = 0,
    mie_batch_count: int = 1,
    render_progress: bool = True,
    clear_cuda_cache_between_cases: bool = True,
) -> tuple[ValidationStudyConfig, KaggleBatchPlan]:
    """Build a CUDA-only publication configuration for a Kaggle session.

    The Mie tensor-product sweep can be split along one axis.  Each batch writes
    to an independent output directory and can therefore be executed in a
    separate Kaggle session without changing the scientific configuration.
    """

    config = ValidationStudyConfig.publication(
        output_root=Path(output_root),
        device="cuda",
    )
    runtime = RuntimeConfig(
        device="cuda",
        precision=precision,
        progress=bool(render_progress),
        clear_cuda_cache_between_cases=bool(clear_cuda_cache_between_cases),
    )
    solver = config.solver
    if solver_grid_size is not None:
        if int(solver_grid_size) <= 0:
            raise ValueError("solver_grid_size must be positive")
        solver = replace(solver, grid_size=int(solver_grid_size))

    mie = config.mie
    if mie_grid_sizes is not None:
        mie = replace(mie, grid_sizes=tuple(int(value) for value in mie_grid_sizes))
    if mie_eps_values is not None:
        mie = replace(mie, eps_values=tuple(complex(value) for value in mie_eps_values))
    if mie_k0a_values is not None:
        mie = replace(mie, k0a_values=tuple(float(value) for value in mie_k0a_values))

    axis_values: tuple[object, ...]
    if mie_batch_axis == "k0a":
        axis_values = tuple(mie.k0a_values)
    elif mie_batch_axis == "eps":
        axis_values = tuple(mie.eps_values)
    elif mie_batch_axis == "grid":
        axis_values = tuple(mie.grid_sizes)
    else:  # pragma: no cover - Literal guards typed callers
        raise ValueError(f"unsupported batch axis {mie_batch_axis!r}")

    selected = _partition(
        axis_values,
        index=int(mie_batch_index),
        count=int(mie_batch_count),
    )
    if mie_batch_axis == "k0a":
        mie = replace(mie, k0a_values=tuple(float(value) for value in selected))
    elif mie_batch_axis == "eps":
        mie = replace(mie, eps_values=tuple(complex(value) for value in selected))
    else:
        selected_grids = tuple(int(value) for value in selected)
        field_reference_grids = tuple(
            value for value in mie.field_reference_grids if value in selected_grids
        )
        mie = replace(
            mie,
            grid_sizes=selected_grids,
            field_reference_grids=field_reference_grids,
        )

    plan = KaggleBatchPlan(
        index=int(mie_batch_index),
        count=int(mie_batch_count),
        axis=mie_batch_axis,
        selected_values=selected,
    )
    return replace(config, runtime=runtime, solver=solver, mie=mie), plan


def archive_results(
    output_root: str | Path,
    *,
    destination: str | Path | None = None,
) -> tuple[Path, str]:
    """Create a ZIP archive and return its path and SHA-256 digest."""

    root = Path(output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if destination is None:
        destination_path = root.parent / f"{root.name}.zip"
    else:
        destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = destination_path.with_suffix("")
    archive = Path(
        shutil.make_archive(
            str(base_name),
            "zip",
            root_dir=root.parent,
            base_dir=root.name,
        )
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def write_session_metadata(
    output_root: str | Path,
    payload: dict,
) -> Path:
    """Write notebook-level metadata beside the package-generated manifest."""

    path = Path(output_root) / "kaggle_session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


__all__ = [
    "BatchAxis",
    "KaggleBatchPlan",
    "archive_results",
    "publication_config_for_kaggle",
    "write_session_metadata",
]
