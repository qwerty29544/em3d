from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

import em3d
from em3d.operator import Operator, PreparedEMKernel
from em3d.experiments.spectral_transfer import BuiltSpectralCase

from .config import CudaMemoryPolicy

_GIB = float(1024**3)


@dataclass(frozen=True)
class MemoryEstimate:
    grid_size: int
    precision: str
    include_adjoint: bool
    adjoint_storage: str
    kernel_bytes: int
    problem_bytes: int
    solver_workspace_bytes: int
    farfield_workspace_bytes: int
    fft_workspace_allowance_bytes: int
    estimated_peak_bytes: int

    @property
    def estimated_peak_gib(self) -> float:
        return float(self.estimated_peak_bytes) / _GIB

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["estimated_peak_gib"] = self.estimated_peak_gib
        return row


@dataclass(frozen=True)
class MemoryDecision:
    allowed: bool
    estimated_peak_bytes: int
    available_bytes: int | None
    usable_bytes: int | None
    reason: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryCheckpoint:
    stage: str
    elapsed_seconds: float
    free_bytes: int | None
    total_bytes: int | None
    pool_used_bytes: int | None
    pool_total_bytes: int | None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("free_bytes", "total_bytes", "pool_used_bytes", "pool_total_bytes"):
            value = row[key]
            row[key.replace("_bytes", "_gib")] = (
                float(value) / _GIB if value is not None else np.nan
            )
        return row


def _complex_itemsize(precision: str) -> int:
    return 8 if str(precision).lower() == "single" else 16


def _real_itemsize(precision: str) -> int:
    return 4 if str(precision).lower() == "single" else 8


def estimate_large_grid_memory(
    grid_size: int,
    *,
    precision: str = "double",
    include_adjoint: bool = True,
    adjoint_storage: str = "derived",
    rcs_batch_size: int = 64,
) -> MemoryEstimate:
    """Conservative memory model for one electrodynamic validation job.

    The estimate is deliberately more conservative than the persistent array
    footprint.  It includes the nine-block FFT kernel, material/problem arrays,
    simultaneous solver vectors required by TwoStep, a batched far-field phase
    matrix, and an empirical cuFFT/build allowance.  It is used only as an OOM
    guard; measured checkpoints remain the authoritative diagnostics.
    """

    n = int(grid_size)
    if n <= 0:
        raise ValueError("grid_size must be positive")
    cbytes = _complex_itemsize(precision)
    rbytes = _real_itemsize(precision)
    cells = n**3
    doubled = (2 * n) ** 3

    forward_kernel = 9 * doubled * cbytes
    if include_adjoint and adjoint_storage == "explicit":
        kernel_bytes = 2 * forward_kernel
    else:
        kernel_bytes = forward_kernel

    # contrast (9 components), incident/solution/right-hand-side fields, and
    # grid axes.  Coordinate meshes are not assumed to be retained.
    problem_bytes = (9 + 3 + 3 + 3) * cells * cbytes + 3 * n * rbytes

    # TwoStep is the most demanding supported solver.  Allow fourteen vector
    # fields and two padded FFT fields.  This also covers BiCGStab/SIM.
    solver_workspace = 14 * 3 * cells * cbytes + 2 * 3 * doubled * cbytes

    # Direct far-field batching stores phase(batch,cells) plus compact vectors.
    farfield_workspace = max(1, int(rcs_batch_size)) * cells * cbytes

    # Streamed kernel construction still needs scalar geometry and one spatial
    # block.  cuFFT plans/work areas are implementation dependent, so retain a
    # 35% allowance over the principal spectral/solver arrays plus 256 MiB.
    geometry_and_block = (6 * doubled * rbytes) + doubled * cbytes
    principal = kernel_bytes + problem_bytes + solver_workspace + geometry_and_block
    fft_allowance = int(0.35 * principal + 256 * 1024**2)

    estimated_peak = (
        kernel_bytes
        + problem_bytes
        + solver_workspace
        + farfield_workspace
        + geometry_and_block
        + fft_allowance
    )
    return MemoryEstimate(
        grid_size=n,
        precision=str(precision),
        include_adjoint=bool(include_adjoint),
        adjoint_storage=str(adjoint_storage),
        kernel_bytes=int(kernel_bytes),
        problem_bytes=int(problem_bytes),
        solver_workspace_bytes=int(solver_workspace),
        farfield_workspace_bytes=int(farfield_workspace),
        fft_workspace_allowance_bytes=int(fft_allowance),
        estimated_peak_bytes=int(estimated_peak),
    )


def memory_decision(
    estimate: MemoryEstimate,
    backend: em3d.Backend,
    policy: CudaMemoryPolicy,
) -> MemoryDecision:
    info = backend.memory_info()
    if info is None:
        return MemoryDecision(
            allowed=True,
            estimated_peak_bytes=estimate.estimated_peak_bytes,
            available_bytes=None,
            usable_bytes=None,
            reason="cpu_or_unreported",
        )
    free_bytes = int(info["free_bytes"])
    reserve = int(float(policy.reserve_gib) * _GIB)
    usable = max(0, int(free_bytes * float(policy.usable_fraction)) - reserve)
    allowed = estimate.estimated_peak_bytes <= usable
    return MemoryDecision(
        allowed=bool(allowed),
        estimated_peak_bytes=estimate.estimated_peak_bytes,
        available_bytes=free_bytes,
        usable_bytes=usable,
        reason="within_policy" if allowed else "estimated_peak_exceeds_policy",
    )


def clear_cuda_runtime_caches(backend: em3d.Backend) -> None:
    if backend.device != "cuda":
        return
    backend.synchronize()
    try:  # CuPy exposes the plan cache only when cupyx.scipy.fft is available.
        from cupyx.scipy.fft import get_plan_cache

        get_plan_cache().clear()
    except Exception:
        try:
            cache = backend.xp.fft.config.get_plan_cache()
            cache.clear()
        except Exception:
            pass
    backend.clear_memory_pool()


def memory_checkpoint(
    backend: em3d.Backend,
    stage: str,
    *,
    started_at: float,
) -> MemoryCheckpoint:
    backend.synchronize()
    info = backend.memory_info()
    if info is None:
        return MemoryCheckpoint(
            stage=str(stage),
            elapsed_seconds=float(perf_counter() - started_at),
            free_bytes=None,
            total_bytes=None,
            pool_used_bytes=None,
            pool_total_bytes=None,
        )
    return MemoryCheckpoint(
        stage=str(stage),
        elapsed_seconds=float(perf_counter() - started_at),
        free_bytes=int(info["free_bytes"]),
        total_bytes=int(info["total_bytes"]),
        pool_used_bytes=int(info["pool_used_bytes"]),
        pool_total_bytes=int(info["pool_total_bytes"]),
    )


def run_operator_memory_probe(
    built: BuiltSpectralCase,
    *,
    policy: CudaMemoryPolicy,
    include_adjoint: bool = True,
) -> tuple[PreparedEMKernel, tuple[MemoryCheckpoint, ...]]:
    """Build and exercise the operator while recording measured memory stages.

    The returned prepared kernel can be reused by the actual solver run.  The
    probe intentionally performs only one forward and one adjoint action; solver
    iteration probes are covered by the normal all-solver smoke profile.
    """

    backend = built.problem.backend
    if policy.clear_pool_before_case:
        clear_cuda_runtime_caches(backend)
    started = perf_counter()
    rows: list[MemoryCheckpoint] = [
        memory_checkpoint(backend, "before_case", started_at=started)
    ]
    rows.append(memory_checkpoint(backend, "after_problem", started_at=started))
    prepared = PreparedEMKernel.build(
        built.problem.grid,
        k=built.problem.k0,
        include_adjoint=include_adjoint,
        adjoint_storage=policy.adjoint_storage,
        build_strategy=policy.kernel_build_strategy,
    )
    rows.append(memory_checkpoint(backend, "after_kernel", started_at=started))
    operator = Operator(built.problem, prepared_kernel=prepared)
    vector = backend.zeros((3,) + built.problem.grid.N, kind="complex")
    vector[0] = 1.0
    forward = operator.matvec(vector)
    backend.synchronize()
    rows.append(memory_checkpoint(backend, "after_matvec", started_at=started))
    if include_adjoint:
        adjoint = operator.rmatvec(vector)
        backend.synchronize()
        rows.append(memory_checkpoint(backend, "after_rmatvec", started_at=started))
        del adjoint
    del forward, vector, operator
    rows.append(memory_checkpoint(backend, "after_probe_cleanup", started_at=started))
    return prepared, tuple(rows)


__all__ = [
    "MemoryCheckpoint",
    "MemoryDecision",
    "MemoryEstimate",
    "clear_cuda_runtime_caches",
    "estimate_large_grid_memory",
    "memory_checkpoint",
    "memory_decision",
    "run_operator_memory_probe",
]
