from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class ConvergenceClass(str, Enum):
    CONVERGED = "converged"
    CONTRACTING_UNRESOLVED = "contracting_unresolved"
    UNSTABLE = "unstable"
    UNRESOLVED = "unresolved"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    classification: ConvergenceClass
    final_residual: float
    minimum_residual: float
    minimum_residual_iteration: int
    asymptotic_ratio: float
    observations_used: int


def estimate_asymptotic_ratio(
    history: Sequence[float],
    *,
    rtol: float,
    maximum_tail: int = 60,
) -> tuple[float, int]:
    residuals = np.asarray(history, dtype=np.float64)
    if len(residuals) < 8:
        return float("nan"), 0
    ratios = residuals[1:] / residuals[:-1]
    valid = (
        np.isfinite(ratios)
        & (ratios > 0.0)
        & np.isfinite(residuals[:-1])
        & (residuals[:-1] > max(10.0 * float(rtol), 1e-14))
    )
    ratios = ratios[valid]
    if len(ratios) < 4:
        return float("nan"), int(len(ratios))
    ratios = ratios[len(ratios) // 3 :]
    if len(ratios) > maximum_tail:
        ratios = ratios[-maximum_tail:]
    return float(np.median(ratios)), int(len(ratios))


def analyze_residual_history(
    history: Sequence[float],
    *,
    rtol: float,
    ratio_tolerance: float = 5e-3,
) -> ConvergenceDiagnostics:
    values = np.asarray(history, dtype=np.float64)
    if len(values) == 0:
        return ConvergenceDiagnostics(
            classification=ConvergenceClass.UNRESOLVED,
            final_residual=float("nan"),
            minimum_residual=float("nan"),
            minimum_residual_iteration=0,
            asymptotic_ratio=float("nan"),
            observations_used=0,
        )
    finite = np.isfinite(values)
    minimum_index = int(np.nanargmin(values)) if np.any(finite) else 0
    minimum = float(values[minimum_index]) if np.any(finite) else float("nan")
    final = float(values[-1])
    ratio, observations = estimate_asymptotic_ratio(values, rtol=rtol)

    if not np.all(finite):
        classification = ConvergenceClass.NONFINITE
    elif minimum <= float(rtol):
        classification = ConvergenceClass.CONVERGED
    elif np.isfinite(ratio) and ratio < 1.0 - ratio_tolerance:
        classification = ConvergenceClass.CONTRACTING_UNRESOLVED
    elif np.isfinite(ratio) and ratio > 1.0 + ratio_tolerance:
        classification = ConvergenceClass.UNSTABLE
    else:
        classification = ConvergenceClass.UNRESOLVED

    return ConvergenceDiagnostics(
        classification=classification,
        final_residual=final,
        minimum_residual=minimum,
        minimum_residual_iteration=minimum_index,
        asymptotic_ratio=float(ratio),
        observations_used=observations,
    )
