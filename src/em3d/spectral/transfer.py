from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .complex_geometry import directed_polygon_distance, polygon_hausdorff_distance
from .localization import (
    SpectrumLocalization,
    analyze_spectrum,
    spectral_factor,
)


@dataclass(frozen=True)
class TransferAssessment:
    directed_error: float
    hausdorff_distance: float
    coarse_factor: float
    coarse_margin: float
    normalized_error: float
    target_factor: float
    target_factor_bound: float
    certified: bool


@dataclass(frozen=True)
class EnsembleLocalization:
    levels: tuple[int, ...]
    localization: SpectrumLocalization
    adjacent_directed_errors: tuple[float, ...]
    adjacent_hausdorff_distances: tuple[float, ...]


def _require_circle(localization: SpectrumLocalization):
    if localization.circle is None:
        raise ValueError(
            f"localization has no admissible parameter: {localization.status.value}"
        )
    return localization.circle


def assess_transfer(
    coarse: SpectrumLocalization,
    target: SpectrumLocalization,
) -> TransferAssessment:
    """Assess transfer of the coarse localization parameter to ``target``."""

    circle = _require_circle(coarse)
    directed = directed_polygon_distance(target.hull, coarse.hull)
    hausdorff = polygon_hausdorff_distance(target.hull, coarse.hull)
    normalized = directed / circle.margin if circle.margin > 0.0 else float("inf")
    target_factor = spectral_factor(target.spectrum, circle.mu)
    bound = circle.q + directed / abs(circle.mu)
    return TransferAssessment(
        directed_error=float(directed),
        hausdorff_distance=float(hausdorff),
        coarse_factor=float(circle.q),
        coarse_margin=float(circle.margin),
        normalized_error=float(normalized),
        target_factor=float(target_factor),
        target_factor_bound=float(bound),
        certified=bool(directed < circle.margin),
    )


def build_ensemble_localization(
    localizations: Mapping[int, SpectrumLocalization],
    levels: tuple[int, ...] | None = None,
) -> EnsembleLocalization:
    if levels is None:
        levels = tuple(sorted(int(level) for level in localizations))
    else:
        levels = tuple(int(level) for level in levels)
    if not levels:
        raise ValueError("at least one localization is required")

    selected = [localizations[level] for level in levels]
    union_vertices = np.concatenate([item.hull for item in selected])
    ensemble = analyze_spectrum(union_vertices)

    directed_errors: list[float] = []
    hausdorff_distances: list[float] = []
    for previous, current in zip(selected, selected[1:]):
        directed_errors.append(
            directed_polygon_distance(current.hull, previous.hull)
        )
        hausdorff_distances.append(
            polygon_hausdorff_distance(current.hull, previous.hull)
        )

    return EnsembleLocalization(
        levels=levels,
        localization=ensemble,
        adjacent_directed_errors=tuple(float(value) for value in directed_errors),
        adjacent_hausdorff_distances=tuple(
            float(value) for value in hausdorff_distances
        ),
    )
