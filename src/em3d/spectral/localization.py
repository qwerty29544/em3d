from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np

from .complex_geometry import (
    HullConfig,
    convex_hull_complex,
    point_in_convex_polygon,
    point_polygon_distance,
)


class LocalizationStatus(str, Enum):
    OK = "ok"
    ORIGIN_IN_HULL = "origin_in_hull"
    DEGENERATE_SPECTRUM = "degenerate_spectrum"
    CIRCLE_NOT_FOUND = "circle_not_found"


@dataclass(frozen=True)
class CircleLocalization:
    mu: complex
    radius: float
    q: float
    margin: float


@dataclass(frozen=True)
class SpectrumLocalization:
    spectrum: np.ndarray
    hull: np.ndarray
    origin_in_hull: bool
    origin_distance: float
    status: LocalizationStatus
    circle: CircleLocalization | None

    @property
    def parameter_exists(self) -> bool:
        return self.status is LocalizationStatus.OK and self.circle is not None


def circle_from_two_points(z1: complex, z2: complex) -> tuple[complex, float]:
    product = complex(z1) * np.conj(complex(z2))
    denominator = 2.0 * (abs(product) + product.real)
    if abs(denominator) < 1e-30:
        raise ValueError("two-point localization circle is undefined")
    midpoint = 0.5 * (complex(z1) + complex(z2))
    correction = 1j * product.imag * (complex(z2) - complex(z1)) / denominator
    mu = complex(midpoint + correction)

    product_radius = np.conj(complex(z1)) * complex(z2)
    denominator_radius = 2.0 * (abs(product_radius) + product_radius.real)
    if abs(denominator_radius) < 1e-30:
        raise ValueError("two-point localization radius is undefined")
    value = abs(complex(z1) - complex(z2)) ** 2 * abs(product_radius) / denominator_radius
    return mu, float(np.sqrt(max(float(np.real(value)), 0.0)))


def circle_from_three_points(
    z1: complex,
    z2: complex,
    z3: complex,
) -> tuple[complex, float]:
    ax, ay = complex(z1).real, complex(z1).imag
    bx, by = complex(z2).real, complex(z2).imag
    cx, cy = complex(z3).real, complex(z3).imag
    denominator = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(denominator) < 1e-30:
        raise ValueError("three points are collinear")
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / denominator
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / denominator
    mu = complex(ux, uy)
    return mu, float(abs(mu - complex(z1)))


def circle_contains_points(
    mu: complex,
    radius: float,
    points: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> bool:
    values = np.asarray(points, dtype=np.complex128).reshape(-1)
    return bool(np.all(np.abs(values - complex(mu)) <= float(radius) + tolerance))


def circle_contains_origin(
    mu: complex,
    radius: float,
    *,
    tolerance: float = 1e-9,
) -> bool:
    return abs(complex(mu)) <= float(radius) + tolerance


def find_optimal_circle(
    hull: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> CircleLocalization:
    values = np.asarray(hull, dtype=np.complex128).reshape(-1)
    candidates: list[CircleLocalization] = []

    def add(mu: complex, radius: float) -> None:
        if not circle_contains_points(mu, radius, values, tolerance=tolerance):
            return
        if circle_contains_origin(mu, radius, tolerance=tolerance):
            return
        modulus = abs(mu)
        if modulus <= tolerance:
            return
        q = float(radius / modulus)
        candidates.append(
            CircleLocalization(
                mu=complex(mu),
                radius=float(radius),
                q=q,
                margin=float(modulus - radius),
            )
        )

    for z1, z2 in combinations(values, 2):
        try:
            add(*circle_from_two_points(z1, z2))
        except ValueError:
            pass
    for z1, z2, z3 in combinations(values, 3):
        try:
            add(*circle_from_three_points(z1, z2, z3))
        except ValueError:
            pass

    if not candidates:
        raise ValueError("no admissible localization circle excludes the origin")
    return min(candidates, key=lambda item: item.q)


def spectral_factor(spectrum: np.ndarray, mu: complex) -> float:
    values = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
    if len(values) == 0:
        raise ValueError("spectrum must be non-empty")
    if abs(mu) == 0.0:
        return float("inf")
    return float(np.max(np.abs(1.0 - values / complex(mu))))


def analyze_spectrum(
    spectrum: np.ndarray,
    *,
    hull_config: HullConfig = HullConfig(),
) -> SpectrumLocalization:
    values = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
    try:
        hull = convex_hull_complex(values, config=hull_config)
    except ValueError:
        return SpectrumLocalization(
            spectrum=values,
            hull=np.asarray(values, dtype=np.complex128),
            origin_in_hull=False,
            origin_distance=float("nan"),
            status=LocalizationStatus.DEGENERATE_SPECTRUM,
            circle=None,
        )

    origin_in_hull = point_in_convex_polygon(
        0.0 + 0.0j,
        hull,
        tolerance=hull_config.containment_tolerance,
    )
    distance = point_polygon_distance(0.0 + 0.0j, hull)
    if origin_in_hull:
        return SpectrumLocalization(
            spectrum=values,
            hull=hull,
            origin_in_hull=True,
            origin_distance=0.0,
            status=LocalizationStatus.ORIGIN_IN_HULL,
            circle=None,
        )

    try:
        circle = find_optimal_circle(
            hull,
            tolerance=hull_config.containment_tolerance,
        )
    except ValueError:
        return SpectrumLocalization(
            spectrum=values,
            hull=hull,
            origin_in_hull=False,
            origin_distance=float(distance),
            status=LocalizationStatus.CIRCLE_NOT_FOUND,
            circle=None,
        )

    return SpectrumLocalization(
        spectrum=values,
        hull=hull,
        origin_in_hull=False,
        origin_distance=float(distance),
        status=LocalizationStatus.OK,
        circle=circle,
    )
