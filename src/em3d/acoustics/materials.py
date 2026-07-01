"""Scalar eta field generators for acoustic scattering."""
from __future__ import annotations

from ..grid import Grid


def _as_complex(be, value):
    return be.complex_dtype(value)


def eta_homogeneous(grid: Grid, eta_value):
    """Return scalar eta field with a constant value."""
    be = grid.backend
    out = be.zeros(grid.N, kind="complex")
    out[...] = _as_complex(be, eta_value)
    return out


def eta_slab(
    grid: Grid,
    *,
    eta_inside,
    eta_outside=1.0,
    axis: int = 0,
    width_fraction: float = 0.5,
):
    """Return eta field with a centered slab along one coordinate axis."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if not (0.0 < float(width_fraction) <= 1.0):
        raise ValueError(f"width_fraction must be in (0, 1], got {width_fraction!r}")

    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    coords = (X, Y, Z)
    half_width = 0.5 * float(grid.L[axis]) * float(width_fraction)
    centre = float(grid.center[axis])
    mask = xp.abs(coords[axis] - centre) <= half_width
    out = xp.where(mask, _as_complex(be, eta_inside), _as_complex(be, eta_outside))
    return out.astype(be.complex_dtype, copy=False)


def eta_sphere(
    grid: Grid,
    *,
    center: tuple[float, float, float],
    radius: float,
    eta_inside,
    eta_outside=1.0,
):
    """Return eta field with a spherical inclusion."""
    return eta_ellipsoid(
        grid,
        center=center,
        radii=(float(radius), float(radius), float(radius)),
        eta_inside=eta_inside,
        eta_outside=eta_outside,
    )


def eta_ellipsoid(
    grid: Grid,
    *,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    eta_inside,
    eta_outside=1.0,
):
    """Return eta field with an axis-aligned ellipsoidal inclusion."""
    if len(center) != 3:
        raise ValueError(f"center must have length 3, got {center!r}")
    if len(radii) != 3 or any(float(r) <= 0.0 for r in radii):
        raise ValueError(f"radii must contain three positive values, got {radii!r}")

    be = grid.backend
    xp = be.xp
    X, Y, Z = grid.coords()
    cx, cy, cz = (float(x) for x in center)
    rx, ry, rz = (float(x) for x in radii)
    metric = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    out = xp.where(metric <= 1.0, _as_complex(be, eta_inside), _as_complex(be, eta_outside))
    return out.astype(be.complex_dtype, copy=False)
