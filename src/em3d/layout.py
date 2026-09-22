from __future__ import annotations

from math import prod
from typing import Any, Sequence


def _normalize_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    normalized = tuple(int(n) for n in shape)
    if len(normalized) != 3 or any(n <= 0 for n in normalized):
        raise ValueError(f"shape must be a positive 3-tuple, got {shape!r}")
    return normalized


def flatten_field(field: Any) -> Any:
    """Flatten ``(3, Nx, Ny, Nz)`` in cell-major/component-minor order.

    The returned ordering is ``(ix, iy, iz, component)``.  It is the ordering
    used by the dense block matrices in :mod:`em3d.dense`.
    """

    if getattr(field, "ndim", None) != 4 or field.shape[0] != 3:
        raise ValueError(
            "field must have shape (3, Nx, Ny, Nz), "
            f"got {getattr(field, 'shape', None)!r}"
        )
    return field.transpose(1, 2, 3, 0).reshape(-1)


def unflatten_field(vector: Any, shape: Sequence[int]) -> Any:
    """Inverse of :func:`flatten_field` for a structured three-dimensional grid."""

    Nx, Ny, Nz = _normalize_shape(shape)
    expected = 3 * prod((Nx, Ny, Nz))
    if getattr(vector, "ndim", None) != 1 or int(vector.size) != expected:
        raise ValueError(
            f"vector must be one-dimensional with {expected} entries, "
            f"got shape {getattr(vector, 'shape', None)!r}"
        )
    return vector.reshape(Nx, Ny, Nz, 3).transpose(3, 0, 1, 2)


def field_size(shape: Sequence[int]) -> int:
    """Number of complex scalar unknowns for a vector field on ``shape``."""

    return 3 * prod(_normalize_shape(shape))
