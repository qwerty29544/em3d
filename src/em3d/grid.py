"""Structured 3D Cartesian grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .backend import Backend


def _axis(n: int, length: float, centre: float, be: Backend):
    step = length / n
    start = centre - length / 2 + step / 2
    return be.xp.linspace(start, start + step * (n - 1), n, dtype=be.real_dtype)


@dataclass(frozen=True)
class Grid:
    N: Tuple[int, int, int]
    L: Tuple[float, float, float]
    center: Tuple[float, float, float]
    backend: Backend
    _x: object = field(init=False, repr=False)
    _y: object = field(init=False, repr=False)
    _z: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if any(n <= 0 for n in self.N):
            raise ValueError(f"Grid.N must be strictly positive, got {self.N}")
        if any(l <= 0 for l in self.L):
            raise ValueError(f"Grid.L must be strictly positive, got {self.L}")
        object.__setattr__(self, "_x", _axis(self.N[0], self.L[0], self.center[0], self.backend))
        object.__setattr__(self, "_y", _axis(self.N[1], self.L[1], self.center[1], self.backend))
        object.__setattr__(self, "_z", _axis(self.N[2], self.L[2], self.center[2], self.backend))

    @property
    def dv(self) -> float:
        return (self.L[0] / self.N[0]) * (self.L[1] / self.N[1]) * (self.L[2] / self.N[2])

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def coords(self):
        return self.backend.xp.meshgrid(self._x, self._y, self._z, indexing="ij")
