"""Problem: Grid + ε-tensor + incident wave + k₀ + volume of Q."""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid


@dataclass(frozen=True)
class Problem:
    grid: Grid
    eps_tensor: object  # shape (3, 3) + grid.N, complex
    wave: object        # shape (3,) + grid.N, complex
    k0: float
    volume: float

    def __post_init__(self) -> None:
        be = self.grid.backend
        expected_eta = (3, 3) + self.grid.N
        expected_wave = (3,) + self.grid.N
        if self.eps_tensor.shape != expected_eta:
            raise ValueError(
                f"eps_tensor.shape {self.eps_tensor.shape} != expected {expected_eta}"
            )
        if self.wave.shape != expected_wave:
            raise ValueError(
                f"wave.shape {self.wave.shape} != expected {expected_wave}"
            )
        if self.eps_tensor.dtype != be.complex_dtype:
            raise TypeError(
                f"eps_tensor.dtype {self.eps_tensor.dtype} != {be.complex_dtype}"
            )
        if self.wave.dtype != be.complex_dtype:
            raise TypeError(f"wave.dtype {self.wave.dtype} != {be.complex_dtype}")

    @property
    def backend(self):
        return self.grid.backend
