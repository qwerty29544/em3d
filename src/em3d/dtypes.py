"""Precision levels and dtype pairs for em3d."""
from __future__ import annotations

from enum import Enum

import numpy as np


class Precision(Enum):
    SINGLE = "single"
    DOUBLE = "double"

    @property
    def real_dtype(self) -> type:
        return np.float32 if self is Precision.SINGLE else np.float64

    @property
    def complex_dtype(self) -> type:
        return np.complex64 if self is Precision.SINGLE else np.complex128
