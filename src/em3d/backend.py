"""Array-namespace backend: selects numpy or cupy, carries precision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .dtypes import Precision


@dataclass(frozen=True)
class Backend:
    xp: Any  # module: numpy or cupy
    device: Literal["cpu", "cuda"]
    precision: Precision

    @property
    def real_dtype(self) -> type:
        return self.precision.real_dtype

    @property
    def complex_dtype(self) -> type:
        return self.precision.complex_dtype

    @classmethod
    def numpy(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
        return cls(xp=np, device="cpu", precision=precision)

    @classmethod
    def cupy(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
        import cupy as cp  # local import keeps cupy optional

        if not cp.cuda.is_available():
            raise RuntimeError("cupy imported but no CUDA device is available")
        return cls(xp=cp, device="cuda", precision=precision)

    @classmethod
    def auto(cls, precision: Precision = Precision.DOUBLE) -> "Backend":
        try:
            import cupy as cp

            if cp.cuda.is_available():
                return cls(xp=cp, device="cuda", precision=precision)
        except ImportError:
            pass
        return cls.numpy(precision=precision)

    def array(self, obj, dtype=None):
        return self.xp.asarray(obj, dtype=dtype)

    def zeros(self, shape, kind: Literal["real", "complex"]):
        if kind == "real":
            dtype = self.real_dtype
        elif kind == "complex":
            dtype = self.complex_dtype
        else:
            raise ValueError(f"kind must be 'real' or 'complex', got {kind!r}")
        return self.xp.zeros(shape, dtype=dtype)

    def empty(self, shape, kind: Literal["real", "complex"]):
        if kind == "real":
            dtype = self.real_dtype
        elif kind == "complex":
            dtype = self.complex_dtype
        else:
            raise ValueError(f"kind must be 'real' or 'complex', got {kind!r}")
        return self.xp.empty(shape, dtype=dtype)

    def to_host(self, arr) -> np.ndarray:
        if self.xp is np:
            return np.asarray(arr)
        return arr.get()  # cupy ndarray -> numpy

    def fftn(self, x, axes=None):
        return self.xp.fft.fftn(x, axes=axes)

    def ifftn(self, x, axes=None):
        return self.xp.fft.ifftn(x, axes=axes)

    def asarray_of_kind(self, obj, kind: Literal["real", "complex"]):
        dtype = self.real_dtype if kind == "real" else self.complex_dtype
        return self.xp.asarray(obj, dtype=dtype)
