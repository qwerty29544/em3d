from __future__ import annotations
 
from dataclasses import dataclass
from typing import Any, Literal
 
import numpy as np
 
from .dtypes import Precision
 
@dataclass(frozen=True)
class Backend:
    xp: Any
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
        import cupy as cp
 
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
        return arr.get()
 
    def fftn(self, x, axes=None):
        return self.xp.fft.fftn(x, axes=axes)
 
    def ifftn(self, x, axes=None):
        return self.xp.fft.ifftn(x, axes=axes)
 
    def asarray_of_kind(self, obj, kind: Literal["real", "complex"]):
        dtype = self.real_dtype if kind == "real" else self.complex_dtype
        return self.xp.asarray(obj, dtype=dtype)

    def synchronize(self) -> None:
        """Synchronize the active device when asynchronous CUDA work is used."""

        if self.device == "cuda":
            self.xp.cuda.Stream.null.synchronize()

    def memory_info(self) -> dict[str, int | str] | None:
        """Return lightweight CUDA memory telemetry for manifests and notebooks."""

        if self.device != "cuda":
            return None
        device = self.xp.cuda.Device()
        free_bytes, total_bytes = device.mem_info
        pool = self.xp.get_default_memory_pool()
        pinned_pool = self.xp.get_default_pinned_memory_pool()
        properties = self.xp.cuda.runtime.getDeviceProperties(device.id)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return {
            "device_id": int(device.id),
            "device_name": str(name),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
            "pinned_pool_free_blocks": int(pinned_pool.n_free_blocks()),
        }

    def clear_memory_pool(self) -> None:
        """Release cached CuPy blocks after a completed experiment stage."""

        if self.device != "cuda":
            return
        self.synchronize()
        self.xp.get_default_memory_pool().free_all_blocks()
        self.xp.get_default_pinned_memory_pool().free_all_blocks()
