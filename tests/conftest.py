"""Shared pytest fixtures: backend parametrisation and GPU skip logic."""
from __future__ import annotations

import pytest


def _cupy_available() -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return bool(cp.cuda.is_available())
    except Exception:
        return False


CUPY_AVAILABLE = _cupy_available()


def pytest_collection_modifyitems(config, items):
    if CUPY_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="cupy / CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(params=["numpy-double", "numpy-single"])
def backend_cpu(request):
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    precision = Precision.DOUBLE if request.param.endswith("double") else Precision.SINGLE
    return Backend.numpy(precision=precision)


@pytest.fixture
def backend_numpy_double():
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    return Backend.numpy(precision=Precision.DOUBLE)


@pytest.fixture
def backend_numpy_single():
    from em3d.backend import Backend
    from em3d.dtypes import Precision

    return Backend.numpy(precision=Precision.SINGLE)
