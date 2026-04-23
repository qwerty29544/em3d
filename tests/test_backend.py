import numpy as np
import pytest

from em3d.backend import Backend
from em3d.dtypes import Precision


def test_precision_dtype_mapping():
    assert Precision.DOUBLE.real_dtype is np.float64
    assert Precision.DOUBLE.complex_dtype is np.complex128
    assert Precision.SINGLE.real_dtype is np.float32
    assert Precision.SINGLE.complex_dtype is np.complex64


def test_backend_numpy_double():
    be = Backend.numpy(precision=Precision.DOUBLE)
    assert be.device == "cpu"
    assert be.real_dtype is np.float64
    assert be.complex_dtype is np.complex128
    assert be.xp is np


def test_backend_numpy_single():
    be = Backend.numpy(precision=Precision.SINGLE)
    assert be.real_dtype is np.float32
    assert be.complex_dtype is np.complex64


def test_backend_zeros_real_and_complex():
    be = Backend.numpy(precision=Precision.DOUBLE)
    r = be.zeros((2, 3), kind="real")
    c = be.zeros((2, 3), kind="complex")
    assert r.dtype == np.float64 and r.shape == (2, 3)
    assert c.dtype == np.complex128 and c.shape == (2, 3)


def test_backend_zeros_invalid_kind():
    be = Backend.numpy(precision=Precision.DOUBLE)
    with pytest.raises(ValueError):
        be.zeros((2,), kind="bogus")


def test_backend_to_host_numpy_roundtrip():
    be = Backend.numpy(precision=Precision.DOUBLE)
    arr = be.array([1.0, 2.0, 3.0], dtype=np.float64)
    host = be.to_host(arr)
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, [1.0, 2.0, 3.0])


def test_backend_auto_returns_numpy_when_cupy_absent():
    be = Backend.auto(precision=Precision.DOUBLE)
    assert be.device in ("cpu", "cuda")
    assert be.precision is Precision.DOUBLE


def test_backend_fftn_roundtrip():
    be = Backend.numpy(precision=Precision.DOUBLE)
    x = be.array(np.random.default_rng(0).standard_normal((4, 4, 4)).astype(np.complex128))
    y = be.ifftn(be.fftn(x))
    np.testing.assert_allclose(y, x, atol=1e-12)
