import numpy as np
import pytest

from em3d.layout import field_size, flatten_field, unflatten_field


def test_flatten_unflatten_field_roundtrip():
    field = np.arange(3 * 2 * 3 * 4).reshape(3, 2, 3, 4)
    vector = flatten_field(field)
    assert vector.shape == (72,)
    np.testing.assert_array_equal(unflatten_field(vector, (2, 3, 4)), field)


def test_flatten_field_is_cell_major_component_minor():
    field = np.zeros((3, 1, 1, 2), dtype=np.int64)
    field[:, 0, 0, 0] = [1, 2, 3]
    field[:, 0, 0, 1] = [4, 5, 6]
    np.testing.assert_array_equal(flatten_field(field), [1, 2, 3, 4, 5, 6])


def test_layout_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        flatten_field(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError):
        unflatten_field(np.zeros(3), (2, 2, 2))
    assert field_size((2, 3, 4)) == 72
