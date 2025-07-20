"""Tests for the alter module in the flight.utils package.

This module contains tests for the np_2_pa and pa_2_np functions, which convert
between NumPy arrays and PyArrow Tables while preserving array shapes and data types.
"""

import numpy as np
import pyarrow as pa

from flight.utils.alter import np_2_pa, pa_2_np


def test_conversion_numpy_to_pyarrow() -> None:
    """Test conversion from NumPy arrays to PyArrow Table and back.

    This test verifies that:
    1. NumPy arrays of different types and shapes can be converted to a PyArrow Table
    2. The resulting Table can be converted back to NumPy arrays
    3. The original data is preserved through the round-trip conversion
    """
    data = {"a": 2, "b": np.array([3, 4, 5]), "c": np.random.randn(5, 4), "d": np.array(["a", "b", "c"])}

    table = np_2_pa(data)

    assert isinstance(table, pa.Table)

    recover = pa_2_np(table)

    assert np.allclose(recover["a"], data["a"])
    assert np.allclose(recover["b"], data["b"])
    assert np.allclose(recover["c"], data["c"])
    assert np.array_equal(recover["d"], data["d"])


def test_conversion_pyarrow_to_numpy() -> None:
    """Test conversion from PyArrow Table to NumPy arrays.

    This test verifies that a PyArrow Table with structured arrays containing
    'data' and 'shape' fields can be correctly converted to NumPy arrays with
    the original shapes restored.
    """
    table = pa.Table.from_pydict(
        {
            "values": pa.array([{"data": [1, 2, 3, 4], "shape": [2, 2]}]),
            "labels": pa.array([{"data": ["a", "b", "c", "d"], "shape": [4]}]),
        }
    )

    # Convert the table to NumPy arrays
    a = pa_2_np(table)

    # Verify the conversion results
    assert np.array_equal(a["values"], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(a["labels"], np.array(["a", "b", "c", "d"]))
