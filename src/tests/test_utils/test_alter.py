import numpy as np
import pyarrow as pa

from tschm.flight.utils.alter import np_2_pa, pa_2_np


def test_conversion_numpy_to_pyarrow():
    data = {"a": 2, "b": np.array([3, 4, 5]), "c": np.random.randn(5, 4), "d": np.array(["a", "b", "c"])}

    table = np_2_pa(data)

    assert isinstance(table, pa.Table)

    recover = pa_2_np(table)

    assert np.allclose(recover["a"], data["a"])
    assert np.allclose(recover["b"], data["b"])
    assert np.allclose(recover["c"], data["c"])
    assert np.array_equal(recover["d"], data["d"])


def test_conversion_pyarrow_to_numpy():
    table = pa.Table.from_pydict(
        {
            "values": pa.array([{"data": [1, 2, 3, 4], "shape": [2, 2]}]),
            "labels": pa.array([{"data": ["a", "b", "c", "d"], "shape": [4]}]),
        }
    )

    # print the column name

    a = pa_2_np(table)
    np.array_equal(a["values"], np.array([[1, 2], [3, 4]]))
    np.array_equal(a["labels"], np.array(["a", "b", "c", "d"]))
