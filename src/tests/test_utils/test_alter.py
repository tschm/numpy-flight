import numpy as np
import pyarrow as pa

from np.client.utils.alter import np_2_pa, pa_2_np


def test_conversion_numpy_vs_pyarrow():
    data = {"a": 2, "b": np.array([3, 4, 5]), "c": np.random.randn(5, 4)}
    table = np_2_pa(data)

    assert isinstance(table, pa.Table)

    recover = pa_2_np(table)

    assert np.allclose(recover["a"], data["a"])
    assert np.allclose(recover["b"], data["b"])
    assert np.allclose(recover["c"], data["c"])
