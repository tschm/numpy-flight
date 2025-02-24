"""global fixtures"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest


@pytest.fixture(name="root_dir")
def root_fixture() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def test_data():
    """
    Create sample test data as a dictionary of NumPy arrays.
    """
    return {"values": np.array([1, 2, 3, 4, 5]), "labels": np.array(["a", "b", "c", "d", "e"])}


@pytest.fixture
def test_table():
    """
    Create a sample Arrow table for testing.
    """
    arrays = [
        pa.array([{"data": [1, 2, 3, 4, 5], "shape": [5]}]),
        pa.array([{"data": ["a", "b", "c", "d", "e"], "shape": [5]}]),
    ]
    return pa.Table.from_arrays(arrays, names=["values", "labels"])
