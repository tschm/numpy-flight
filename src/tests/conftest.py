#    Copyright 2025 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""global fixtures"""

import numpy as np
import pyarrow as pa
import pytest


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
