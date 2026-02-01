"""Tests for the numpy-flight package.

This module tests the Client, Server, and utility functions for converting
between NumPy arrays and PyArrow Tables.
"""

import logging
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.flight as fl
import pytest

from flight import Client, Server
from flight.utils.alter import np_2_pa, pa_2_np


class TestNp2Pa:
    """Tests for the np_2_pa function."""

    def test_single_1d_array(self):
        """Test converting a single 1D array."""
        data = {"arr": np.array([1, 2, 3])}
        table = np_2_pa(data)
        assert table.num_rows == 1
        assert table.num_columns == 1
        assert "arr" in table.column_names

    def test_single_2d_array(self):
        """Test converting a 2D matrix."""
        data = {"matrix": np.array([[1, 2], [3, 4]])}
        table = np_2_pa(data)
        assert table.num_rows == 1
        assert "matrix" in table.column_names

    def test_multiple_arrays(self):
        """Test converting multiple arrays of different shapes."""
        data = {
            "vector": np.array([1, 2, 3]),
            "matrix": np.array([[1, 2], [3, 4]]),
            "scalar": np.array(42),
        }
        table = np_2_pa(data)
        assert table.num_columns == 3
        assert set(table.column_names) == {"vector", "matrix", "scalar"}

    def test_float_array(self):
        """Test converting float arrays."""
        data = {"floats": np.array([1.5, 2.5, 3.5])}
        table = np_2_pa(data)
        assert table.num_rows == 1

    def test_none_values_filtered(self):
        """Test that None values are filtered out."""
        data = {"arr": np.array([1, 2]), "none_arr": None}
        table = np_2_pa(data)
        assert table.num_columns == 1
        assert "arr" in table.column_names
        assert "none_arr" not in table.column_names

    def test_empty_dict(self):
        """Test converting an empty dictionary."""
        data = {}
        table = np_2_pa(data)
        assert table.num_rows == 0
        assert table.num_columns == 0


class TestPa2Np:
    """Tests for the pa_2_np function."""

    def test_roundtrip_1d_array(self):
        """Test that 1D arrays survive roundtrip conversion."""
        original = {"arr": np.array([1, 2, 3, 4, 5])}
        table = np_2_pa(original)
        result = pa_2_np(table)
        np.testing.assert_array_equal(result["arr"], original["arr"])

    def test_roundtrip_2d_array(self):
        """Test that 2D arrays survive roundtrip conversion."""
        original = {"matrix": np.array([[1, 2, 3], [4, 5, 6]])}
        table = np_2_pa(original)
        result = pa_2_np(table)
        np.testing.assert_array_equal(result["matrix"], original["matrix"])

    def test_roundtrip_3d_array(self):
        """Test that 3D arrays survive roundtrip conversion."""
        original = {"tensor": np.arange(24).reshape(2, 3, 4)}
        table = np_2_pa(original)
        result = pa_2_np(table)
        np.testing.assert_array_equal(result["tensor"], original["tensor"])

    def test_roundtrip_scalar(self):
        """Test that scalar arrays survive roundtrip conversion."""
        original = {"scalar": np.array(42)}
        table = np_2_pa(original)
        result = pa_2_np(table)
        np.testing.assert_array_equal(result["scalar"], original["scalar"])

    def test_roundtrip_multiple_arrays(self):
        """Test roundtrip with multiple arrays."""
        original = {
            "a": np.array([1, 2, 3]),
            "b": np.array([[1, 2], [3, 4]]),
        }
        table = np_2_pa(original)
        result = pa_2_np(table)
        for key in original:
            np.testing.assert_array_equal(result[key], original[key])

    def test_roundtrip_float_array(self):
        """Test roundtrip with float arrays."""
        original = {"floats": np.array([1.1, 2.2, 3.3])}
        table = np_2_pa(original)
        result = pa_2_np(table)
        np.testing.assert_array_almost_equal(result["floats"], original["floats"])


class TestClientDescriptor:
    """Tests for the Client.descriptor static method."""

    def test_descriptor_returns_flight_descriptor(self):
        """Test that descriptor returns a FlightDescriptor."""
        desc = Client.descriptor("test_command")
        assert isinstance(desc, fl.FlightDescriptor)

    def test_descriptor_encodes_command(self):
        """Test that the command is properly encoded."""
        command = "my_command"
        desc = Client.descriptor(command)
        assert desc.command == command.encode()


class TestClientInit:
    """Tests for Client initialization."""

    def test_client_init_stores_location(self):
        """Test that Client stores the location."""
        location = "grpc://localhost:8080"
        client = Client(location)
        assert client._location == location

    def test_client_init_stores_kwargs(self):
        """Test that Client stores additional kwargs."""
        location = "grpc://localhost:8080"
        client = Client(location, timeout=30)
        assert client._kwargs == {"timeout": 30}


class EchoServer(Server):
    """A simple echo server for testing that returns the input data."""

    def f(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return the input matrices unchanged."""
        return matrices


class DoubleServer(Server):
    """A server that doubles all input values."""

    def f(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Double all values in the input matrices."""
        return {key: value * 2 for key, value in matrices.items()}


class TestServerInit:
    """Tests for Server initialization."""

    def test_server_init_default_logger(self):
        """Test that Server creates a default logger."""
        server = EchoServer()
        assert server.logger is not None
        assert isinstance(server.logger, logging.Logger)
        server.shutdown()

    def test_server_init_custom_logger(self):
        """Test that Server uses a custom logger when provided."""
        custom_logger = logging.getLogger("custom")
        server = EchoServer(logger=custom_logger)
        assert server.logger is custom_logger
        server.shutdown()

    def test_server_init_custom_host_port(self):
        """Test Server initialization with custom host and port."""
        server = EchoServer(host="0.0.0.0", port=9090)
        assert server.logger is not None
        server.shutdown()


class TestServerExtractCommand:
    """Tests for Server._extract_command_from_ticket."""

    def test_extract_command(self):
        """Test extracting command from ticket."""
        server = EchoServer()
        ticket = fl.Ticket("test_command")
        command = server._extract_command_from_ticket(ticket)
        assert command == "test_command"
        server.shutdown()


class TestClientServerIntegration:
    """Integration tests for Client and Server working together."""

    @pytest.fixture
    def server_and_client(self):
        """Create a server and client for testing."""
        port = 18815  # Use a unique port for testing
        server = EchoServer(port=port)
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(0.5)  # Give server time to start

        location = f"grpc://127.0.0.1:{port}"
        yield server, location

        server.shutdown()

    def test_client_context_manager(self, server_and_client):
        """Test using Client as a context manager."""
        _, location = server_and_client
        with Client(location) as client:
            assert client.flight is not None

    def test_client_flight_property(self, server_and_client):
        """Test that flight property returns FlightClient."""
        _, location = server_and_client
        with Client(location) as client:
            assert isinstance(client.flight, fl.FlightClient)

    def test_write_and_get(self, server_and_client):
        """Test writing and getting data."""
        _, location = server_and_client
        with Client(location) as client:
            data = {"arr": np.array([1, 2, 3, 4, 5])}
            client.write("test", data)
            result = client.get("test")
            assert isinstance(result, pa.Table)

    def test_compute_roundtrip(self, server_and_client):
        """Test the compute method for roundtrip."""
        _, location = server_and_client
        with Client(location) as client:
            data = {"arr": np.array([1, 2, 3])}
            result = client.compute("test", data)
            np.testing.assert_array_equal(result["arr"], data["arr"])

    def test_compute_2d_array(self, server_and_client):
        """Test compute with 2D arrays."""
        _, location = server_and_client
        with Client(location) as client:
            data = {"matrix": np.array([[1, 2], [3, 4]])}
            result = client.compute("test", data)
            np.testing.assert_array_equal(result["matrix"], data["matrix"])

    def test_compute_multiple_arrays(self, server_and_client):
        """Test compute with multiple arrays."""
        _, location = server_and_client
        with Client(location) as client:
            data = {
                "a": np.array([1, 2, 3]),
                "b": np.array([[1, 2], [3, 4]]),
            }
            result = client.compute("test", data)
            for key in data:
                np.testing.assert_array_equal(result[key], data[key])


class TestClientWriteValidation:
    """Tests for Client.write validation."""

    @pytest.fixture
    def server_and_client(self):
        """Create a server and client for testing."""
        port = 18816
        server = EchoServer(port=port)
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        location = f"grpc://127.0.0.1:{port}"
        yield server, location

        server.shutdown()

    def test_write_empty_dict_raises_value_error(self, server_and_client):
        """Test that write raises ValueError for empty data."""
        _, location = server_and_client
        with Client(location) as client:
            with pytest.raises(ValueError, match="Empty data"):
                client.write("test", {})

    def test_write_empty_array_succeeds(self, server_and_client):
        """Test that write with empty array succeeds (creates table with 1 row)."""
        _, location = server_and_client
        with Client(location) as client:
            # An empty array creates a table with 1 row containing empty data
            # This is valid behavior - the table has num_rows=1
            client.write("test", {"arr": np.array([])})


class TestDoubleServerIntegration:
    """Integration tests with DoubleServer."""

    @pytest.fixture
    def server_and_client(self):
        """Create a DoubleServer and client for testing."""
        port = 18817
        server = DoubleServer(port=port)
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        location = f"grpc://127.0.0.1:{port}"
        yield server, location

        server.shutdown()

    def test_compute_doubles_values(self, server_and_client):
        """Test that DoubleServer doubles input values."""
        _, location = server_and_client
        with Client(location) as client:
            data = {"arr": np.array([1, 2, 3])}
            result = client.compute("test", data)
            np.testing.assert_array_equal(result["arr"], np.array([2, 4, 6]))

    def test_compute_doubles_matrix(self, server_and_client):
        """Test that DoubleServer doubles matrix values."""
        _, location = server_and_client
        with Client(location) as client:
            data = {"matrix": np.array([[1, 2], [3, 4]])}
            result = client.compute("test", data)
            expected = np.array([[2, 4], [6, 8]])
            np.testing.assert_array_equal(result["matrix"], expected)


class TestServerDoGetError:
    """Tests for Server.do_get error handling."""

    @pytest.fixture
    def server_and_client(self):
        """Create a server and client for testing."""
        port = 18818
        server = EchoServer(port=port)
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        location = f"grpc://127.0.0.1:{port}"
        yield server, location

        server.shutdown()

    def test_get_nonexistent_command_raises_error(self, server_and_client):
        """Test that getting non-existent data raises an error."""
        _, location = server_and_client
        with Client(location) as client:
            with pytest.raises(fl.FlightServerError):
                client.get("nonexistent_command")


class TestServerDoPutDirect:
    """Direct tests for Server.do_put method."""

    def test_do_put_stores_data(self):
        """Test that do_put stores data in _storage."""
        server = EchoServer(port=18819)

        # Create test data
        data = {"arr": np.array([1, 2, 3])}
        table = np_2_pa(data)

        # Create a mock reader that returns our table
        class MockReader:
            def read_all(self):
                return table

        # Create a mock descriptor
        class MockDescriptor:
            command = b"test_cmd"

        # Call do_put directly
        result = server.do_put(None, MockDescriptor(), MockReader(), None)

        # Verify data was stored
        assert "test_cmd" in server._storage
        assert result is not None
        server.shutdown()

    def test_do_put_multiple_commands(self):
        """Test that do_put handles multiple commands."""
        server = EchoServer(port=18820)

        data1 = {"a": np.array([1, 2])}
        data2 = {"b": np.array([3, 4])}
        table1 = np_2_pa(data1)
        table2 = np_2_pa(data2)

        class MockReader1:
            def read_all(self):
                return table1

        class MockReader2:
            def read_all(self):
                return table2

        class MockDescriptor1:
            command = b"cmd1"

        class MockDescriptor2:
            command = b"cmd2"

        server.do_put(None, MockDescriptor1(), MockReader1(), None)
        server.do_put(None, MockDescriptor2(), MockReader2(), None)

        assert "cmd1" in server._storage
        assert "cmd2" in server._storage
        server.shutdown()


class TestServerDoGetDirect:
    """Direct tests for Server.do_get method."""

    def test_do_get_retrieves_and_processes_data(self):
        """Test that do_get retrieves data and applies f()."""
        server = EchoServer(port=18821)

        # First store some data
        data = {"arr": np.array([1, 2, 3])}
        table = np_2_pa(data)
        server._storage["test_cmd"] = table

        # Create a mock ticket
        ticket = fl.Ticket(b"test_cmd")

        # Call do_get
        result = server.do_get(None, ticket)

        # Verify result is a RecordBatchStream
        assert isinstance(result, fl.RecordBatchStream)
        server.shutdown()

    def test_do_get_nonexistent_raises_error(self):
        """Test that do_get raises error for missing data."""
        server = EchoServer(port=18822)

        ticket = fl.Ticket(b"nonexistent")

        with pytest.raises(fl.FlightServerError):
            server.do_get(None, ticket)

        server.shutdown()


class TestDoubleServerDirect:
    """Direct tests for DoubleServer.f method."""

    def test_f_doubles_values(self):
        """Test that DoubleServer.f doubles values."""
        server = DoubleServer(port=18823)

        data = {"arr": np.array([1, 2, 3])}
        result = server.f(data)

        np.testing.assert_array_equal(result["arr"], np.array([2, 4, 6]))
        server.shutdown()

    def test_f_doubles_multiple_arrays(self):
        """Test that DoubleServer.f doubles multiple arrays."""
        server = DoubleServer(port=18824)

        data = {
            "a": np.array([1, 2]),
            "b": np.array([[1, 2], [3, 4]]),
        }
        result = server.f(data)

        np.testing.assert_array_equal(result["a"], np.array([2, 4]))
        np.testing.assert_array_equal(result["b"], np.array([[2, 4], [6, 8]]))
        server.shutdown()


class TestServerLogger:
    """Tests for Server logger property."""

    def test_logger_returns_configured_logger(self):
        """Test that logger property returns the configured logger."""
        custom_logger = logging.getLogger("test_logger")
        server = EchoServer(port=18825, logger=custom_logger)
        assert server.logger is custom_logger
        server.shutdown()

    def test_logger_returns_default_logger(self):
        """Test that logger property returns default logger."""
        server = EchoServer(port=18826)
        assert server.logger is not None
        assert server.logger.name == "flight.numpy_server"
        server.shutdown()
