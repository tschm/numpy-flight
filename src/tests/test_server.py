"""Tests for the Server class in the flight package.

This module contains tests for the Server class, which provides a server implementation
for handling NumPy array operations over Apache Arrow Flight. It tests the server's
ability to receive data, store it, process it, and return results.
"""

import concurrent.futures
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.flight as fl
import pytest
from loguru import logger

from flight import Server  # Adjust import path as needed


# Create a concrete implementation of the abstract Server class for testing
class TestServer(Server):
    """Test implementation of the Server abstract class.

    This class provides a simple implementation of the abstract Server class
    for testing purposes. It implements the required 'f' method to simply
    return the input matrices without modification.
    """

    def f(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process the input matrices by returning them unchanged.

        Args:
            matrices: A dictionary of NumPy arrays to process.

        Returns:
            The same dictionary of NumPy arrays, unmodified.
        """
        # Simple implementation for testing - just return the input as a table
        return matrices


@pytest.fixture(scope="module")
def server() -> TestServer:
    """Fixture providing a test server instance.

    Returns:
        TestServer: An instance of the TestServer class configured for testing.
    """
    return TestServer(host="localhost", port=5008, logger=logger)


@pytest.mark.parametrize("test_command", ["test_command1", "test_command2", "complex/command/path", "123456"])
def test_extract_command_from_ticket(server: TestServer, test_command: str) -> None:
    """Test command extraction from Flight ticket with various commands.

    This test verifies that the server can correctly extract command strings from
    Flight tickets with different formats and characters.

    Args:
        server: The test server instance.
        test_command: The command string to test.
    """
    ticket = fl.Ticket(test_command.encode())
    extracted_command = server._extract_command_from_ticket(ticket)
    assert extracted_command == test_command


def test_do_put(server: TestServer, test_table: pa.Table) -> None:
    """Test PUT operation with mock reader and writer.

    This test verifies that the server can correctly handle PUT requests by:
    1. Receiving data from a client
    2. Storing it in the server's storage
    3. Returning a proper descriptor

    Args:
        server: The test server instance.
        test_table: A sample PyArrow table for testing.
    """
    # Create test command and descriptor
    test_command = "test_put"
    descriptor = fl.FlightDescriptor.for_command(test_command)

    # Mock reader with sample data
    reader = Mock()
    reader.read_all.return_value = test_table

    # Execute PUT operation
    result = server.do_put(None, descriptor, reader, None)

    # Verify results
    assert test_command in server._storage
    assert pa.Table.equals(server._storage[test_command], test_table)
    assert result.command == descriptor.command


def test_do_get_existing_data(server: TestServer, test_table: pa.Table) -> None:
    """Test GET operation with existing data.

    This test verifies that the server can correctly handle GET requests for
    existing data by:
    1. Retrieving the data from storage
    2. Processing it through the computation function
    3. Returning a proper RecordBatchStream

    Args:
        server: The test server instance.
        test_table: A sample PyArrow table for testing.
    """
    # Store test data
    test_command = "test_get"
    server._storage[test_command] = test_table

    # Create ticket for GET request
    ticket = fl.Ticket(test_command.encode())

    # Execute GET operation
    result_stream = server.do_get(None, ticket)

    # Verify the result
    assert isinstance(result_stream, fl.RecordBatchStream)
    # Get the table from the stream's first batch
    # result_table = pa.Table.from_batches([next(result_stream)])
    # assert 'result' in result_table.column_names


def test_do_get_nonexistent_data(server: TestServer) -> None:
    """Test GET operation with non-existent data.

    This test verifies that the server correctly handles GET requests for
    non-existent data by raising an appropriate FlightServerError.

    Args:
        server: The test server instance.
    """
    test_command = "nonexistent_command"
    ticket = fl.Ticket(test_command.encode())

    with pytest.raises(fl.FlightServerError) as exc_info:
        server.do_get(None, ticket)

    assert f"No data found for command: {test_command}" in str(exc_info.value)


def test_thread_safety_put(server: TestServer, test_table: pa.Table) -> None:
    """Test thread safety of PUT operation.

    This test verifies that the server can handle multiple concurrent PUT operations
    correctly by using a thread pool to submit several requests simultaneously.

    Args:
        server: The test server instance.
        test_table: A sample PyArrow table for testing.
    """

    def put_data(command: str) -> fl.FlightDescriptor:
        """Helper function to perform a PUT operation with the given command.

        Args:
            command: The command string to use for the PUT operation.

        Returns:
            The FlightDescriptor returned by the PUT operation.
        """
        context = Mock()
        writer = Mock()
        descriptor = fl.FlightDescriptor.for_command(command)
        reader = Mock()
        reader.read_all.return_value = test_table
        return server.do_put(context, descriptor, reader, writer)

    # Execute multiple PUT operations concurrently
    commands = [f"command_{i}" for i in range(5)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(put_data, cmd) for cmd in commands]
        concurrent.futures.wait(futures)

    # Verify all data was stored correctly
    for command in commands:
        assert command in server._storage
        assert pa.Table.equals(server._storage[command], test_table)


# @pytest.mark.parametrize("port", [
#     5010,
#     8080,
#     12345
# ])
# def test_server_different_ports(port):
#     """Test server initialization with different ports."""
#     #flight_client = fl.FlightClient(host="localhost", port=port).start()
#
#     server = TestServer(host="localhost", port=port)
#
#     assert isinstance(server, Server)
#     # Verify the port is correctly set in the URI
#     assert str(port) in server._location


def test_invalid_table_schema(server: TestServer) -> None:
    """Test handling of invalid table schema.

    This test verifies that the server correctly handles tables with invalid schemas
    by raising an appropriate exception when attempting to process them.

    Args:
        server: The test server instance.
    """
    # Create an invalid table (missing required columns)
    invalid_data = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["invalid_column"])

    test_command = "invalid_data"
    descriptor = fl.FlightDescriptor.for_command(test_command)

    reader = Mock()
    reader.read_all.return_value = invalid_data

    # Store the invalid data
    server.do_put(None, descriptor, reader, None)

    # Try to get and process the invalid data
    ticket = fl.Ticket(test_command.encode())
    with pytest.raises(Exception):  # Specific exception type should match your implementation
        server.do_get(None, ticket)
