from unittest.mock import Mock

import pyarrow as pa
import pyarrow.flight as fl
import pytest
from loguru import logger

from np.flight import Server  # Adjust import path as needed


# Create a concrete implementation of the abstract Server class for testing
class TestServer(Server):
    def f(self, matrices):
        # Simple implementation for testing - just return the input as a table
        return pa.Table.from_arrays([pa.array(matrices)], names=["result"])


@pytest.fixture(scope="module")
def server():
    """Fixture providing a test server instance."""
    return TestServer(host="localhost", port=5008, logger=logger)


# def test_server_initialization_with_default_logger():
#    """Test server initialization with default logger."""
#    server = TestServer("localhost", 5010)
#    assert server.logger == logger
#    assert server._storage == {}


# def test_server_initialization_with_custom_logger():
#    """Test server initialization with custom logger."""
#    custom_logger = Mock()
#    server = TestServer("localhost", 5009, logger=custom_logger)
#    assert server.logger == custom_logger


@pytest.mark.parametrize("test_command", ["test_command1", "test_command2", "complex/command/path", "123456"])
def test_extract_command_from_ticket(server, test_command):
    """Test command extraction from Flight ticket with various commands."""
    ticket = fl.Ticket(test_command.encode())
    extracted_command = server._extract_command_from_ticket(ticket)
    assert extracted_command == test_command


def test_do_put(server, test_table):
    """Test PUT operation with mock reader and writer."""
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


def test_do_get_existing_data(server, test_table):
    """Test GET operation with existing data."""
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


def test_do_get_nonexistent_data(server):
    """Test GET operation with non-existent data."""
    test_command = "nonexistent_command"
    ticket = fl.Ticket(test_command.encode())

    with pytest.raises(fl.FlightServerError) as exc_info:
        server.do_get(None, ticket)

    assert f"No data found for command: {test_command}" in str(exc_info.value)


def test_thread_safety_put(server, test_table):
    """Test thread safety of PUT operation."""
    import concurrent.futures

    def put_data(command):
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


def test_invalid_table_schema(server):
    """Test handling of invalid table schema."""
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
