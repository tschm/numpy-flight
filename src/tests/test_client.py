import numpy as np
import pyarrow as pa
import pyarrow.flight as fl
import pytest

from np.flight import Client  # Adjust import path as needed
from np.flight.utils.alter import np_2_pa, pa_2_np


@pytest.fixture
def flight_client(mocker):
    """
    Create a mock Flight client for testing using pytest-mock.
    """
    return mocker.create_autospec(fl.FlightClient)


@pytest.fixture
def numpy_client(mocker, flight_client):
    """
    Create a NumpyClient instance with a mock Flight client.
    """
    mock_fl_connect = mocker.patch("pyarrow.flight.connect")  # Replace with the actual module path
    # mock_client = MagicMock()
    mock_fl_connect.return_value = flight_client

    location = "some_location"
    kwargs = {"key": "value"}

    # Act
    with Client(location, **kwargs) as numpy_client:
        # Assert
        mock_fl_connect.assert_called_once_with(location, **kwargs)
        assert numpy_client.flight == flight_client
        yield numpy_client


def test_numpy_client_init(numpy_client, flight_client):
    assert numpy_client.flight == flight_client


def test_descriptor_creation():
    """Test that Flight descriptors are created correctly."""
    command = "test_command"
    descriptor = Client.descriptor(command)

    assert isinstance(descriptor, fl.FlightDescriptor)
    assert descriptor.command == command.encode()


def test_np_2_pa(test_data, test_table):
    assert np_2_pa(test_data).equals(test_table)
    assert np.array_equal(pa_2_np(test_table)["values"], test_data["values"])
    assert np.array_equal(pa_2_np(test_table)["labels"], test_data["labels"])


def test_write_operation(numpy_client, test_data, test_table, mocker):
    """Test the write operation with mock Flight client."""
    # Set up mock writer using pytest-mock
    mock_writer = mocker.create_autospec(fl.FlightStreamWriter)
    numpy_client.flight.do_put.return_value = (mock_writer, None)
    # flight_client.do_put.return_value = (mock_writer, None)

    # Spy on np_2_pa to capture the converted table
    # np_2_pa_spy = mocker.spy(numpy_client.np_2_pa, '__call__')

    # Perform write operation
    numpy_client.write("test_write", test_data)

    # Verify interactions
    numpy_client.flight.do_put.assert_called_once()
    mock_writer.write_table.assert_called_once()
    assert mock_writer.write_table.call_args.args[0].equals(test_table)
    mock_writer.close.assert_called_once()


def test_write_handles_errors(numpy_client, test_data, mocker):
    """Test that write operation properly handles errors."""
    # Set up mock writer that raises an error
    mock_writer = mocker.create_autospec(fl.FlightStreamWriter)
    mock_writer.write_table.side_effect = fl.FlightError("Test error")
    numpy_client.flight.do_put.return_value = (mock_writer, None)

    # Verify error handling
    with pytest.raises(fl.FlightError):
        numpy_client.write("test_error", test_data)

    # Verify writer was still closed
    mock_writer.close.assert_called_once()


def test_get_operation(numpy_client, test_table, mocker):
    """Test the get operation with mock Flight client."""
    # Set up mock reader using pytest-mock
    mock_reader = mocker.create_autospec(fl.FlightStreamReader)
    mock_reader.read_all.return_value = test_table
    numpy_client.flight.do_get.return_value = mock_reader

    # Perform get operation
    result = numpy_client.get("test_get")

    # Verify interactions and result
    numpy_client.flight.do_get.assert_called_once()
    mock_reader.read_all.assert_called_once()
    assert isinstance(result, pa.Table)
    assert result.schema == test_table.schema
    assert result.equals(test_table)


def test_compute_operation(numpy_client, test_data, test_table, mocker):
    """Test the compute operation (combination of write and get)."""
    # Set up mocks for both write and get operations
    mock_writer = mocker.create_autospec(fl.FlightStreamWriter)
    numpy_client.flight.do_put.return_value = (mock_writer, None)

    mock_reader = mocker.create_autospec(fl.FlightStreamReader)
    mock_reader.read_all.return_value = test_table
    numpy_client.flight.do_get.return_value = mock_reader

    # Perform compute operation
    print(test_data)
    result = numpy_client.compute("test_compute", test_data)

    # Verify the result is a dictionary of NumPy arrays
    assert isinstance(result, dict)
    assert all(isinstance(v, np.ndarray) for v in result.values())
    assert set(result.keys()) == set(test_data.keys())


def test_invalid_data_handling(numpy_client):
    """Test handling of invalid input data."""
    invalid_data = {
        "values": "not_an_array",  # Invalid type
        "labels": np.array([1, 2, 3]),
    }

    with pytest.raises(ValueError):
        numpy_client.write("test_invalid", invalid_data)


@pytest.mark.parametrize(
    "command,expected_encoded",
    [
        ("test", b"test"),
        ("custom_command", b"custom_command"),
        ("", b""),
    ],
)
def test_descriptor_with_different_commands(command, expected_encoded):
    """Test descriptor creation with different commands."""
    descriptor = Client.descriptor(command)
    assert descriptor.command == expected_encoded


@pytest.mark.parametrize(
    "data,expected_error",
    [
        ({"values": None}, TypeError),
        ({}, ValueError),
        # ({"values": np.array([]), "labels": None}, TypeError),
    ],
)
def test_write_with_invalid_inputs(numpy_client, data, expected_error, mocker):
    """Test write operation with various invalid inputs."""
    with pytest.raises(expected_error):
        mock_writer = mocker.create_autospec(fl.FlightStreamWriter)
        numpy_client.flight.do_put.return_value = (mock_writer, None)

        numpy_client.write("test", data)


def test_large_data_handling(numpy_client, mocker):
    """Test handling of large data arrays."""
    large_data = {"values": np.arange(1000000, dtype=np.int64), "labels": np.array(["label"] * 1000000)}

    mock_writer = mocker.create_autospec(fl.FlightStreamWriter)
    numpy_client.flight.do_put.return_value = (mock_writer, None)

    numpy_client.write("test_large", large_data)
    mock_writer.write_table.assert_called_once()


def test_schema_preservation(numpy_client, test_data, test_table, mocker):
    """Test that schema information is preserved through write/get operations."""
    mock_writer = mocker.create_autospec(fl.FlightStreamWriter)
    numpy_client.flight.do_put.return_value = (mock_writer, None)

    mock_reader = mocker.create_autospec(fl.FlightStreamReader)
    mock_reader.read_all.return_value = test_table
    numpy_client.flight.do_get.return_value = mock_reader

    result = numpy_client.compute("test_schema", test_data)

    assert set(result.keys()) == set(test_data.keys())
    assert all(result[k].dtype == test_data[k].dtype for k in result)
