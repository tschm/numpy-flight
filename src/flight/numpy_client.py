"""Client module for handling NumPy array operations over Apache Arrow Flight.

This module provides a client interface for sending NumPy arrays to a Flight server,
retrieving data, and performing computations with automatic conversion between
NumPy arrays and Arrow Tables.
"""

import numpy as np
import pyarrow as pa
import pyarrow.flight as fl

from .utils.alter import np_2_pa, pa_2_np


class Client:
    """A client for handling NumPy array operations over Apache Arrow Flight.

    This class provides an interface for sending NumPy arrays to a Flight server,
    retrieving data, and performing computations. It handles the conversion between
    NumPy arrays and Arrow Tables automatically.

    Attributes:
        _client (fl.FlightClient): The underlying Flight client for network communication.
    """

    def __init__(self, location: str, **kwargs: object) -> None:
        """Initialize the NumpyClient with a Flight server location.

        Args:
            location: The URI location of the Flight server to connect to.
            **kwargs: Additional keyword arguments to pass to the Flight client.
        """
        self._location = location
        self._kwargs = kwargs

    def __enter__(self) -> "Client":
        """Open the database connection.

        Returns:
            Client: The client instance with an active connection.
        """
        self._client = fl.connect(self._location, **self._kwargs)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Close the connection.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_val: The exception value if an exception was raised in the context.
            exc_tb: The traceback if an exception was raised in the context.

        Raises:
            Exception: Re-raises any exception that occurred in the context.
        """
        self._client.close()
        if exc_val:  # pragma: no cover
            raise

    @property
    def flight(self) -> fl.FlightClient:
        """Get the underlying Flight client.

        Returns:
            fl.FlightClient: The Flight client instance used for communication.
        """
        return self._client

    @staticmethod
    def descriptor(command: str) -> fl.FlightDescriptor:
        """Create a FlightDescriptor for an opaque command.

        A FlightDescriptor is used to identify and describe the data being transferred
        over the Flight protocol.

        Args:
            command: The command string that identifies the operation to perform.

        Returns:
            A FlightDescriptor containing the command.
        """
        return fl.FlightDescriptor.for_command(command.encode())

    def write(self, command: str, data: dict[str, np.ndarray]) -> None:
        """Write NumPy array data to the Flight server.

        This method converts the input NumPy arrays to an Arrow Table and sends it
        to the server using the specified command.

        Args:
            command: The command string identifying the operation.
            data: A dictionary mapping column names to NumPy arrays.

        Raises:
            FlightError: If there's an error in the Flight protocol communication.
            ValueError: If the data cannot be converted to an Arrow Table.
        """
        # Create a descriptor for the data transfer
        descriptor = self.descriptor(command)

        # check if the dictionary is empty
        if not data:
            msg = "Empty data"
            raise ValueError(msg)

        # Convert NumPy arrays to Arrow Table
        table = np_2_pa(data)

        if table.num_rows == 0:
            msg = "Empty table"
            raise TypeError(msg)

        # Initialize the write operation with the server
        writer, _ = self.flight.do_put(descriptor, table.schema)

        try:
            # Send the data to the server
            writer.write_table(table)
        finally:
            # Ensure the writer is closed even if an error occurs
            writer.close()

    def get(self, command: str) -> pa.Table:
        """Retrieve data from the Flight server.

        Issues a GET request to the server and returns the results as an Arrow Table.

        Args:
            command: The command string identifying the data to retrieve.

        Returns:
            An Arrow Table containing the retrieved data.

        Raises:
            FlightError: If there's an error in the Flight protocol communication.
        """
        # Create a ticket for the data request
        ticket = fl.Ticket(command)

        # Get a reader for the requested data
        reader = self.flight.do_get(ticket)

        # Read and return all data as an Arrow Table
        return reader.read_all()

    def compute(self, command: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Send data to the server, perform computation, and retrieve results.

        This is a convenience method that combines write and get operations into
        a single call. It handles the conversion between NumPy arrays and Arrow
        Tables in both directions.

        Args:
            command: The command string identifying the computation to perform.
            data: A dictionary mapping column names to NumPy arrays for input.

        Returns:
            A dictionary mapping column names to NumPy arrays containing the results.

        Raises:
            FlightError: If there's an error in the Flight protocol communication.
            ValueError: If the data cannot be converted between formats.
        """
        # Send input data to the server
        self.write(command, data)

        # Retrieve and convert results back to NumPy arrays
        return pa_2_np(self.get(command))
