"""Server module for handling NumPy array operations over Apache Arrow Flight.

This module provides a server implementation that can receive NumPy arrays via
Arrow Flight protocol, perform computations on them, and return the results.
It includes thread-safe storage for data and an abstract method for implementing
specific computation logic.
"""

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np
import pyarrow.flight as fl

from .utils.alter import np_2_pa, pa_2_np


class Server(fl.FlightServerBase, ABC):
    """A Flight Server implementation that handles matrix data and performs computations on it.

    This abstract base class provides the foundation for creating Flight servers that can
    receive NumPy arrays, store them, perform computations, and return results. Subclasses
    must implement the `f` method to define specific computation logic.

    Attributes:
        _logger: Logger instance for recording server activities.
        _storage: Dictionary to store uploaded data associated with specific commands.
        _lock: Threading lock for ensuring thread safety when accessing shared resources.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8080, logger: logging.Logger | None = None, **kwargs
    ) -> None:
        """Initialize the server with the provided host and port, and optionally a logger.

        Args:
            host: Host address for the server to bind to.
            port: Port on which the server will listen.
            logger: Optional logger to use for logging messages.
            **kwargs: Additional arguments passed to the FlightServerBase constructor.
        """
        uri = f"grpc://{host}:{port}"
        super().__init__(uri, **kwargs)
        self._logger = logger or logging.getLogger(__name__)
        self._storage = {}  # Dictionary to store uploaded data
        self._lock = threading.Lock()  # Lock for thread safety

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance used by this server.

        Returns:
            The logger instance for recording server activities.
        """
        return self._logger

    @staticmethod
    def _extract_command_from_ticket(ticket: fl.Ticket) -> str:
        """Extract the command string from a Flight Ticket.

        Args:
            ticket: The Flight Ticket containing the command.

        Returns:
            The command string extracted from the ticket.
        """
        return ticket.ticket.decode("utf-8")

    def do_put(
        self,
        context: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.FlightMetadataWriter,
    ) -> fl.FlightDescriptor:
        """Handle a PUT request, storing the provided data in the server's storage.

        This method is called when a client sends data to the server. It reads the
        Arrow Table from the reader, stores it in the server's storage using the
        command from the descriptor as the key, and returns a descriptor confirming
        the storage.

        Args:
            context: The request context containing client information.
            descriptor: The Flight Descriptor for the PUT request containing the command.
            reader: Reader for reading the Arrow Table data sent by the client.
            writer: Writer for writing metadata responses back to the client.

        Returns:
            A Flight Descriptor confirming the data storage.
        """
        with self._lock:  # Ensure thread safety
            # Extract command and read data
            command = descriptor.command.decode("utf-8")
            self.logger.info(f"Processing PUT request for command: {command}")

            table = reader.read_all()
            self.logger.info(f"Table: {table}")

            # Store the table using the command as the key
            self._storage[command] = table
            self.logger.info(f"Data stored for command: {command}")

        return fl.FlightDescriptor.for_command(command)

    def do_get(self, context: fl.ServerCallContext, ticket: fl.Ticket) -> fl.RecordBatchStream:
        """Handle a GET request, retrieving and processing stored data.

        This method is called when a client requests data from the server. It extracts
        the command from the ticket, retrieves the corresponding data from storage,
        processes it using the abstract `f` method, and returns the results.

        Args:
            context: The request context containing client information.
            ticket: The Flight Ticket containing the command for the GET request.

        Returns:
            A RecordBatchStream containing the processed result data.

        Raises:
            fl.FlightServerError: If no data is found for the requested command.
        """
        # Extract command from ticket
        command = self._extract_command_from_ticket(ticket)
        self.logger.info(f"Processing GET request for command: {command}")

        # Retrieve the stored table
        if command not in self._storage:
            raise fl.FlightServerError(f"No data found for command: {command}")

        table = self._storage[command]
        self.logger.info(f"Retrieved data for command: {command}")

        # Convert Arrow Table to NumPy arrays
        matrices = pa_2_np(table)

        # Apply the computation function and convert results back to Arrow
        np_data = self.f(matrices)
        result_table = np_2_pa(np_data)

        self.logger.info(f"Result schema: {result_table.schema.names}")
        self.logger.info("Computation completed. Returning results.")

        # Return results as a RecordBatchStream
        return fl.RecordBatchStream(result_table)

    @classmethod
    def start(
        cls, host: str = "127.0.0.1", port: int = 8080, logger: logging.Logger | None = None, **kwargs
    ) -> "Server":  # pragma: no cover
        """Create and start a server instance.

        This class method creates a new server instance with the specified parameters
        and returns it. The actual server is not started (serve() is not called).

        Args:
            host: The host address to bind the server to.
            port: The port on which to run the server.
            logger: Optional logger to use for recording server activities.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A configured server instance ready to be started.
        """
        server = cls(host=host, port=port, logger=logger, **kwargs)
        server.logger.info(f"Starting {cls.__name__} Flight server on {host}:{port}...")
        # server.serve()  # Uncomment to actually start the server
        return server

    @abstractmethod
    def f(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # pragma: no cover
        """Process the input matrices and return the computation results.

        This abstract method must be implemented by subclasses to define the specific
        computation logic to be applied to the input data.

        Args:
            matrices: A dictionary mapping column names to NumPy arrays containing
                     the input data to process.

        Returns:
            A dictionary mapping column names to NumPy arrays containing the
            computation results.
        """
        ...
