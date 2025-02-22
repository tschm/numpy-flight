import logging
import threading  # Module for creating and managing threads; used for thread safety with locking.
from abc import ABC, abstractmethod

import numpy as np
import pyarrow.flight as fl  # PyArrow's Flight module to handle gRPC-based data transfer with Arrow.

from .utils.alter import np_2_pa, pa_2_np


class Server(fl.FlightServerBase, ABC):
    """
    A Flight Server implementation that handles matrix data and performs computations on it.
    """

    def __init__(self, host="0.0.0.0", port=8080, logger=None, **kwargs):
        """
        Initialize the server with the provided host and port, and optionally a logger.

        :param host: Host for the server.
        :param port: Port on which the server will listen.
        :param logger: Optional logger to use for logging messages (defaults to loguru).
        :param kwargs: Additional arguments passed to the FlightServerBase constructor.
        """
        uri = f"grpc://{host}:{port}"
        super().__init__(uri, **kwargs)  # Initialize the base FlightServer with the URI.
        self._logger = logger or logging.getLogger(__name__)  # Use provided logger or default to loguru's logger.
        self._storage = {}  # Dictionary to store uploaded data associated with specific commands.
        self._lock = threading.Lock()  # Lock for ensuring thread safety when accessing shared resources.

    @property
    def logger(self):
        """Getter for the logger."""
        return self._logger

    @staticmethod
    def _extract_command_from_ticket(ticket):
        """
        Helper method to extract the command from a Flight Ticket.

        :param ticket: The Flight Ticket containing the command.
        :return: The command extracted from the ticket.
        """
        return ticket.ticket.decode("utf-8")

    def do_put(self, context, descriptor, reader, writer):
        """
        Handle a PUT request, store the provided data (Arrow Table) in the server's storage.

        :param context: The request context.
        :param descriptor: The Flight Descriptor for the PUT request.
        :param reader: Reader for reading the Arrow Table data.
        :param writer: Writer for writing responses.
        :return: A Flight Descriptor confirming the data storage.
        """
        with self._lock:  # Ensure thread safety when accessing shared resources.
            # Read and store the data
            command = descriptor.command.decode("utf-8")
            self.logger.info(f"Processing PUT request for command: {command}")

            table = reader.read_all()  # Read the complete Arrow Table data.
            self.logger.info(f"Table: {table}")

            # Store the table using the command as the key
            self._storage[command] = table

            self.logger.info(f"Data stored for command: {command}")

        return fl.FlightDescriptor.for_command(command)  # Return a Flight Descriptor.

    def do_get(self, context, ticket):
        """
        Handle a GET request, retrieve the stored data based on the ticket's command.

        :param context: The request context.
        :param ticket: The Flight Ticket for the GET request.
        :return: A RecordBatchStream containing the result data.
        """
        # Get the command from the ticket
        command = self._extract_command_from_ticket(ticket)
        self.logger.info(f"Processing GET request for command: {command}")

        # Retrieve the stored table
        if command not in self._storage:
            raise fl.FlightServerError(f"No data found for command: {command}")

        table = self._storage[command]
        self.logger.info(f"Retrieved data for command: {command}")

        # Process the table to extract matrices
        matrices = pa_2_np(table)

        # Compute results (e.g., perform computations based on matrices)
        np_data = self.f(matrices)
        result_table = np_2_pa(np_data)

        self.logger.info(result_table.schema.names)
        self.logger.info("Computation completed. Returning results.")

        # Create and return a RecordBatchStream with the result
        return fl.RecordBatchStream(result_table)

    @classmethod
    def start(cls, host="0.0.0.0", port=8080, logger=None, **kwargs):  # pragma: no cover
        """
        Start the server with the specified port and logger.

        :param port: The port on which to run the server.
        :param logger: Optional logger to use.
        :param kwargs: Additional arguments passed to the constructor.
        """
        server = cls(host=host, port=port, logger=logger, **kwargs)  # Instantiate the server.
        server.logger.info(f"Starting {cls} Flight server on port {port}...")  # Log the server start.
        server.serve()  # Start the server to handle incoming requests.

    @abstractmethod
    def f(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...  # pragma: no cover
