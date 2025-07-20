"""NumPy over Apache Arrow Flight package.

This package provides a client-server implementation for sending NumPy arrays
over the network using Apache Arrow Flight protocol. It includes:

- Client: For sending NumPy arrays to a server and retrieving results
- Server: An abstract base class for implementing computation servers
"""

from .numpy_client import Client  # noqa: F401
from .numpy_server import Server  # noqa: F401
