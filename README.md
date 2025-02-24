# [Numpy with Apache Arrow Flight](https://tschm.github.io/numpy-flight/book)

[![PyPI version](https://badge.fury.io/py/numpy-flight.svg)](https://badge.fury.io/py/numpy-flight)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![CI](https://github.com/tschm/numpy-flight/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/numpy-flight/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/tschm/numpy-flight/badge.svg?branch=main)](https://coveralls.io/github/tschm/numpy-flight?branch=main)
[![Created with qCradle](https://img.shields.io/badge/Created%20with-qCradle-blue?style=flat-square)](https://github.com/tschm/package)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/numpy-flight)

## A Problem

We provide

- An abstract base class for an Apache flight server
- A client class to communicate with such servers

We efficiently transfer NumPy arrays over Apache Arrow Flight using a custom Client.
The client provides a simple interface for sending NumPy arrays,
performing computations, and retrieving results, all while handling
the serialization and deserialization automatically in the background.

To create a server we expect the user to overload a function performing
the calcutation based on a dictionary of numpy arrays.

## Features

- Seamless conversion between NumPy arrays and Arrow Tables
- Simple interface for data transfer operations
- Support for batch computations
- Automatic resource management
- Type-safe operations with proper error handling

## Installation

You can install this client via

```bash
pip install numpy-flight
```

## Usage

### Basic Setup

We introduce the Baseclass 'Server':

```python
>> > from numpy.flight import Server

>> >

class TestServer(Server):


    ...


def f(self, matrices):


    ...
self.logger.info(f"{matrices.keys()}")
...  # Simple implementation for testing - just return the input
...
return {key: 2 * value for key, value in matrices.items()}
```

All complexity is hidden in the class 'Server' which is itself a child
of the pyarrrow's FlightServerBase class. It is enough to implement
the method 'f' which is expecting a dictionary of numpy arrays. It will
also return a dictionary of numpy arrays.

The server can be started locally with

```python
>>> server = TestServer.start(host="127.0.0.1", port=5555)
```

While the server is running we can use a client for computations

```python
>> > import numpy as np
>> > from numpy.flight import Client

>> > with Client(location="grpc://127.0.0.1:5555") as client:
    ...
output = client.compute(command="compute", data={"input": np.array([1, 2, 3])})

>> > print(output["input"])
[2 4 6]

```

```python
server.shutdown()
```

### Retrieving Data

```python
# Get data from the server
result_table = client.get('retrieve_data')
```

### Computing with Data

```python
# Send data and get results in one operation
input_data = {
    'x': np.array([1, 2, 3]),
    'y': np.array([4, 5, 6])
}
results = client.compute('multiply_arrays', input_data)
```

## API Reference

### `Client`

#### `__init__(client: fl.FlightClient)`

Initialize the client with a Flight client instance.

#### `write(command: str, data: Dict[str, np.ndarray])`

Write NumPy arrays to the Flight server.

- `command`: String identifying the operation
- `data`: Dictionary mapping column names to NumPy arrays

#### `get(command: str) -> pa.Table`

Retrieve data from the Flight server.

- `command`: String identifying the data to retrieve
- Returns: PyArrow Table containing the retrieved data

#### `compute(command: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]`

Perform a computation on the server and retrieve results.

- `command`: String identifying the computation
- `data`: Input data as dictionary of NumPy arrays
- Returns: Dictionary of NumPy arrays containing results

## Error Handling

The client includes proper error handling for common scenarios:

- `FlightError`: Raised for Flight protocol communication errors
- `ValueError`: Raised for data conversion errors
- Resource cleanup is handled automatically, even in error cases

## Best Practices

- Always close the Flight client when done:

```python
client.close()
```

```python
client = Client('grpc://localhost:8815')
# ... perform operations
```

- Handle large datasets in chunks to manage memory usage effectively.

### **Set Up Environment**

```bash
make install
```

This installs/updates [uv](https://github.com/astral-sh/uv),
creates your virtual environment and installs dependencies.

For adding or removing packages:

```bash
uv add/remove requests  # for main dependencies
uv add/remove requests --dev  # for dev dependencies
```

## Contributing

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request
