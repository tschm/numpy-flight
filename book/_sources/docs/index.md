# [Numpy with Apache Arrow Flight](https://tschm.github.io/numpy-flight/book)

[![PyPI version](https://badge.fury.io/py/numpy-flight.svg)](https://badge.fury.io/py/numpy-flight)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![CI](https://github.com/tschm/numpy-flight/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/numpy-flight/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/tschm/numpy-flight/badge.svg?branch=main)](https://coveralls.io/github/tschm/numpy-flight?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/numpy-flight/badge)](https://www.codefactor.io/repository/github/tschm/numpy-flight)
[![Created with qCradle](https://img.shields.io/badge/Created%20with-qCradle-blue?style=flat-square)](https://github.com/tschm/package)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)

## A Problem

We provide

- An abstract base class for an Apache flight server
- A client class to communicate with such servers

The client provides a simple interface for sending NumPy arrays,
performing computations, and retrieving results, all while handling
the serialization and deserialization automatically in the background
using Apache Arrow.

To create a server we expect the user to overload a function performing
the calcutation based on a dictionary of numpy arrays.

## Features

- Seamless conversion between NumPy arrays and Arrow Tables
- Simple interface for data transfer operations
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
>>> from tschm.flight import Server

>>> class TestServer(Server):
...     def f(self, matrices):
...          self.logger.info(f"{matrices.keys()}")
...          # Simple implementation for testing - just return the input
...          return {key : 2*value for key, value in matrices.items()}
```

All complexity is hidden in the class 'Server' which is itself a child
of the pyarrrow's FlightServerBase class. It is enough to implement
the method 'f' which is expecting a dictionary of numpy arrays. It will
also return a dictionary of numpy arrays.

The server can be started locally with

```python
>>> server = TestServer.start(host="127.0.0.1", port=5555)
```

While the server is running we can use a Python client for computations

```python
>>> import numpy as np
>>> from tschm.flight import Client

>>> with Client(location="grpc://127.0.0.1:5555") as client:
...     output = client.compute(command="compute", data={"input": np.array([1,2,3])})

>>> print(output["input"])
[2 4 6]

```

Clients for other languages are thinkable.
We shut the server down with

```python
server.shutdown()
```
