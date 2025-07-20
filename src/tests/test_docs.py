"""Tests for documentation code examples.

This module contains tests that verify the code examples in the project's README.md
file are valid and can be executed without errors using doctest.
"""

import doctest
import os
import re
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture


@pytest.fixture()
def docstring(root_dir: Path) -> str:
    """Extract Python code blocks from README.md and prepare them for doctest.

    Args:
        root_dir: The root directory of the project.

    Returns:
        A string containing all Python code blocks from README.md, formatted for doctest.
    """
    # Read the README.md file
    with open(root_dir / "README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


def test_blocks(root_dir: Path, docstring: str, capfd: CaptureFixture) -> None:
    """Test that Python code blocks in README.md execute without errors.

    This test runs all Python code examples from the README.md file through
    doctest to ensure they are valid and produce the expected output.

    Args:
        root_dir: The root directory of the project.
        docstring: The extracted Python code blocks from README.md.
        capfd: Pytest fixture to capture stdout/stderr output.
    """
    os.chdir(root_dir)

    try:
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")

    # Capture the output after running doctests
    captured = capfd.readouterr()

    # If there is any output (error message), fail the test
    if captured.out:
        pytest.fail(f"Doctests failed with the following output:\n{captured.out} and \n{docstring}")
