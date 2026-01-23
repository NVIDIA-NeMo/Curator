# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest


@pytest.fixture
def simple_html() -> str:
    """Simple HTML content for basic testing."""
    return "<html><body>Content</body></html>"


@pytest.fixture
def html_with_content() -> str:
    """HTML with paragraph content for testing."""
    return "<html><body><p>Test content</p></body></html>"


@pytest.fixture
def complex_html() -> str:
    """Complex HTML structure for comprehensive testing."""
    return """
        <html>
        <head>
            <title>Test</title>
        </head>
        <body>
            <p>Content</p>
        </body>
        </html>
        """


@pytest.fixture
def math_html() -> str:
    """HTML containing mathematical content."""
    return """
        <html>
        <head>
            <title>Mathematical Formulas</title>
            <script type="text/javascript" async
                src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
            </script>
        </head>
        <body>
            <h1>Quadratic Formula</h1>
            <p>The quadratic formula is:</p>
            <div class="math">
                $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$
            </div>
            <p>Where <em>a</em>, <em>b</em>, and <em>c</em> are coefficients.</p>

            <h2>Example</h2>
            <p>For the equation $x^2 + 5x + 6 = 0$:</p>
            <ul>
                <li>a = 1</li>
                <li>b = 5</li>
                <li>c = 6</li>
            </ul>
        </body>
        </html>
        """


@pytest.fixture
def basic_notebook_json() -> str:
    """Basic notebook JSON for testing."""
    return json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 2,
            "cells": [{"cell_type": "code", "source": ["print('hello')"], "outputs": []}],
        }
    )


@pytest.fixture
def complex_notebook_json() -> str:
    """Complex notebook with multiple cell types and outputs - used for comprehensive testing."""
    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 2,
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Mathematical Analysis\n", "This notebook demonstrates mathematical concepts."],
            },
            {
                "cell_type": "code",
                "source": ["import numpy as np\n", "x = np.array([1, 2, 3, 4, 5])\n", "print(f'Mean: {np.mean(x)}')"],
                "outputs": [{"output_type": "stream", "text": ["Mean: 3.0\n"]}],
            },
            {
                "cell_type": "code",
                "source": [
                    "# Calculate standard deviation\n",
                    "std_dev = np.std(x)\n",
                    "print(f'Standard deviation: {std_dev}')",
                ],
                "outputs": [{"output_type": "execute_result", "data": {"text/plain": ["1.4142135623730951"]}}],
            },
        ],
    }
    return json.dumps(notebook_content)


@pytest.fixture
def comprehensive_notebook_json() -> str:
    """Comprehensive notebook with multiple cell types and outputs for text conversion testing."""
    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 2,
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "This is markdown content."]},
            {
                "cell_type": "code",
                "source": ["import numpy as np\n", "print('Hello World')"],
                "outputs": [
                    {"output_type": "stream", "text": ["Hello World\n"]},
                    {"output_type": "execute_result", "data": {"text/plain": ["42"]}},
                ],
            },
            {"cell_type": "raw", "source": ["Raw cell content"]},
            {
                "cell_type": "code",
                "source": ["x = 5"],
                "outputs": [{"output_type": "display_data", "data": {"text/plain": ["<matplotlib.figure.Figure>"]}}],
            },
        ],
    }
    return json.dumps(notebook_content)


@pytest.fixture
def empty_notebook_json() -> str:
    """Empty notebook for edge case testing."""
    return json.dumps({"nbformat": 4, "nbformat_minor": 2, "cells": []})


@pytest.fixture
def plain_text() -> str:
    """Plain text content for testing."""
    return "Plain text content"


@pytest.fixture
def unknown_content() -> str:
    """Unknown content type for fallback testing."""
    return "Some unknown content that doesn't match any specific type"


@pytest.fixture
def sample_text_content() -> str:
    """Sample plain text content for testing."""
    return "This is plain text content."


@pytest.fixture
def sample_html_content() -> str:
    """Sample HTML content for testing."""
    return "<html><body><p>Test</p></body></html>"


@pytest.fixture
def sample_test_content() -> str:
    """Generic test content string."""
    return "test content"


@pytest.fixture
def extracted_text_responses() -> dict[str, str]:
    """Common extracted text responses for mocking."""
    return {"html": "Extracted HTML text", "lynx": "Extracted text", "generic": "Extracted HTML"}


@pytest.fixture
def sample_urls() -> dict[str, str]:
    """Common URLs used in tests."""
    return {
        "notebook": "http://example.com/notebook.ipynb",
        "html": "http://example.com/page.html",
        "text": "http://example.com/file.txt",
        "empty": "http://example.com/empty.txt",
        "none": "http://example.com/none.txt",
    }


@pytest.fixture
def test_records(sample_text_content: str, sample_urls: dict):
    """Sample record structures for different content types."""
    return {
        "notebook": {
            "binary_content": b'{"nbformat": 4, "nbformat_minor": 2, "cells": []}',
            "url": sample_urls["notebook"],
            "mime_type": "application/json",
        },
        "html": {
            "binary_content": b"<html><body><p>Test content</p></body></html>",
            "url": sample_urls["html"],
            "mime_type": "text/html",
        },
        "text": {
            "binary_content": sample_text_content.encode("utf-8"),
            "url": sample_urls["text"],
            "mime_type": "text/plain",
        },
        "empty": {"binary_content": b"", "url": sample_urls["empty"], "mime_type": "text/plain"},
        "none": {"binary_content": None, "url": sample_urls["none"], "mime_type": "text/plain"},
    }
