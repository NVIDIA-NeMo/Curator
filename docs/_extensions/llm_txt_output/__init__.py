"""Sphinx extension to generate llm.txt files for LLM consumption.

This extension creates parallel llm.txt files for each document in a standardized
markdown format that Large Language Models can easily parse and understand.

The llm.txt format includes:
- Document title and summary
- Clean overview text
- Key sections with descriptions
- Related resources/links
- Metadata

See README.md for detailed configuration options and usage examples.
"""

from typing import Any

from sphinx.application import Sphinx

from .config import get_default_settings, validate_config
from .processor import on_build_finished


def setup(app: Sphinx) -> dict[str, Any]:
    """Set up Sphinx extension for llm.txt generation."""
    # Add configuration with default settings
    default_settings = get_default_settings()
    app.add_config_value("llm_txt_settings", default_settings, "html")

    # Connect to build events
    app.connect("config-inited", validate_config)
    app.connect("build-finished", on_build_finished)

    return {
        "version": "1.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
