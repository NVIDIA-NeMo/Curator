"""File writing operations for llm.txt output."""

from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def write_llm_txt_file(app: Sphinx, docname: str, content: str) -> None:
    """Write llm.txt file to output directory."""
    try:
        outdir = Path(app.outdir)

        # Determine output path
        if docname == "index":
            txt_path = outdir / "index.llm.txt"
        elif docname.endswith("/index"):
            # Directory index: docs/about/index.md -> _build/html/about/index.llm.txt
            txt_path = outdir / docname[:-6] / "index.llm.txt"
        else:
            # Regular file: docs/install.md -> _build/html/install.llm.txt
            txt_path = outdir / f"{docname}.llm.txt"

        # Create directory if needed
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Generated llm.txt: {txt_path}")

    except Exception:
        logger.exception(f"Failed to write llm.txt for {docname}")


def get_output_path(app: Sphinx, docname: str) -> Path:
    """Get the output path for a document's llm.txt file."""
    outdir = Path(app.outdir)

    if docname == "index":
        return outdir / "index.llm.txt"
    elif docname.endswith("/index"):
        return outdir / docname[:-6] / "index.llm.txt"
    else:
        return outdir / f"{docname}.llm.txt"
