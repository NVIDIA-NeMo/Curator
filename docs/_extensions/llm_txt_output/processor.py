"""Document processing and build orchestration for llm.txt output."""

import fnmatch
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.util import logging

from .content_extractor import extract_document_content
from .template_builder import build_llm_txt_content
from .writer import write_llm_txt_file

logger = logging.getLogger(__name__)


def on_build_finished(app: Sphinx, exception: Exception) -> None:
    """Generate llm.txt files after HTML build is complete."""
    if exception is not None:
        return

    settings = getattr(app.config, "llm_txt_settings", {})

    # Check if enabled
    if not settings.get("enabled", True):
        return

    verbose = settings.get("verbose", False)
    log_func = logger.info if verbose else logger.debug
    log_func("Generating llm.txt output files...")

    # Process all documents
    generated_count = 0
    failed_count = 0
    all_content = []  # Collect all llm.txt content for full file

    for docname in app.env.all_docs:
        if should_generate_llm_txt(app.config, docname):
            try:
                # Extract content
                content_data = extract_document_content(app.env, docname, settings)

                # Build llm.txt content
                llm_txt_content = build_llm_txt_content(content_data)

                # Write individual file
                write_llm_txt_file(app, docname, llm_txt_content)

                # Collect for full file
                all_content.append(
                    {
                        "docname": docname,
                        "title": content_data.get("title", "Untitled"),
                        "content": llm_txt_content,
                    }
                )

                generated_count += 1
                logger.debug(f"Generated llm.txt for {docname}")

            except Exception:
                logger.exception(f"Error generating llm.txt for {docname}")
                failed_count += 1

    # Generate llm-full.txt aggregated file
    if all_content and settings.get("generate_full_file", True):
        try:
            _write_full_llm_txt(app, all_content, settings)
            log_func("Generated llm-full.txt with all documentation")
        except Exception:
            logger.exception("Failed to generate llm-full.txt")

    # Final logging
    log_func(f"Generated {generated_count} llm.txt files")
    if failed_count > 0:
        logger.warning(f"Failed to generate {failed_count} llm.txt files")


def should_generate_llm_txt(config: Config, docname: str) -> bool:
    """Check if llm.txt should be generated for this document."""
    settings = getattr(config, "llm_txt_settings", {})

    if not settings.get("enabled", True):
        return False

    if not docname or not isinstance(docname, str):
        logger.warning(f"Invalid docname for llm.txt generation: {docname}")
        return False

    # Check if document is content gated (respects Sphinx exclude_patterns)
    if is_content_gated(config, docname):
        logger.info(f"Excluding {docname} from llm.txt generation due to content gating")
        return False

    # Check llm.txt extension's own exclude patterns
    for pattern in settings.get("exclude_patterns", []):
        if isinstance(pattern, str) and docname.startswith(pattern):
            return False

    return True


def is_content_gated(config: Config, docname: str) -> bool:
    """
    Check if a document is content gated by checking Sphinx's exclude_patterns.
    This works with the content_gating extension.
    """
    sphinx_exclude_patterns = getattr(config, "exclude_patterns", [])
    if not sphinx_exclude_patterns:
        return False

    # Convert docname to potential file paths
    possible_paths = [docname + ".md", docname + ".rst", docname]

    for possible_path in possible_paths:
        # Check if this path matches any exclude pattern
        for pattern in sphinx_exclude_patterns:
            if isinstance(pattern, str) and fnmatch.fnmatch(possible_path, pattern):
                logger.debug(f"Document {docname} is content gated (matches pattern: {pattern})")
                return True

    return False


def _write_full_llm_txt(app: Sphinx, all_content: list[dict[str, str]], _settings: dict) -> None:
    """Write aggregated llm-full.txt file with all documentation."""
    outdir = Path(app.outdir)
    full_path = outdir / "llm-full.txt"

    # Build introduction header
    lines = []
    lines.append("# Complete Documentation")
    lines.append("")

    # Add site metadata if available
    project_name = getattr(app.config, "project", "Documentation")
    if hasattr(app.config, "version"):
        version = app.config.version
        lines.append(f"> {project_name} {version} - Complete documentation in LLM-friendly format")
    else:
        lines.append(f"> {project_name} - Complete documentation in LLM-friendly format")
    lines.append("")

    # Add table of contents
    lines.append("## Table of Contents")
    lines.append("")
    lines.append(f"This file contains {len(all_content)} documentation pages:")
    lines.append("")

    # Sort content by docname for consistent ordering
    sorted_content = sorted(all_content, key=lambda x: x["docname"])

    for item in sorted_content:
        docname = item["docname"]
        title = item["title"]
        lines.append(f"- {title} (`{docname}`)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Add each document with clear separators
    for i, item in enumerate(sorted_content, 1):
        docname = item["docname"]
        content = item["content"]

        # Add document separator
        lines.append(f"<!-- Document {i}/{len(sorted_content)}: {docname} -->")
        lines.append("")

        # Add content
        lines.append(content)
        lines.append("")

        # Add visual separator between documents
        if i < len(sorted_content):
            lines.append("---")
            lines.append("")

    # Write file
    full_content = "\n".join(lines)
    full_path.write_text(full_content, encoding="utf-8")

    # Log file size
    file_size_mb = len(full_content) / (1024 * 1024)
    logger.info(f"Generated llm-full.txt ({file_size_mb:.2f} MB, {len(sorted_content)} documents)")
