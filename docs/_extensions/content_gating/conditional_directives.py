"""
Conditional directives for content gating.

Supports conditional rendering for:
- toctree directive (global and per-entry conditions)
- grid-item-card directive

Usage examples:

Grid card with condition:
:::{grid-item-card} Title
:only: not ga

Content here
:::

Toctree with global condition:
::::{toctree}
:only: not ga
:hidden:
:caption: Section Title

document1.md
document2.md
::::

Toctree with inline conditions:
::::{toctree}
:hidden:
:caption: Section Title

document1.md
document2.md :only: not ga
document3.md :only: ea
::::
"""

import re
from typing import Any, ClassVar

from docutils.nodes import Node, container
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.directives.other import TocTree
from sphinx.util import logging
from sphinx_design.grids import GridItemCardDirective

from .condition_evaluator import should_include_content

logger = logging.getLogger(__name__)


class ConditionalGridItemCardDirective(GridItemCardDirective):
    """
    Extended grid-item-card directive that supports conditional rendering.
    """

    option_spec = GridItemCardDirective.option_spec.copy()
    option_spec["only"] = directives.unchanged

    def run(self) -> list[Node]:
        """
        Run the directive, checking the :only: condition first.
        """
        # Check if we have an :only: condition
        only_condition = self.options.get("only")

        if only_condition:
            # Parse and evaluate the condition using shared evaluator
            app = self.state.document.settings.env.app
            if not should_include_content(app, only_condition):
                # Return empty list to skip rendering this card
                logger.debug(f"Excluding grid-item-card due to condition: {only_condition}")
                return []

        # If no condition or condition is met, render normally
        return super().run()


class ConditionalTocTreeDirective(TocTree):
    """
    Extended toctree directive that supports conditional rendering at both
    the global level (entire toctree) and individual entry level.
    """

    option_spec = TocTree.option_spec.copy()
    option_spec["only"] = directives.unchanged

    def run(self) -> list[Node]:
        """
        Run the directive, applying both global and inline :only: conditions.
        """
        app = self.state.document.settings.env.app

        # Step 1: Check global :only: condition first
        global_only_condition = self.options.get("only")

        if global_only_condition:
            # Parse and evaluate the global condition using shared evaluator
            if not should_include_content(app, global_only_condition):
                # Global condition failed, skip entire toctree
                logger.debug(f"Excluding entire toctree due to global condition: {global_only_condition}")
                return []
            else:
                logger.debug(f"Global toctree condition passed: {global_only_condition}")

        # Step 2: Filter individual entries based on inline :only: conditions
        filtered_content = self._filter_content_lines(app)

        # Update the content with filtered lines
        if filtered_content != self.content:
            self.content = filtered_content

        # Step 3: If no content remains after filtering, return empty
        if not self.content or all(not line.strip() for line in self.content):
            logger.debug("No content remaining after filtering, excluding toctree")
            return []

        # Step 4: Render normally with filtered content
        return super().run()

    def _filter_content_lines(self, app: Sphinx) -> list[str]:
        """
        Filter toctree content lines based on inline :only: conditions.
        """
        filtered_lines = []

        for line in self.content:
            # Skip empty lines
            if not line.strip():
                filtered_lines.append(line)
                continue

            # Check if line has an inline :only: condition
            only_match = re.search(r"\s+:only:\s+(.+)$", line)

            if only_match:
                # Extract the condition and clean the line
                condition = only_match.group(1).strip()
                clean_line = line[: only_match.start()].rstrip()

                # Evaluate the condition using shared evaluator
                if should_include_content(app, condition):
                    # Include the line without the :only: part
                    filtered_lines.append(clean_line)
                    logger.debug(f"Including toctree entry: {clean_line}")
                else:
                    logger.debug(f"Excluding toctree entry: {clean_line} (condition: {condition})")
                    # Skip this line entirely
            else:
                # No inline condition, include the line as-is
                filtered_lines.append(line)

        return filtered_lines


class ConditionalContentDirective(Directive):
    """
    A directive for conditionally including content based on build tags.

    Usage:
    :::{conditional}
    :only: not ga

    This content only appears when ga tag is not present.
    :::
    """

    has_content = True
    option_spec: ClassVar[dict[str, Any]] = {
        "only": directives.unchanged,
    }

    def run(self) -> list[Node]:
        """
        Run the directive, checking the :only: condition first.
        """
        # Check if we have an :only: condition
        only_condition = self.options.get("only")

        if only_condition:
            # Parse and evaluate the condition using shared evaluator
            app = self.state.document.settings.env.app
            if not should_include_content(app, only_condition):
                # Return empty list to skip rendering this content
                logger.debug(f"Excluding conditional content due to condition: {only_condition}")
                return []

        # If no condition or condition is met, parse and return the content
        # Create a container node to hold the content
        container_node = container()

        # Parse the content as nested RST/MyST
        self.state.nested_parse(StringList(self.content), self.content_offset, container_node)

        return [container_node]


def setup(app: Sphinx) -> None:
    """
    Setup function for the conditional directives component.
    """
    # Override the default directives with our conditional versions
    app.add_directive("grid-item-card", ConditionalGridItemCardDirective, override=True)
    app.add_directive("toctree", ConditionalTocTreeDirective, override=True)
    app.add_directive("conditional", ConditionalContentDirective, override=True)
