"""
Content Gating Extension for Sphinx

Provides conditional content rendering based on release stage tags.
Supports filtering at multiple levels:
- Document level (via frontmatter)
- Toctree level (global and per-entry)
- Grid card level

Usage:
- Add tags during build: sphinx-build -t ga docs/ _build/
- Use :only: conditions in directives and frontmatter
- Supports conditions like 'ga', 'not ga', 'ea', 'not ea', 'internal', 'not internal'
"""

from sphinx.application import Sphinx
from .document_filter import setup as setup_document_filter
from .conditional_directives import setup as setup_conditional_directives


def setup(app: Sphinx):
    """
    Setup function for the content gating extension.
    """
    # Setup document-level filtering
    setup_document_filter(app)
    
    # Setup conditional directives (toctree and grid-item-card)
    setup_conditional_directives(app)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 