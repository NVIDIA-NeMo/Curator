"""
Document-level content filtering based on frontmatter only conditions.

Usage in document frontmatter:
---
only: not ga
---

This will only include the document when building without the GA tag.
Directory-level exclusion: If a parent directory's index.md has only requirements,
all child documents in that directory will inherit those requirements.
"""

import os
import yaml
from sphinx.application import Sphinx
from sphinx.util import logging
from .condition_evaluator import should_include_content

logger = logging.getLogger(__name__)


def get_only_condition_for_document(app: Sphinx, docname: str) -> str:
    """
    Get only condition for a document, checking the document itself
    and parent directories for inherited requirements.
    """
    source_dir = app.srcdir
    
    # Check the document itself first
    source_path = os.path.join(source_dir, docname + '.md')
    if not os.path.exists(source_path):
        source_path = os.path.join(source_dir, docname + '.rst')
    
    if os.path.exists(source_path):
        condition = extract_only_condition(source_path)
        if condition:
            return condition
    
    # Check parent directories for inherited requirements
    doc_parts = docname.split('/')
    for i in range(len(doc_parts) - 1, 0, -1):
        parent_path_parts = doc_parts[:i]
        parent_docname = '/'.join(parent_path_parts) + '/index'
        parent_source_path = os.path.join(source_dir, parent_docname + '.md')
        
        if not os.path.exists(parent_source_path):
            parent_source_path = os.path.join(source_dir, parent_docname + '.rst')
        
        if os.path.exists(parent_source_path):
            condition = extract_only_condition(parent_source_path)
            if condition:
                logger.debug(f"Document {docname} inheriting only condition '{condition}' from parent {parent_docname}")
                return condition
    
    return None


def extract_only_condition(file_path: str) -> str:
    """
    Extract only condition from a file's frontmatter.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.startswith('---'):
            return None
        
        try:
            end_marker = content.find('\n---\n', 3)
            if end_marker == -1:
                return None
            
            frontmatter_text = content[3:end_marker]
            frontmatter = yaml.safe_load(frontmatter_text)
            
            if isinstance(frontmatter, dict):
                return frontmatter.get('only')
                
        except yaml.YAMLError:
            logger.warning(f"Failed to parse frontmatter in {file_path}")
            return None
            
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return None


def should_exclude_document(app: Sphinx, docname: str) -> bool:
    """
    Check if a document should be excluded based on its only condition
    or inherited from parent directories.
    """
    only_condition = get_only_condition_for_document(app, docname)
    
    if not only_condition:
        return False
    
    # Use shared condition evaluator (invert result since we're checking for exclusion)
    should_include = should_include_content(app, only_condition)
    should_exclude = not should_include
    
    if should_exclude:
        logger.info(f"Excluding document {docname} (condition: {only_condition})")
    
    return should_exclude


def apply_build_filters(app: Sphinx, config):
    """
    Apply build filters by adding excluded documents to exclude_patterns.
    """
    # Find all markdown files in the source directory
    source_dir = app.srcdir
    markdown_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.md', '.rst')):
                rel_path = os.path.relpath(os.path.join(root, file), source_dir)
                # Convert to docname (remove extension)
                docname = os.path.splitext(rel_path)[0]
                markdown_files.append(docname)
    
    # Check each file and add to exclude_patterns if needed
    excluded_files = []
    for docname in markdown_files:
        if should_exclude_document(app, docname):
            # Add both .md and .rst versions to be safe
            config.exclude_patterns.append(docname + '.md')
            config.exclude_patterns.append(docname + '.rst')
            excluded_files.append(docname)
    
    if excluded_files:
        logger.info(f"Document filter applied: Excluding {len(excluded_files)} documents based on only conditions")


def setup(app: Sphinx):
    """
    Setup function for the document filter component.
    """
    # Connect to the config initialization event
    app.connect('config-inited', apply_build_filters) 