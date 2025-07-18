"""
Shared condition evaluation logic for content gating.
"""

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def should_include_content(app: Sphinx, condition: str) -> bool:
    """
    Evaluate an :only: condition against current build tags.
    
    Supports conditions like:
    - 'ga' - include only if 'ga' tag is present
    - 'not ga' - include only if 'ga' tag is not present
    - 'ea' - include only if 'ea' tag is present
    - 'not ea' - include only if 'ea' tag is not present
    - 'internal' - include only if 'internal' tag is present
    - 'not internal' - include only if 'internal' tag is not present
    
    Args:
        app: Sphinx application instance
        condition: The condition string to evaluate
        
    Returns:
        True if content should be included, False otherwise
    """
    try:
        # Get current build tags
        current_tags = set()
        
        # Use newer Sphinx API for tags
        if hasattr(app, 'tags'):
            try:
                # For Sphinx 9.0+ - tags object supports iteration
                current_tags = set(app.tags)
            except TypeError:
                # Fallback for older versions that still use .tags attribute
                if hasattr(app.tags, 'tags'):
                    current_tags = app.tags.tags
        
        # Parse the condition
        condition = condition.strip()
        
        if condition.startswith('not '):
            # Negated condition
            tag = condition[4:].strip()
            result = tag not in current_tags
            logger.debug(f"Condition 'not {tag}' evaluated to {result} (current tags: {current_tags})")
            return result
        else:
            # Positive condition
            tag = condition.strip()
            result = tag in current_tags
            logger.debug(f"Condition '{tag}' evaluated to {result} (current tags: {current_tags})")
            return result
            
    except Exception as e:
        logger.warning(f"Error evaluating :only: condition '{condition}': {e}")
        # Default to including the content if there's an error
        return True 