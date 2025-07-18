"""Configuration management for JSON output extension."""

from typing import Dict, Any
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

def get_default_settings() -> Dict[str, Any]:
    """Get default configuration settings for json_output extension."""
    return {
        'enabled': True,
        'exclude_patterns': ['_build', '_templates', '_static'],
        'verbose': True,              # Enable by default for better user feedback
        'parallel': True,             # Enable parallel processing by default for speed
        'include_children': True,
        'include_child_content': True,
        'main_index_mode': 'full',  # 'disabled', 'metadata_only', 'full'
        'max_main_index_docs': 0,     # No limit by default for comprehensive search
        
        # Search optimization features
        'extract_code_blocks': True,  # Include code blocks in search data
        'extract_links': True,       # Include internal/external links
        'extract_images': True,      # Include image references
        'extract_keywords': True,    # Auto-extract technical keywords
        'include_doc_type': True,    # Auto-detect document types
        'include_section_path': True, # Include hierarchical section paths
        
        # Performance controls
        'content_max_length': 50000,  # Max content length per document (0 = no limit)
        'summary_max_length': 500,    # Max summary length
        'keywords_max_count': 50,     # Max keywords per document
        
        # Output format options
        'minify_json': True,          # Minify JSON by default for better performance
        'separate_content': False,    # Store content in separate .content.json files
        
        # Speed optimizations
        'parallel_workers': 'auto',   # Number of parallel workers
        'batch_size': 50,             # Process documents in batches
        'cache_aggressive': True,     # Enable aggressive caching
        'lazy_extraction': True,      # Only extract when needed
        'skip_large_files': 100000,   # Skip files larger than N bytes
        'incremental_build': True,    # Only process changed files
        'memory_limit_mb': 512,       # Memory limit per worker
        'fast_text_extraction': True, # Use faster text extraction
        'skip_complex_parsing': False, # Skip complex parsing features
        
        # Content filtering
        'filter_search_clutter': True, # Remove SVG, toctree, and other non-searchable content
    }

def apply_config_defaults(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values to settings dictionary."""
    defaults = get_default_settings()
    
    for key, default_value in defaults.items():
        if key not in settings:
            settings[key] = default_value
    
    return settings

def validate_config(app: Sphinx, config) -> None:
    """Validate configuration values."""
    settings = getattr(config, 'json_output_settings', {})
    
    # Ensure settings is a dictionary
    if not isinstance(settings, dict):
        logger.warning("json_output_settings must be a dictionary. Using defaults.")
        settings = {}
        config.json_output_settings = settings
    
    # Apply defaults for missing keys
    settings = apply_config_defaults(settings)
    config.json_output_settings = settings
    
    # Validate main index mode
    valid_modes = ['disabled', 'metadata_only', 'full']
    mode = settings.get('main_index_mode', 'full')
    if mode not in valid_modes:
        logger.warning(f"Invalid main_index_mode '{mode}'. Using 'full'. Valid options: {valid_modes}")
        settings['main_index_mode'] = 'full'
    
    # Validate max docs limit (0 means no limit)
    max_docs = settings.get('max_main_index_docs', 0)
    if not isinstance(max_docs, int) or max_docs < 0:
        logger.warning(f"Invalid max_main_index_docs '{max_docs}'. Using 0 (no limit).")
        settings['max_main_index_docs'] = 0
    
    # Validate content limits
    content_limit = settings.get('content_max_length', 50000)
    if not isinstance(content_limit, int) or content_limit < 0:
        logger.warning(f"Invalid content_max_length '{content_limit}'. Using 50000 (0 = no limit).")
        settings['content_max_length'] = 50000
    
    summary_limit = settings.get('summary_max_length', 500)
    if not isinstance(summary_limit, int) or summary_limit < 0:
        logger.warning(f"Invalid summary_max_length '{summary_limit}'. Using 500.")
        settings['summary_max_length'] = 500
    
    keywords_limit = settings.get('keywords_max_count', 50)
    if not isinstance(keywords_limit, int) or keywords_limit < 0:
        logger.warning(f"Invalid keywords_max_count '{keywords_limit}'. Using 50.")
        settings['keywords_max_count'] = 50
    
    # Validate exclude patterns
    patterns = settings.get('exclude_patterns', [])
    if not isinstance(patterns, list):
        logger.warning("exclude_patterns must be a list. Using default.")
        settings['exclude_patterns'] = ['_build', '_templates', '_static']
    
    # Validate boolean settings
    bool_settings = [
        'enabled', 'verbose', 'parallel', 'include_children', 'include_child_content',
        'extract_code_blocks', 'extract_links', 'extract_images', 'extract_keywords',
        'include_doc_type', 'include_section_path', 'minify_json', 'separate_content',
        'cache_aggressive', 'lazy_extraction', 'incremental_build', 
        'fast_text_extraction', 'skip_complex_parsing', 'filter_search_clutter'
    ]
    defaults = get_default_settings()
    for setting in bool_settings:
        if setting in settings and not isinstance(settings.get(setting), bool):
            logger.warning(f"Setting '{setting}' must be boolean. Using default.")
            settings[setting] = defaults[setting]
    
    # Validate integer settings
    int_settings = {
        'batch_size': (1, 1000),  # min, max
        'skip_large_files': (0, None),  # 0 = disabled
        'memory_limit_mb': (64, 8192),  # reasonable memory limits
    }
    for setting, (min_val, max_val) in int_settings.items():
        if setting in settings:
            value = settings[setting]
            if not isinstance(value, int) or value < min_val or (max_val and value > max_val):
                logger.warning(f"Setting '{setting}' must be integer between {min_val} and {max_val or 'unlimited'}. Using default.")
                settings[setting] = defaults[setting]
    
    # Validate parallel_workers (can be 'auto' or integer)
    if 'parallel_workers' in settings:
        value = settings['parallel_workers']
        if value != 'auto' and (not isinstance(value, int) or value < 1 or value > 32):
            logger.warning("Setting 'parallel_workers' must be 'auto' or integer between 1 and 32. Using default.")
            settings['parallel_workers'] = defaults['parallel_workers'] 