"""Main content extraction orchestration."""

from typing import Dict, Any
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

from .text import (
    extract_raw_markdown, extract_text_content, extract_clean_text_content,
    clean_text_for_llm, extract_summary, extract_keywords
)
from .structured import extract_headings, extract_code_blocks, extract_links, extract_images

logger = logging.getLogger(__name__)

def extract_document_content(env: BuildEnvironment, docname: str, 
                           content_cache: Dict) -> Dict[str, Any]:
    """Extract content from document optimized for LLM/search use cases."""
    if docname in content_cache:
        return content_cache[docname]
        
    content = {}
    
    try:
        logger.debug(f"Starting content extraction for {docname}")
        doctree = env.get_doctree(docname)
        
        # Check for fast text extraction setting
        config = getattr(env.app, 'config', None)
        fast_extraction = getattr(config, 'json_output_settings', {}).get('fast_text_extraction', False) if config else False
        lazy_extraction = getattr(config, 'json_output_settings', {}).get('lazy_extraction', False) if config else False
        skip_complex = getattr(config, 'json_output_settings', {}).get('skip_complex_parsing', False) if config else False
        
        # Extract clean text content (always needed)
        try:
            if fast_extraction:
                # Use faster, simpler text extraction
                content['content'] = extract_text_content(doctree)
                content['format'] = 'text'
                logger.debug(f"Fast text extraction for {docname}: {len(content['content'])} chars")
            else:
                clean_text = extract_clean_text_content(doctree)
                if clean_text:
                    content['content'] = clean_text
                    content['format'] = 'text'
                    logger.debug(f"Extracted clean text content for {docname}: {len(clean_text)} chars")
                else:
                    # Fallback to raw markdown
                    raw_markdown = extract_raw_markdown(env, docname)
                    if raw_markdown:
                        content['content'] = raw_markdown
                        content['format'] = 'markdown'
                        logger.debug(f"Fallback to raw markdown for {docname}: {len(raw_markdown)} chars")
                    else:
                        content['content'] = extract_text_content(doctree)
                        content['format'] = 'text'
                        logger.debug(f"Fallback to basic text extraction for {docname}")
            
            # Apply content filtering if enabled
            filter_clutter = getattr(config, 'json_output_settings', {}).get('filter_search_clutter', True) if config else True
            if filter_clutter and content.get('content'):
                original_length = len(content['content'])
                content['content'] = clean_text_for_llm(content['content'])
                filtered_length = len(content['content'])
                if original_length != filtered_length:
                    logger.debug(f"Content filtering for {docname}: {original_length} -> {filtered_length} chars")
                    
        except Exception as e:
            logger.warning(f"Error extracting main content from {docname}: {e}")
            content['content'] = ''
            content['format'] = 'text'
        
        # Conditional extraction of additional features based on settings
        if not lazy_extraction or not skip_complex:
            # Extract additional search-relevant data with individual error handling
            try:
                content['headings'] = extract_headings(doctree)
                logger.debug(f"Extracted {len(content['headings'])} headings from {docname}")
            except Exception as e:
                logger.warning(f"Error extracting headings from {docname}: {e}")
                content['headings'] = []
            
            try:
                content['summary'] = extract_summary(doctree)
            except Exception as e:
                logger.warning(f"Error extracting summary from {docname}: {e}")
                content['summary'] = ''
            
            if not skip_complex:
                try:
                    content['code_blocks'] = extract_code_blocks(doctree)
                    logger.debug(f"Extracted {len(content['code_blocks'])} code blocks from {docname}")
                except Exception as e:
                    logger.warning(f"Error extracting code blocks from {docname}: {e}")
                    content['code_blocks'] = []
                
                try:
                    content['links'] = extract_links(doctree)
                    logger.debug(f"Extracted {len(content['links'])} links from {docname}")
                except Exception as e:
                    logger.warning(f"Error extracting links from {docname}: {e}")
                    content['links'] = []
                
                try:
                    content['images'] = extract_images(doctree)
                    logger.debug(f"Extracted {len(content['images'])} images from {docname}")
                except Exception as e:
                    logger.warning(f"Error extracting images from {docname}: {e}")
                    content['images'] = []
            else:
                # Skip complex parsing - use empty defaults
                content['code_blocks'] = []
                content['links'] = []
                content['images'] = []
            
            # Add search keywords extracted from content (if not lazy)
            if not lazy_extraction:
                try:
                    content['keywords'] = extract_keywords(content.get('content', ''), content.get('headings', []))
                    logger.debug(f"Extracted {len(content['keywords'])} keywords from {docname}")
                except Exception as e:
                    logger.warning(f"Error extracting keywords from {docname}: {e}")
                    content['keywords'] = []
            else:
                content['keywords'] = []
        else:
            # Lazy extraction - minimal processing
            content['headings'] = []
            content['summary'] = ''
            content['code_blocks'] = []
            content['links'] = []
            content['images'] = []
            content['keywords'] = []
        
        # Cache the result
        content_cache[docname] = content
        logger.debug(f"Successfully extracted content for {docname}")
        
    except Exception as e:
        logger.error(f"Critical error extracting content from {docname}: {e}")
        content = {
            'content': '',
            'format': 'text',
            'headings': [],
            'summary': '',
            'code_blocks': [],
            'links': [],
            'images': [],
            'keywords': []
        }
        content_cache[docname] = content
        
    return content_cache[docname] 