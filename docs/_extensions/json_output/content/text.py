"""Text content extraction functions."""

import re
from typing import Optional, List, Dict, Any
from docutils import nodes
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

logger = logging.getLogger(__name__)

def extract_raw_markdown(env: BuildEnvironment, docname: str) -> Optional[str]:
    """Extract raw markdown from source file."""
    try:
        source_path = env.doc2path(docname)
        if not source_path or not source_path.exists():
            return None
            
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Remove frontmatter if present
        if content.startswith('---'):
            end_marker = content.find('\n---\n', 3)
            if end_marker != -1:
                content = content[end_marker + 5:]  # Skip the second ---\n
                
        return content.strip()
        
    except Exception as e:
        logger.debug(f"Could not extract raw markdown from {docname}: {e}")
        return None

def extract_text_content(doctree: nodes.document) -> str:
    """Extract plain text content from document tree."""
    text_parts = []
    
    for node in doctree.traverse(nodes.Text):
        text_parts.append(node.astext())
    
    return ' '.join(text_parts).strip()

def extract_clean_text_content(doctree: nodes.document) -> str:
    """Extract clean text content, filtering out navigation elements."""
    text_parts = []
    
    for node in doctree.traverse():
        # Skip certain node types that aren't content
        if isinstance(node, (nodes.target, nodes.reference, nodes.substitution_definition)):
            continue
            
        # Skip toctree and other directive content
        if hasattr(node, 'tagname') and node.tagname in ['toctree', 'index', 'meta']:
            continue
            
        # Extract text from text nodes
        if isinstance(node, nodes.Text):
            text = node.astext().strip()
            if text and not text.startswith('¶'):  # Skip permalink symbols
                text_parts.append(text)
    
    # Join and clean up the text
    full_text = ' '.join(text_parts)
    
    # Clean up whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = full_text.strip()
    
    return full_text

def clean_text_for_llm(text: str) -> str:
    """Clean text content to make it more suitable for LLM processing and search indexing."""
    if not text:
        return ""
    
    # Remove SVG content (common in documentation)
    text = re.sub(r'<svg[^>]*>.*?</svg>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove empty directive blocks (common MyST artifacts)
    text = re.sub(r'^\s*```\{[^}]+\}\s*```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove toctree artifacts
    text = re.sub(r'^\s*:caption:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*:hidden:\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*:glob:\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*:maxdepth:\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common MyST directive markers that aren't useful for search
    text = re.sub(r'^\s*:::\{[^}]+\}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*:::\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up code block language indicators
    text = re.sub(r'```(\w+)\s*\n', '```\n', text)
    
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks -> double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
    
    # Remove lines that are just punctuation or symbols
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep line if it has actual words (not just punctuation/symbols)
        if stripped and re.search(r'[a-zA-Z0-9]', stripped):
            # Remove standalone punctuation at start/end
            stripped = re.sub(r'^[^\w\s]+\s*', '', stripped)
            stripped = re.sub(r'\s*[^\w\s]+$', '', stripped)
            if stripped:
                cleaned_lines.append(stripped)
    
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup
    text = text.strip()
    
    return text

def extract_directive_content(directive_block: str) -> str:
    """Extract meaningful content from MyST directive blocks."""
    if not directive_block:
        return ""
    
    # Remove the directive syntax but keep the content
    lines = directive_block.split('\n')
    content_lines = []
    in_content = False
    
    for line in lines:
        # Skip directive header lines
        if line.strip().startswith(':::') or line.strip().startswith('```{'):
            in_content = True
            continue
        elif line.strip() == ':::' or line.strip() == '```':
            continue
        elif line.strip().startswith(':') and not in_content:
            # Skip directive options
            continue
        
        # Include content lines
        if in_content or not line.strip().startswith(':'):
            content_lines.append(line)
    
    return '\n'.join(content_lines).strip()

def extract_summary(doctree: nodes.document) -> str:
    """Extract a summary from the document (first paragraph or section)."""
    # Try to find the first substantial paragraph
    for node in doctree.traverse(nodes.paragraph):
        text = node.astext().strip()
        if text and len(text) > 50:  # Substantial content
            # Clean and truncate
            text = re.sub(r'\s+', ' ', text)
            if len(text) > 300:
                text = text[:297] + '...'
            return text
    
    # Fallback: use first 300 characters of any text
    text = extract_text_content(doctree)
    if text:
        text = re.sub(r'\s+', ' ', text)
        if len(text) > 300:
            text = text[:297] + '...'
        return text
    
    return ""

def extract_keywords(content: str, headings: List[Dict[str, Any]]) -> List[str]:
    """Extract relevant keywords from content for search optimization."""
    if not content:
        return []
    
    keywords = set()
    
    # Add heading text as keywords
    for heading in headings:
        if 'text' in heading:
            # Split heading into words and add significant ones
            words = re.findall(r'\b[a-zA-Z]{3,}\b', heading['text'].lower())
            keywords.update(words)
    
    # Extract technical terms (often capitalized or have specific patterns)
    # API names, class names, function names, etc.
    tech_terms = re.findall(r'\b[A-Z][a-zA-Z0-9_]*[a-z][a-zA-Z0-9_]*\b', content)
    keywords.update(term.lower() for term in tech_terms)
    
    # Extract quoted terms (often important concepts)
    quoted_terms = re.findall(r'["`]([^"`]{3,20})["`]', content)
    for term in quoted_terms:
        if re.match(r'^[a-zA-Z][a-zA-Z0-9_\-\s]*$', term):
            keywords.add(term.lower().strip())
    
    # Extract common patterns for documentation keywords
    # Configuration keys, file extensions, command names
    config_keys = re.findall(r'\b[a-z_]+[a-z0-9_]*\s*[:=]', content)
    keywords.update(key.rstrip(':=').strip() for key in config_keys)
    
    # File extensions
    extensions = re.findall(r'\.[a-z]{2,4}\b', content.lower())
    keywords.update(ext.lstrip('.') for ext in extensions)
    
    # Remove common stop words and very short terms
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'what', 'when', 'will'}
    keywords = {kw for kw in keywords if len(kw) >= 3 and kw not in stop_words}
    
    # Return sorted list, limited to reasonable number
    return sorted(list(keywords))[:50] 