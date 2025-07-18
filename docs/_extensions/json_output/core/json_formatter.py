"""JSON data formatting and structure building."""

from datetime import datetime
from typing import Any, Dict

from docutils import nodes
from sphinx.util import logging

from ..utils import get_document_url, get_setting

logger = logging.getLogger(__name__)


class JSONFormatter:
    """Handles JSON data structure building and formatting."""
    
    def __init__(self, app, json_builder):
        self.app = app
        self.env = app.env
        self.config = app.config
        self.json_builder = json_builder
    
    def add_metadata_fields(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Add all metadata fields to JSON data structure."""
        # Basic metadata fields
        if metadata.get('description'):
            data['description'] = metadata['description']
        if metadata.get('tags'):
            data['tags'] = metadata['tags'] if isinstance(metadata['tags'], list) else [metadata['tags']]
        if metadata.get('categories'):
            data['categories'] = metadata['categories'] if isinstance(metadata['categories'], list) else [metadata['categories']]
        if metadata.get('author'):
            data['author'] = metadata['author']
        
        # Rich frontmatter taxonomy fields
        if metadata.get('personas'):
            data['personas'] = metadata['personas'] if isinstance(metadata['personas'], list) else [metadata['personas']]
        if metadata.get('difficulty'):
            data['difficulty'] = metadata['difficulty']
        if metadata.get('content_type'):
            data['content_type'] = metadata['content_type']
        if metadata.get('modality'):
            data['modality'] = metadata['modality']
        if metadata.get('only'):
            data['only'] = metadata['only']
    
    def build_child_json_data(self, docname: str, include_content: bool = None) -> Dict[str, Any]:
        """Build optimized JSON data for child documents (LLM/search focused)."""
        if include_content is None:
            include_content = get_setting(self.config, 'include_child_content', True)
            
        # Get document title
        title = self.env.titles.get(docname, nodes.title()).astext() if docname in self.env.titles else ''
        
        # Extract metadata for tags/categories
        metadata = self.json_builder.extract_document_metadata(docname)
        content_data = self.json_builder.extract_document_content(docname) if include_content else {}
        
        # Build optimized data structure for search engines
        data = {
            'id': docname,  # Use 'id' for search engines
            'title': title,
            'url': get_document_url(self.app, docname),
        }
        
        # Add metadata fields
        self.add_metadata_fields(data, metadata)
        
        # Add search-specific fields
        if include_content:
            self._add_content_fields(data, content_data, docname, title)
        
        return data
    
    def build_json_data(self, docname: str) -> Dict[str, Any]:
        """Build optimized JSON data structure for LLM/search use cases."""
        # Get document title
        title = self.env.titles.get(docname, nodes.title()).astext() if docname in self.env.titles else ''
        
        # Extract metadata and content
        metadata = self.json_builder.extract_document_metadata(docname)
        content_data = self.json_builder.extract_document_content(docname)
        
        # Build data structure
        data = {
            'id': docname,
            'title': title,
            'url': get_document_url(self.app, docname),
            'last_modified': datetime.now().isoformat(),
        }
        
        # Add metadata fields
        self.add_metadata_fields(data, metadata)
        
        # Add content
        if 'content' in content_data and content_data['content']:
            data['content'] = content_data['content']
            data['format'] = content_data.get('format', 'text')
        
        if 'summary' in content_data and content_data['summary']:
            data['summary'] = content_data['summary']
        
        if 'headings' in content_data and content_data['headings']:
            data['headings'] = [
                {'text': h['text'], 'level': h['level']} 
                for h in content_data['headings']
            ]
        
        return data
    
    def _add_content_fields(self, data: Dict[str, Any], content_data: Dict[str, Any], 
                           docname: str, title: str) -> None:
        """Add content-related fields to JSON data."""
        # Apply content length limit
        content_max_length = get_setting(self.config, 'content_max_length', 50000)
        
        # Primary content (markdown or text)
        if 'content' in content_data and content_data['content']:
            content = content_data['content']
            if content_max_length > 0 and len(content) > content_max_length:
                content = content[:content_max_length] + '...'
            
            data['content'] = content
            data['format'] = content_data.get('format', 'text')
            
            # Add search-optimized fields
            data['content_length'] = len(content_data['content'])  # Original length
            data['word_count'] = len(content_data['content'].split()) if content_data['content'] else 0
        
        # Summary with length limit
        if 'summary' in content_data and content_data['summary']:
            summary_max_length = get_setting(self.config, 'summary_max_length', 500)
            summary = content_data['summary']
            if summary_max_length > 0 and len(summary) > summary_max_length:
                summary = summary[:summary_max_length] + '...'
            data['summary'] = summary
        
        # Headings for structure/navigation
        if 'headings' in content_data and content_data['headings']:
            # Simplify headings for LLM use
            data['headings'] = [
                {'text': h['text'], 'level': h['level'], 'id': h.get('id', '')} 
                for h in content_data['headings']
            ]
            
            # Add searchable heading text
            data['headings_text'] = ' '.join([h['text'] for h in content_data['headings']])
        
        # Optional search enhancement features
        if get_setting(self.config, 'extract_keywords', True) and 'keywords' in content_data:
            keywords_max_count = get_setting(self.config, 'keywords_max_count', 50)
            keywords = content_data['keywords'][:keywords_max_count] if keywords_max_count > 0 else content_data['keywords']
            data['keywords'] = keywords
        
        if get_setting(self.config, 'extract_code_blocks', True) and 'code_blocks' in content_data:
            data['code_blocks'] = content_data['code_blocks']
        
        if get_setting(self.config, 'extract_links', True) and 'links' in content_data:
            data['links'] = content_data['links']
        
        if get_setting(self.config, 'extract_images', True) and 'images' in content_data:
            data['images'] = content_data['images']
        
        # Add document type detection
        if get_setting(self.config, 'include_doc_type', True):
            # Import here to avoid circular import
            from .document_discovery import DocumentDiscovery
            discovery = DocumentDiscovery(self.app, self.json_builder)
            data['doc_type'] = discovery.detect_document_type(docname, title, content_data.get('content', ''))
        
        # Add section hierarchy
        if get_setting(self.config, 'include_section_path', True):
            from .document_discovery import DocumentDiscovery
            discovery = DocumentDiscovery(self.app, self.json_builder)
            data['section_path'] = discovery.get_section_path(docname) 