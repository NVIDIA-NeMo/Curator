"""Structured content extraction functions for headings, code blocks, links, and images."""

import re
from typing import List, Dict, Any
from docutils import nodes
from sphinx.util import logging

logger = logging.getLogger(__name__)

def extract_headings(doctree: nodes.document) -> List[Dict[str, Any]]:
    """Extract headings from document tree."""
    headings = []
    
    # Extract headings from section nodes
    for node in doctree.traverse(nodes.section):
        # Get the title node
        title_node = node.next_node(nodes.title)
        if title_node:
            title_text = title_node.astext().strip()
            if title_text:
                # Determine heading level based on nesting
                level = 1
                parent = node.parent
                while parent and isinstance(parent, nodes.section):
                    level += 1
                    parent = parent.parent
                
                # Generate ID (similar to how Sphinx does it)
                heading_id = re.sub(r'[^\w\-_]', '', title_text.lower().replace(' ', '-'))
                
                headings.append({
                    'text': title_text,
                    'level': level,
                    'id': heading_id
                })
    
    # Also check for standalone title nodes (like document title)
    for node in doctree.traverse(nodes.title):
        if node.parent and not isinstance(node.parent, nodes.section):
            title_text = node.astext().strip()
            if title_text:
                heading_id = re.sub(r'[^\w\-_]', '', title_text.lower().replace(' ', '-'))
                headings.append({
                    'text': title_text,
                    'level': 1,
                    'id': heading_id
                })
    
    # Remove duplicates while preserving order
    seen = set()
    unique_headings = []
    for heading in headings:
        heading_key = (heading['text'], heading['level'])
        if heading_key not in seen:
            seen.add(heading_key)
            unique_headings.append(heading)
    
    return unique_headings

def extract_code_blocks(doctree: nodes.document) -> List[Dict[str, Any]]:
    """Extract code blocks from document tree."""
    code_blocks = []
    
    for node in doctree.traverse(nodes.literal_block):
        code_content = node.astext().strip()
        if code_content:
            # Try to determine language from classes or attributes
            language = 'text'  # default
            
            if hasattr(node, 'attributes') and 'classes' in node.attributes:
                classes = node.attributes['classes']
                for cls in classes:
                    if cls.startswith('language-'):
                        language = cls[9:]  # Remove 'language-' prefix
                        break
                    elif cls in ['python', 'bash', 'javascript', 'json', 'yaml', 'sql', 'html', 'css', 'cpp', 'c', 'java', 'rust', 'go']:
                        language = cls
                        break
                        
            # Also check for highlight language
            if hasattr(node, 'attributes') and 'highlight_args' in node.attributes:
                highlight_args = node.attributes['highlight_args']
                if 'language' in highlight_args:
                    language = highlight_args['language']
            
            code_blocks.append({
                'content': code_content,
                'language': language
            })
    
    return code_blocks

def extract_links(doctree: nodes.document) -> List[Dict[str, Any]]:
    """Extract links from document tree."""
    links = []
    
    for node in doctree.traverse(nodes.reference):
        link_text = node.astext().strip()
        if not link_text:
            continue
            
        link_url = ''
        link_type = 'internal'  # default
        
        # Get URL from various possible attributes
        if hasattr(node, 'attributes'):
            attrs = node.attributes
            if 'refuri' in attrs:
                link_url = attrs['refuri']
                # Determine if external or internal
                if link_url.startswith(('http://', 'https://', 'ftp://', 'mailto:')):
                    link_type = 'external'
                elif link_url.startswith('#'):
                    link_type = 'anchor'
                else:
                    link_type = 'internal'
            elif 'refid' in attrs:
                link_url = '#' + attrs['refid']
                link_type = 'anchor'
            elif 'reftarget' in attrs:
                link_url = attrs['reftarget']
                link_type = 'internal'
        
        if link_text and link_url:
            links.append({
                'text': link_text,
                'url': link_url,
                'type': link_type
            })
    
    return links

def extract_images(doctree: nodes.document) -> List[Dict[str, Any]]:
    """Extract images from document tree."""
    images = []
    
    for node in doctree.traverse(nodes.image):
        if hasattr(node, 'attributes'):
            attrs = node.attributes
            
            # Get image source
            image_src = attrs.get('uri', '')
            if not image_src:
                continue
                
            # Get alt text
            alt_text = attrs.get('alt', '')
            
            # Get title if available
            title = attrs.get('title', '')
            
            # Get width/height if specified
            width = attrs.get('width', '')
            height = attrs.get('height', '')
            
            image_info = {
                'src': image_src,
                'alt': alt_text
            }
            
            if title:
                image_info['title'] = title
            if width:
                image_info['width'] = width
            if height:
                image_info['height'] = height
                
            images.append(image_info)
    
    # Also check for figure nodes (which contain images)
    for node in doctree.traverse(nodes.figure):
        # Find image within figure
        for img_node in node.traverse(nodes.image):
            if hasattr(img_node, 'attributes'):
                attrs = img_node.attributes
                image_src = attrs.get('uri', '')
                if not image_src:
                    continue
                    
                # Get caption from figure
                caption = ''
                for caption_node in node.traverse(nodes.caption):
                    caption = caption_node.astext().strip()
                    break
                
                image_info = {
                    'src': image_src,
                    'alt': attrs.get('alt', ''),
                    'caption': caption
                }
                
                # Add other attributes
                if 'title' in attrs:
                    image_info['title'] = attrs['title']
                if 'width' in attrs:
                    image_info['width'] = attrs['width']
                if 'height' in attrs:
                    image_info['height'] = attrs['height']
                    
                images.append(image_info)
    
    return images 