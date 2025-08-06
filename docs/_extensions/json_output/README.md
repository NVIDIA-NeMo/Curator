# JSON Output Extension

Sphinx extension to generate JSON output for every page alongside HTML output.

Similar to Hugo's output formats, this creates parallel JSON files for each document
containing metadata, content, and other structured data that can be consumed by
search engines, APIs, or other applications.

The main use case is generating comprehensive search indexes for tools like Solr, 
Lunr.js, or custom search implementations.

## Search Index Integration

The main index.json file contains all documents with full content, perfect for:

- **Lunr.js**: Load index.json and build search index from documents
- **Solr**: POST the JSON data to Solr's update endpoint
- **Elasticsearch**: Bulk index the documents array
- **Custom search**: Parse JSON and implement your own search logic

## Enhanced JSON Structure

The JSON structure includes search-optimized fields:

```json
{
    "id": "guide/installation",
    "title": "Installation Guide", 
    "content": "Full markdown content here...",
    "content_length": 5420,
    "word_count": 850,
    "format": "text",
    "summary": "Quick summary for previews...",
    "doc_type": "tutorial",
    "section_path": ["Guide", "Installation"],
    "headings": [
        {"text": "Prerequisites", "level": 2, "id": "prerequisites"}
    ],
    "headings_text": "Prerequisites Installation Steps Troubleshooting",
    "keywords": ["install", "setup", "prerequisites", "docker", "python"],
    "code_blocks": [
        {"content": "pip install package", "language": "bash"}
    ],
    "links": [
        {"text": "API Reference", "url": "/api", "type": "internal"}
    ],
    "tags": ["setup", "guide"],
    "categories": ["tutorials"]
}
```

## Configuration Examples

### Minimal Configuration (Recommended)

Uses optimized defaults for best performance:

```python
# conf.py
json_output_settings = {
    'enabled': True,  # All other settings use performance-optimized defaults
}
```

### Comprehensive Search Index (Default Behavior)

```python
json_output_settings = {
    'enabled': True,
    'verbose': True,               # Default: detailed logging
    'parallel': True,              # Default: parallel processing
    'main_index_mode': 'full',     # Default: full content
    'max_main_index_docs': 0,      # Default: no limit
    'minify_json': True,           # Default: smaller files
    'filter_search_clutter': True, # Default: clean content
}
```

### Large Sites Configuration

```python
json_output_settings = {
    'enabled': True,
    'max_main_index_docs': 500,    # Limit to 500 documents
    'content_max_length': 20000,   # Limit content length
    'skip_large_files': 50000,     # Skip files over 50KB
}
```

### Fastest Builds (Minimal Features)

```python
json_output_settings = {
    'enabled': True,
    'main_index_mode': 'metadata_only',  # Only titles, descriptions, tags
    'extract_code_blocks': False,        # Skip code extraction
    'extract_links': False,              # Skip link extraction  
    'lazy_extraction': True,             # Minimal processing
    'skip_complex_parsing': True,        # Skip complex features
}
```

## Available Settings

### Core Settings

- **enabled** (bool): Enable/disable JSON output generation. Default: `True`
- **verbose** (bool): Enable verbose logging. Default: `True`
- **parallel** (bool): Enable parallel processing. Default: `True`
- **exclude_patterns** (list): Patterns to exclude from JSON generation. Default: `['_build', '_templates', '_static']`
- **include_children** (bool): Include child documents in directory indexes. Default: `True`
- **include_child_content** (bool): Include full content in child documents. Default: `True`
- **main_index_mode** (str): How to handle main index page. Options: `'disabled'`, `'metadata_only'`, `'full'`. Default: `'full'`
- **max_main_index_docs** (int): Maximum documents to include in main index (0 = no limit). Default: `0`

### Search Optimization Features

- **extract_code_blocks** (bool): Include code blocks in search data. Default: `True`
- **extract_links** (bool): Include internal/external links. Default: `True`
- **extract_images** (bool): Include image references. Default: `True`
- **extract_keywords** (bool): Auto-extract technical keywords. Default: `True`
- **include_doc_type** (bool): Auto-detect document types (tutorial, guide, reference, etc.). Default: `True`
- **include_section_path** (bool): Include hierarchical section paths. Default: `True`

### Performance Controls

- **content_max_length** (int): Max content length per document (0 = no limit). Default: `50000`
- **summary_max_length** (int): Max summary length. Default: `500`
- **keywords_max_count** (int): Max keywords per document. Default: `50`

### Output Format Options

- **minify_json** (bool): Minify JSON output (removes indentation for smaller files). Default: `True`
- **separate_content** (bool): Store content in separate .content.json files for better performance. Default: `False`

### Speed Optimizations

- **parallel_workers** (str): Number of parallel workers. Default: `'auto'`
- **batch_size** (int): Process documents in batches. Default: `50`
- **cache_aggressive** (bool): Enable aggressive caching. Default: `True`
- **lazy_extraction** (bool): Only extract when needed. Default: `True`
- **skip_large_files** (int): Skip files larger than N bytes. Default: `100000`
- **incremental_build** (bool): Only process changed files. Default: `True`
- **memory_limit_mb** (int): Memory limit per worker. Default: `512`
- **fast_text_extraction** (bool): Use faster text extraction. Default: `True`
- **skip_complex_parsing** (bool): Skip complex parsing features. Default: `False`

### Content Filtering

- **filter_search_clutter** (bool): Remove SVG, toctree, and other non-searchable content. Default: `True`

## Content Gating Integration

This extension automatically respects content gating rules set by the content_gating extension at multiple levels:

### Document-Level Gating
Documents with 'only' conditions in frontmatter that fail evaluation (e.g., 'only: not ga' when building with -t ga) will be excluded from JSON generation entirely, ensuring sensitive content doesn't leak into search indexes.

### Content-Level Gating  
Content sections wrapped in `{conditional}` directives are also properly filtered. When conditions don't match, the content is excluded from the document tree and won't appear in the generated JSON.

### Integration Details
- **Automatic Detection**: Detects if content_gating extension is loaded
- **Exclude Pattern Sync**: Respects documents added to exclude_patterns by content gating
- **Build Tag Awareness**: Logs current build tags for debugging
- **Debug Logging**: Provides detailed logs when content gating rules are applied

The integration works seamlessly - just enable both extensions and your JSON output will automatically respect all content gating rules without additional configuration.

## Performance Tips

1. **Enable parallel processing** for faster builds on multi-core systems
2. **Use incremental builds** to only process changed files
3. **Set content length limits** for large documentation sites
4. **Enable content filtering** to reduce JSON file sizes
5. **Use batch processing** to control memory usage
6. **Skip large files** to avoid processing massive documents 