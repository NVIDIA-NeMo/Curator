"""
AI Assistant Extension for Sphinx
Handles AI-powered analysis and responses using external AI services
"""

import os
import shutil
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

def bundle_javascript_modules(extension_dir, output_path, minify=False):
    """Bundle all JavaScript modules into a single file."""
    
    # Define the module loading order (dependencies first)
    module_files = [
        ('ui', 'MarkdownProcessor.js'),
        ('ui', 'ResponseRenderer.js'), 
        ('core', 'ResponseProcessor.js'),
        ('core', 'AIClient.js'),
        ('core', 'main.js'),
        ('integrations', 'search-integration.js'),
    ]
    
    bundled_content = []
    bundled_content.append('// AI Assistant Bundle - Generated automatically')
    bundled_content.append('// Contains: MarkdownProcessor, ResponseRenderer, ResponseProcessor, AIClient, main, search-integration')
    bundled_content.append('')
    
    for subdir, filename in module_files:
        module_path = os.path.join(extension_dir, subdir, filename)
        if os.path.exists(module_path):
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Remove module loading code since everything is bundled
            content = content.replace('await this.loadModules();', '// Modules bundled - no loading needed')
            content = content.replace('await this.loadModuleWithFallback(name)', '// Modules bundled - no loading needed')
            
            # Simple minification if requested
            if minify:
                # Remove extra whitespace and comments (basic minification)
                import re
                # Remove single-line comments but preserve URLs
                content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)
                # Remove multi-line comments
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                # Remove extra whitespace
                content = re.sub(r'\n\s*\n', '\n', content)
                content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)
            
            bundled_content.append(f'// === {filename} ===')
            bundled_content.append(content)
            bundled_content.append('')
            
            logger.info(f'Bundled: {filename}')
        else:
            logger.warning(f'Module not found for bundling: {module_path}')
    
    # Write the bundled file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bundled_content))
    
    file_size = os.path.getsize(output_path)
    size_kb = file_size / 1024
    logger.info(f'AI Assistant JavaScript bundle created: {output_path} ({size_kb:.1f}KB)')

def add_template_path(app, config):
    """Add AI assistant template path during config initialization."""
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(extension_dir, 'assets', 'templates')
    
    if os.path.exists(templates_path):
        # Ensure templates_path is a list
        if not isinstance(config.templates_path, list):
            config.templates_path = list(config.templates_path) if config.templates_path else []
        
        # Add our template path if not already present
        if templates_path not in config.templates_path:
            config.templates_path.append(templates_path)
            logger.info(f'AI assistant templates added: {templates_path}')

def copy_assets(app, exc):
    """Copy all assets to _static after build completion."""
    if exc is not None:  # Only run if build succeeded
        return
        
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    static_path = os.path.join(app.outdir, '_static')
    os.makedirs(static_path, exist_ok=True)
    
    # Asset directories to copy
    asset_dirs = ['assets', 'core', 'ui', 'integrations']
    
    for asset_dir in asset_dirs:
        src_dir = os.path.join(extension_dir, asset_dir)
        if os.path.exists(src_dir):
            dest_dir = os.path.join(static_path, asset_dir)
            
            # Copy directory tree, preserving structure
            try:
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                logger.info(f'AI assistant assets copied: {asset_dir}/')
            except Exception as e:
                logger.warning(f'Failed to copy {asset_dir}/: {e}')

def copy_assets_early(app, docname, source):
    """Copy bundled assets to _static at the start of build process."""
    # Only copy once - use a flag to prevent multiple copies
    if hasattr(app, '_ai_assistant_assets_copied'):
        return
    
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    static_path = os.path.join(app.outdir, '_static')
    os.makedirs(static_path, exist_ok=True)
    
    # Create bundled JavaScript file instead of copying individual modules
    bundle_path = os.path.join(static_path, 'ai-assistant.bundle.js')
    bundle_javascript_modules(extension_dir, bundle_path)
    
    # Copy CSS assets if they exist
    assets_dir = os.path.join(extension_dir, 'assets')
    if os.path.exists(assets_dir):
        dest_assets_dir = os.path.join(static_path, 'assets')
        try:
            if os.path.exists(dest_assets_dir):
                shutil.rmtree(dest_assets_dir)
            shutil.copytree(assets_dir, dest_assets_dir)
            logger.info('AI assistant CSS assets copied')
        except Exception as e:
            logger.warning(f'Failed to copy CSS assets: {e}')
    
    # Mark as copied
    app._ai_assistant_assets_copied = True

def setup(app: Sphinx):
    """Setup the AI assistant extension."""
    
    # Get the directory where this extension is located
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Connect to config-inited event to add template path
    app.connect('config-inited', add_template_path)
    
    # Copy assets early in the build process so JS modules are available
    app.connect('source-read', copy_assets_early)
    
    # Also copy assets when build is finished (backup)
    app.connect('build-finished', copy_assets)
    
    # Add CSS files (from assets/styles/)
    css_file = os.path.join(extension_dir, 'assets', 'styles', 'ai-assistant.css')
    if os.path.exists(css_file):
        app.add_css_file('assets/styles/ai-assistant.css')
        logger.info('AI assistant CSS loaded')
    else:
        logger.warning(f'AI assistant CSS not found at {css_file}')
    
    # Add the bundled JavaScript file (contains all modules)
    app.add_js_file('ai-assistant.bundle.js')
    logger.info('AI assistant bundled JS will be loaded')
    
    # Add configuration values
    app.add_config_value('ai_assistant_enabled', True, 'env')
    app.add_config_value('ai_assistant_endpoint', 'https://prod-1-data.ke.pinecone.io/assistant/chat/test-assistant', 'env')
    app.add_config_value('ai_assistant_api_key', '', 'env')
    app.add_config_value('ai_trigger_threshold', 2, 'env')
    app.add_config_value('ai_auto_trigger', True, 'env')
    
    return {
        'version': '1.0.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 