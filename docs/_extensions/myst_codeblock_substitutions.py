"""
Custom Sphinx extension to enable MyST substitutions in standard code blocks.

This extension pre-processes MyST markdown files to replace {{ variable }} substitutions
inside standard ``` code blocks before MyST parses the content.

Usage in any .md file:
```bash
helm install my-release oci://nvcr.io/nvidia/nemo-curator --version {{ version }}
kubectl get pods -n {{ product_name_short }}-namespace
```

The substitutions will be replaced with their values from myst_substitutions in conf.py.
"""

import re
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def process_myst_source(app, docname, source):
    """
    Process MyST source files to handle substitutions in code blocks.
    
    This is called by Sphinx's 'source-read' event for each document.
    """
    # Get substitutions from config
    substitutions = getattr(app.config, 'myst_substitutions', {})
    
    if not substitutions:
        return
    
    # Process the source content
    original_content = source[0]
    processed_content = process_codeblock_substitutions(original_content, substitutions)
    
    # Update the source if changes were made
    if processed_content != original_content:
        source[0] = processed_content
        logger.debug(f"Processed MyST substitutions in code blocks for {docname}")


def process_codeblock_substitutions(content: str, substitutions: dict) -> str:
    """
    Process MyST substitutions inside code blocks.
    
    This finds code blocks (```...```) and replaces {{ variable }} patterns
    with their values from myst_substitutions, but skips languages that 
    commonly use {{ }} syntax natively.
    """
    # Languages that commonly use {{ }} syntax and should be skipped
    TEMPLATE_LANGUAGES = {
        'yaml', 'yml', 'helm', 'jinja', 'jinja2', 'ansible', 'j2',
        'go-template', 'gotmpl', 'handlebars', 'hbs', 'mustache',
        'django', 'twig', 'liquid', 'smarty', 'docker-compose'
    }
    
    # Pattern to match code blocks: ```language\ncode\n```
    # This pattern handles multiline code blocks with optional language specifier
    code_block_pattern = r'```(\w*)\n(.*?)\n```'
    
    def replace_in_code_block(match):
        language = match.group(1).lower()
        code_content = match.group(2)
        
        # For template languages, be more selective
        if language in TEMPLATE_LANGUAGES:
            # Still process MyST substitutions, but be very careful
            processed_code = replace_substitutions_carefully(code_content, substitutions)
            return f'```{language}\n{processed_code}\n```'
        
        # Skip if the content looks like template syntax (for non-template languages)
        if is_likely_template_syntax(code_content):
            logger.debug("Skipping substitutions for code block with template-like syntax")
            return match.group(0)  # Return original unchanged
        
        # Replace substitutions in the code content
        processed_code = replace_substitutions(code_content, substitutions)
        
        # Return the code block with processed content
        return f'```{language}\n{processed_code}\n```'
    
    # Process all code blocks
    processed_content = re.sub(
        code_block_pattern, 
        replace_in_code_block, 
        content, 
        flags=re.DOTALL
    )
    
    return processed_content


def is_likely_template_syntax(content: str) -> bool:
    """
    Check if content looks like it contains template syntax that we shouldn't modify.
    
    Common patterns:
    - {{ .Values.something }} (Helm)
    - {{ ansible_variable }} (Ansible) 
    - {{ item.property }} (loops)
    - {{- .Values.something }} (Helm with whitespace control)
    """
    template_patterns = [
        r'\{\{\s*\.[\w.]+\s*\}\}',      # {{ .Values.something }}
        r'\{\{\s*ansible_\w+\s*\}\}',   # {{ ansible_variable }}
        r'\{\{\s*item\.[\w.]+\s*\}\}',  # {{ item.property }}
        r'\{\{[-+]\s*[\w.]+\s*[-+]?\}\}',  # {{- variable }} or {{ variable -}}
        r'\{\{\s*\w+\.\w+',             # {{ object.property (general)
        r'\{\{\s*range\s+',             # {{ range ... }} (Go templates)
        r'\{\{\s*if\s+',                # {{ if ... }} (conditionals)
        r'\{\{\s*with\s+',              # {{ with ... }} (Go templates)
    ]
    
    for pattern in template_patterns:
        if re.search(pattern, content):
            return True
    
    return False


def replace_substitutions(text: str, substitutions: dict) -> str:
    """
    Replace {{ variable }} patterns with their values.
    """
    def replace_var(match):
        var_name = match.group(1).strip()
        if var_name in substitutions:
            replacement = str(substitutions[var_name])
            logger.debug(f"Replacing {{ {var_name} }} with '{replacement}' in code block")
            return replacement
        else:
            logger.debug(f"Unknown substitution variable: {var_name}")
            return match.group(0)  # Return original if not found
    
    # Pattern to match {{ variable_name }} - only alphanumeric and underscore
    substitution_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
    return re.sub(substitution_pattern, replace_var, text)


def replace_substitutions_carefully(text: str, substitutions: dict) -> str:
    """
    Replace {{ variable }} patterns with their values, but only for exact MyST variable matches.
    This is used for template languages where we want to avoid breaking existing template syntax.
    """
    def replace_var(match):
        full_match = match.group(0)
        var_name = match.group(1).strip()
        
        # Only replace if it's an exact match for one of our MyST variables
        if var_name in substitutions:
            # Double-check this isn't template syntax by looking for template patterns
            if not re.search(r'[.|\-+]', full_match):  # No dots, pipes, or whitespace control
                replacement = str(substitutions[var_name])
                logger.debug(f"Carefully replacing {{ {var_name} }} with '{replacement}' in template language")
                return replacement
        
        # Leave everything else untouched
        return full_match
    
    # Pattern to match {{ variable_name }} - only alphanumeric and underscore
    substitution_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
    return re.sub(substitution_pattern, replace_var, text)


def setup(app: Sphinx):
    """
    Setup function for the MyST code block substitution extension.
    """
    # Connect to the source-read event to process files before parsing
    app.connect('source-read', process_myst_source)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 