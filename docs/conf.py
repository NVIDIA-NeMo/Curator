# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add custom extensions directory to Python path
sys.path.insert(0, os.path.abspath("_extensions"))

project = "NeMo-Curator"
project_copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.25.7"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    # "autodoc2" - Added conditionally below based on package availability
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks,
    "sphinx_design",  # For grid layout
    "sphinx.ext.ifconfig",  # For conditional content
    "content_gating",  # Unified content gating extension
    "myst_codeblock_substitutions",  # Our custom MyST substitutions in code blocks
    "json_output",  # Generate JSON output for each page
    "search_assets",  # Enhanced search assets extension
    # "ai_assistant",  # AI assistant extension
    # "swagger_plugin_for_sphinx",  # For Swagger API documentation
    "sphinxcontrib.mermaid",  # For Mermaid diagrams
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_extensions/*/README.md",  # Exclude README files in extension directories
    "_extensions/README.md",  # Exclude main extensions README
    "_extensions/*/__pycache__",  # Exclude Python cache directories
    "_extensions/*/*/__pycache__",  # Exclude nested Python cache directories
]

# -- Options for Intersphinx -------------------------------------------------
# Cross-references to external NVIDIA documentation
intersphinx_mapping = {
    "ctk": ("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest", None),
    "gpu-op": ("https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest", None),
    "ngr-tk": ("https://docs.nvidia.com/nemo/guardrails/latest", None),
    "nim-cs": ("https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/", None),
    "nim-tc": ("https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/", None),
    "nim-jd": ("https://docs.nvidia.com/nim/nemoguard-jailbreakdetect/latest/", None),
    "nim-llm": ("https://docs.nvidia.com/nim/large-language-models/latest/", None),
    "driver-linux": ("https://docs.nvidia.com/datacenter/tesla/driver-installation-guide", None),
    "nim-op": ("https://docs.nvidia.com/nim-operator/latest", None),
}

# Intersphinx timeout for slow connections
intersphinx_timeout = 30

# -- Options for JSON Output -------------------------------------------------
# Configure the JSON output extension for comprehensive search indexes
json_output_settings = {
    "enabled": True,
}

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
    "attrs_inline",  # Enables inline attributes for markdown
    "substitution",  # Enables substitution for markdown
    "html_admonition",  # Better admonition support for notes/warnings
    "html_image",  # Enhanced image handling
]

# Better MyST rendering for docstrings
myst_dmath_allow_labels = True
myst_dmath_allow_space = True
myst_dmath_allow_digits = True

myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# MyST substitutions for reusable variables across documentation
myst_substitutions = {
    "product_name": "NeMo Curator",
    "product_name_short": "Curator",
    "company": "NVIDIA",
    "version": release,
    "current_year": "2025",
    "github_repo": "https://github.com/NVIDIA/NeMo-Curator",
    "docs_url": "https://docs.nvidia.com/nemo-curator",
    "support_email": "nemo-curator-support@nvidia.com",
    "min_python_version": "3.8",
    "recommended_cuda": "12.0+",
}

# Enable figure numbering
numfig = True

# Optional: customize numbering format
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

# Optional: number within sections
numfig_secnum_depth = 1  # Gives you "Figure 1.1, 1.2, 2.1, etc."


# Suppress expected warnings for conditional content builds
suppress_warnings = [
    "toc.not_included",  # Expected when video docs are excluded from GA builds
    "toc.no_title",  # Expected for helm docs that include external README files
    "docutils",  # Expected for autodoc2-generated content with regex patterns and complex syntax
]

# -- Options for Autodoc2 ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# This should generate shorter filenames without the nemo_curator. prefix
autodoc2_packages_list = [
    # Execution backends and adapters
    "../nemo_curator/backends",
    # Pipeline orchestration
    "../nemo_curator/pipeline",
    # All processing stages (download/extract, modules, text, io, etc.)
    "../nemo_curator/stages",
    # Core task data structures
    "../nemo_curator/tasks",
    # Shared utilities
    "../nemo_curator/utils",
]

# Check if any of the packages actually exist before enabling autodoc2
autodoc2_packages = []
for pkg_path in autodoc2_packages_list:
    abs_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), pkg_path))
    if os.path.exists(abs_pkg_path):
        autodoc2_packages.append(pkg_path)

# Only include autodoc2 in extensions if we have valid packages
if autodoc2_packages:
    if "autodoc2" not in extensions:
        extensions.append("autodoc2")

    # NOTE: autodoc2 currently outputs files in a flat directory structure
    # with dot-notation filenames (e.g., nemo_curator.datasets.doc_dataset.md).
    # It does NOT support hierarchical directory organization.
    #
    # If you need hierarchical output (e.g., nemo_curator/datasets/doc_dataset.md),
    # consider using sphinx.ext.autosummary with :recursive: instead.

    # ==================== SANE DEFAULTS (good for most projects) ====================

    autodoc2_render_plugin = "myst"  # Use MyST for rendering docstrings
    autodoc2_output_dir = "apidocs"  # Output directory for autodoc2 (relative to docs/)

    # Hide implementation details - good defaults for cleaner docs
    autodoc2_hidden_objects = [
        "dunder",  # Hide __methods__ like __init__, __str__, etc.
        "private",  # Hide _private methods and variables
        "inherited",  # Hide inherited methods to reduce clutter
    ]

    # Enable module summaries for better organization
    autodoc2_module_summary = True

    # Sort by name for consistent organization
    autodoc2_sort_names = True

    # Enhanced docstring processing for better formatting
    autodoc2_docstrings = "all"  # Include all docstrings for comprehensive docs

    # Include class inheritance information - useful for users
    autodoc2_class_inheritance = True

    # Handle class docstrings properly (merge __init__ with class doc)
    autodoc2_class_docstring = "merge"

    # Better type annotation handling - use correct autodoc2 options
    autodoc2_type_guard_imports = True

    # Replace common type annotations for better readability
    autodoc2_replace_annotations = [
        ("typing.Union", "Union"),
        ("typing.Optional", "Optional"),
        ("typing.List", "List"),
        ("typing.Dict", "Dict"),
        ("typing.Any", "Any"),
        ("pandas.core.frame.DataFrame", "pd.DataFrame"),
        ("cudf.core.dataframe.DataFrame", "cudf.DataFrame"),
    ]

    # ==================== PROJECT-SPECIFIC CONFIGURATION ====================

    # Skip internal/testing modules and scripts - CUSTOMIZE FOR YOUR PROJECT
    autodoc2_skip_module_regexes = [
        r".*\.tests?.*",  # Skip test modules
        r".*\.test_.*",  # Skip test files
        r".*\._.*",  # Skip private modules
        r".*\.conftest",  # Skip conftest files
        r".*\.hello$",  # Skip hello.py example file (NeMo-Curator specific)
        r".*\.package_info$",  # Skip package metadata (NeMo-Curator specific)
        r".*\.scripts\..*",  # Skip CLI scripts (NeMo-Curator specific - documented elsewhere)
        r".*\.log$",  # Skip logging utilities (NeMo-Curator specific)
        r".*\._compat$",  # Skip compatibility modules (NeMo-Curator specific)
    ]

    # Hide specific internal utilities and constants - CUSTOMIZE FOR YOUR PROJECT
    autodoc2_hidden_regexes = [
        r".*\.MAJOR$",
        r".*\.MINOR$",
        r".*\.PATCH$",
        r".*\.VERSION$",
        r".*\.DEV$",
        r".*\.PRE_RELEASE$",
        r".*\.__.*__$",  # Hide dunder variables
        r".*\.cudf$",  # Hide import aliases (NeMo-Curator specific)
        r".*\.gpu_only_import.*",  # Hide import utilities (NeMo-Curator specific)
    ]

    # Load index template from external file for better maintainability
    template_path = os.path.join(os.path.dirname(__file__), "_templates", "autodoc2_index.rst")
    with open(template_path) as f:
        autodoc2_index_template = f.read()

    # Don't require __all__ to be defined - document all public members
    autodoc2_module_all_regexes = []  # Empty list means don't require __all__

else:
    # Remove autodoc2 from extensions if no valid packages
    if "autodoc2" in extensions:
        extensions.remove("autodoc2")
    print("INFO: autodoc2 disabled - no valid packages found in autodoc2_packages_list")

# -- Options for Napoleon (Google Style Docstrings) -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Focus on Google style only
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True  # Makes examples stand out
napoleon_use_admonition_for_notes = True  # Makes notes/warnings stand out
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True  # Cleans up type annotations in docs

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
    # Configure PyData theme search
    "search_bar_text": "Search NVIDIA docs...",
    "navbar_persistent": ["search-button"],  # Ensure search button is present
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-NeMo/Curator",
            "icon": "fa-brands fa-github",
        }
    ],
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}

# Improve readability and navigation
html_compact_lists = False  # Prevents cramped parameter lists
html_show_sourcelink = True  # Useful "View page source" links
html_copy_source = True  # Copy source files to output
html_use_index = True  # Generate comprehensive index
html_domain_indices = True  # Generate domain-specific indices (Python, etc.)

# Add our static files directory

html_extra_path = ["project.json", "versions1.json"]

# Note: JSON output configuration has been moved to the consolidated
# json_output_settings dictionary above for better organization and new features!

# Github and K8s links are now getting rate limited from the Github Actions
linkcheck_ignore = [
    ".*github\\.com.*",
    ".*githubusercontent\\.com.*",
    ".*kubernetes\\.io*",
]
