#!/usr/bin/env python3
"""Remove all {toctree} blocks from MDX files."""
import re
from pathlib import Path

pages_dir = Path(__file__).parent.parent / "pages"

for mdx_file in pages_dir.rglob("*.mdx"):
    content = mdx_file.read_text()
    
    # Remove toctree blocks
    new_content = re.sub(r"```\{toctree\}.*?```", "", content, flags=re.DOTALL)
    
    # Clean up multiple newlines
    new_content = re.sub(r"\n{3,}", "\n\n", new_content)
    new_content = new_content.rstrip() + "\n"
    
    if content != new_content:
        mdx_file.write_text(new_content)
        print(f"Fixed: {mdx_file}")

print("Done!")
