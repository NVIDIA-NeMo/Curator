#!/bin/bash
# Check for unconverted MyST syntax in Fern docs

set -e

PAGES_DIR="${1:-pages}"

echo "=== Checking for unconverted MyST syntax in $PAGES_DIR ==="
echo ""

# Track if any issues found
ISSUES_FOUND=0

# Check for MyST directive syntax
echo "Checking for MyST directives (:::)..."
if grep -r ':::' "$PAGES_DIR" 2>/dev/null; then
    echo "⚠️  Found unconverted MyST directives"
    ISSUES_FOUND=1
else
    echo "✓ No MyST directives found"
fi
echo ""

# Check for {ref} references
echo "Checking for {ref} references..."
if grep -r '{ref}' "$PAGES_DIR" 2>/dev/null; then
    echo "⚠️  Found unconverted {ref} references"
    ISSUES_FOUND=1
else
    echo "✓ No {ref} references found"
fi
echo ""

# Check for {octicon} icons
echo "Checking for {octicon} icons..."
if grep -r '{octicon}' "$PAGES_DIR" 2>/dev/null; then
    echo "⚠️  Found unconverted {octicon} icons"
    ISSUES_FOUND=1
else
    echo "✓ No {octicon} icons found"
fi
echo ""

# Check for badges
echo "Checking for sphinx-design badges..."
if grep -r '{bdg-' "$PAGES_DIR" 2>/dev/null; then
    echo "⚠️  Found unconverted badges"
    ISSUES_FOUND=1
else
    echo "✓ No badges found"
fi
echo ""

# Check for MyST mermaid syntax
echo "Checking for MyST mermaid syntax..."
if grep -r '```{mermaid}' "$PAGES_DIR" 2>/dev/null; then
    echo "⚠️  Found unconverted mermaid blocks (should be \`\`\`mermaid)"
    ISSUES_FOUND=1
else
    echo "✓ No MyST mermaid syntax found"
fi
echo ""

# Check for unconverted substitutions
echo "Checking for unconverted variable substitutions..."
if grep -r '{{ ' "$PAGES_DIR" 2>/dev/null; then
    echo "⚠️  Found unconverted substitutions (run substitute_variables.py)"
    ISSUES_FOUND=1
else
    echo "✓ No unconverted substitutions found"
fi
echo ""

# Check for HTML attributes that need camelCase in MDX
echo "Checking for HTML attributes needing camelCase..."
if grep -r 'autoplay\|frameborder\|allowfullscreen\|playsinline' "$PAGES_DIR" 2>/dev/null | grep -v 'autoPlay\|frameBorder\|allowFullScreen\|playsInline'; then
    echo "⚠️  Found HTML attributes that should be camelCase in MDX"
    ISSUES_FOUND=1
else
    echo "✓ No HTML attributes needing conversion"
fi
echo ""

# Summary
echo "=== Summary ==="
if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✓ All checks passed"
    exit 0
else
    echo "⚠️  Some issues found - review and fix above"
    exit 1
fi
