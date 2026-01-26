---
name: setup-verify
description: |
  Verify NeMo Curator installation. This skill is now part of `/setup`.
  Use `/setup` for full installation workflow with verification included.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2.0"
  deprecated: true
  redirect: "/setup"
---

# Verify Installation

**This skill has been merged into `/setup`.**

## Quick Verification

Run the verification script directly:

```bash
python dori/setup/scripts/verify_installation.py
```

Options:
- `--core` - Core framework only
- `--text` - Text curation modules
- `--video` - Video curation modules
- `--audio` - Audio curation modules
- `--image` - Image curation modules
- `--gpu` - GPU availability and RAPIDS
- `--all` - Everything (default)
- `--json` - Output as JSON for parsing
- `--verbose` - Show detailed output

## Full Setup Workflow

For complete installation with automatic environment detection and verification:

```
/setup
```

The setup workflow includes:
1. Environment detection (CUDA, GPU, existing packages)
2. Modality selection
3. Installation
4. **Verification** (this step)
5. Troubleshooting guidance
