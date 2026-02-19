---
name: schema-update
description: Regenerate and validate the Agent Tool Schema after NeMo Curator changes. Use when NeMo Curator has been updated and the schema needs to be refreshed, or when contributing new stages.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  type: developer
compatibility: Requires NeMo Curator to be installed. For developers/contributors.
---

# Schema Update Skill

Regenerate the Agent Tool Schema when NeMo Curator changes. This keeps the schema in sync with the actual codebase.

## When This Skill Applies

- NeMo Curator has been updated
- New stages/filters/classifiers have been added
- User is contributing to NeMo Curator
- Schema is out of date or missing

## Skill Workflow

### Step 1: Check Current State

```bash
# Check if schema exists
ls -la skills/shared/schemas/agent-tool-schema.json

# Check schema metadata
python -c "
import json
schema = json.load(open('skills/shared/schemas/agent-tool-schema.json'))
print(f\"Generated: {schema['metadata']['generated_at']}\")
print(f\"Operations: {len(schema['operations'])}\")
"
```

### Step 2: Regenerate Schema

```bash
python skills/shared/scripts/generate_agent_tool_schema.py \
    --package nemo_curator \
    --output skills/shared/schemas/agent-tool-schema.json
```

With statistics:
```bash
python skills/shared/scripts/generate_agent_tool_schema.py \
    --package nemo_curator \
    --stats
```

### Step 3: Validate New Schema

```bash
python skills/shared/scripts/validate_agent_tool_schema.py \
    skills/shared/schemas/agent-tool-schema.json \
    --verbose
```

For strict validation (CI):
```bash
python skills/shared/scripts/validate_agent_tool_schema.py \
    skills/shared/schemas/agent-tool-schema.json \
    --strict --json
```

### Step 4: Review Changes

```bash
git diff skills/shared/schemas/agent-tool-schema.json
```

Look for:
- New operations added
- Parameters changed
- GPU requirements updated
- Types modified

### Step 5: Commit

```bash
git add skills/shared/schemas/agent-tool-schema.json
git commit -m "chore: update agent tool schema for NeMo Curator vX.Y.Z"
```

## Adding New Operations

When adding a new stage to NeMo Curator:

### 1. Ensure Proper Class Structure

```python
@dataclass
class MyNewStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Short description for the schema.
    
    Longer description here.
    """
    
    name: str = "MyNewStage"
    resources: Resources = Resources(cpus=1.0, gpu_memory_gb=4.0)
    
    # Parameters with type hints and defaults
    my_param: int = 100
    another_param: str = "default"
```

### 2. Update Module Paths (if needed)

If your stage is in a new module, add it to `generate_agent_tool_schema.py`:

```python
NEMO_CURATOR_CONFIG = IntrospectionConfig(
    module_paths=[
        # ... existing paths ...
        "nemo_curator.stages.text.my_new_module",  # Add new module
    ],
)
```

### 3. Add GPU Memory Estimate (if GPU)

```python
gpu_memory_estimates={
    # ... existing ...
    "MyNewStage": 4.0,  # GB
},
```

### 4. Add Hints (optional but recommended)

```python
operation_hints={
    "MyNewStage": {
        "purpose": "What this stage does",
        "when_to_use": "When to recommend this stage",
        "typical_retention": "Expected data retention percentage",
    },
},
```

### 5. Regenerate and Validate

```bash
python skills/shared/scripts/generate_agent_tool_schema.py \
    --package nemo_curator \
    --output skills/shared/schemas/agent-tool-schema.json

python skills/shared/scripts/validate_agent_tool_schema.py \
    skills/shared/schemas/agent-tool-schema.json \
    --verbose
```

## CI Integration

Add to CI pipeline:

```yaml
name: Validate Agent Tool Schema

on:
  push:
    paths:
      - 'nemo_curator/stages/**'
      - 'skills/shared/schemas/**'

jobs:
  validate-schema:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: pip install nemo-curator pyyaml jsonschema
      
      - name: Regenerate schema
        run: |
          python skills/shared/scripts/generate_agent_tool_schema.py \
            --package nemo_curator \
            --output skills/shared/schemas/agent-tool-schema.json
      
      - name: Validate schema
        run: |
          python skills/shared/scripts/validate_agent_tool_schema.py \
            skills/shared/schemas/agent-tool-schema.json \
            --strict --json
      
      - name: Check for uncommitted changes
        run: |
          if [[ -n $(git status --porcelain) ]]; then
            echo "Schema is out of date. Please regenerate."
            git diff
            exit 1
          fi
```

## Troubleshooting

### "Package not installed"

```
Warning: Package nemo_curator not installed
```

Install NeMo Curator first:
```bash
pip install nemo-curator
# or
pip install -e .  # if in repo root
```

### "Module not found"

A module in `module_paths` doesn't exist. Check the path is correct or remove it from the config.

### "Warning: Could not load base class"

The base class path has changed. Update `operation_base_classes` in the config.

### Validation errors

Run with `--verbose` to see suggestions:
```bash
python validate_agent_tool_schema.py schema.json --verbose
```
