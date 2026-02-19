---
name: schema
description: Query NeMo Curator's Agent Tool Schema to discover available operations, their parameters, GPU requirements, and composition rules. Use when the user asks what operations exist, what parameters a stage takes, or how to compose stages.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  type: discovery
---

# Schema Discovery Skill

Help users discover what NeMo Curator operations are available and how to use them, using the Agent Tool Schema as the source of truth.

## When This Skill Applies

- User asks "what filters are available?"
- User asks "what parameters does X take?"
- User asks "does X require GPU?"
- User asks "what can connect to what?"
- User wants to explore NeMo Curator capabilities

## Schema Location

The Agent Tool Schema is at:
```
skills/shared/schemas/agent-tool-schema.json
```

If it doesn't exist, generate it:
```bash
python skills/shared/scripts/generate_agent_tool_schema.py \
    --package nemo_curator \
    --output skills/shared/schemas/agent-tool-schema.json
```

## Skill Workflow

### Step 1: Load the Schema

Read the schema file to get the source of truth:
```python
import json
schema = json.load(open("skills/shared/schemas/agent-tool-schema.json"))
```

### Step 2: Answer User Questions

#### "What operations are available?"

List from `schema["operations"]`:
```python
for name, op in schema["operations"].items():
    print(f"{name}: {op['description']}")
    print(f"  Categories: {op['category']}")
    print(f"  GPU: {op['resources']['requires_gpu']}")
```

#### "What parameters does X take?"

Look up in schema:
```python
op = schema["operations"]["WordCountFilter"]
for param, info in op["parameters"].items():
    print(f"{param}: {info['type']}, default={info.get('default')}")
```

#### "Does X require GPU?"

Check resources:
```python
op = schema["operations"]["QualityClassifier"]
print(f"GPU required: {op['resources']['requires_gpu']}")
print(f"GPU memory: {op['resources'].get('gpu_memory_gb', 'N/A')} GB")
```

#### "What can I connect after X?"

Use composition rules:
```python
# What produces DocumentBatch?
producers = schema["composition"]["type_flow"]["DocumentBatch"]["producers"]

# What consumes DocumentBatch?
consumers = schema["composition"]["type_flow"]["DocumentBatch"]["consumers"]
```

### Step 3: Format Response

Present information clearly:

```
## WordCountFilter

**Description:** Filter documents by word count.

**Import:**
```python
from nemo_curator.stages.text.filters import WordCountFilter
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| min_words | integer | 50 | Minimum word count |
| max_words | integer | 100000 | Maximum word count |

**Resources:**
- GPU required: No
- Categories: text, filter
```

## Example Interactions

### Example 1: Discover filters
```
User: What text filters are available?

Agent: [Loads schema, filters by category="filter" and "text"]

Available text filters:

| Filter | Description | GPU |
|--------|-------------|-----|
| WordCountFilter | Filter by word count | No |
| NonAlphaNumericFilter | Filter by non-alphanumeric ratio | No |
| UrlsFilter | Filter by URL content ratio | No |
| RepeatedLinesFilter | Filter by repeated lines | No |
...

All text filters are CPU-only and work on DocumentBatch type.
```

### Example 2: Parameter lookup
```
User: What parameters does FineWebEduClassifier take?

Agent: [Looks up in schema]

## FineWebEduClassifier

Scores educational content quality on a 0-5 scale.

**Parameters:**
| Parameter | Type | Default |
|-----------|------|---------|
| text_field | string | "text" |
| model_inference_batch_size | integer | 256 |
| max_chars | integer | None |
| label_field | string | "fineweb-edu-score-label" |
| float_score_field | string | "fineweb-edu-score-float" |
| int_score_field | string | "fineweb-edu-score-int" |

**Resources:**
- GPU required: Yes
- GPU memory: ~8 GB
- Credentials: None
```

### Example 3: Composition query
```
User: What can I run after QualityClassifier?

Agent: [Checks schema composition rules]

QualityClassifier outputs DocumentBatch, which can be consumed by:

**Filters (CPU):**
- WordCountFilter, UrlsFilter, RepeatedLinesFilter...

**Classifiers (GPU):**
- FineWebEduClassifier, DomainClassifier, AegisClassifier...

**Deduplication (GPU):**
- FuzzyDeduplicationWorkflow

**Writers:**
- ParquetWriter, JsonlWriter
```

## Utility Commands

### List all GPU operations
```bash
python -c "
import json
schema = json.load(open('skills/shared/schemas/agent-tool-schema.json'))
for name, op in schema['operations'].items():
    if op['resources'].get('requires_gpu'):
        mem = op['resources'].get('gpu_memory_gb', '?')
        print(f'{name}: {mem} GB')
"
```

### Search operations
```bash
python -c "
import json, sys
schema = json.load(open('skills/shared/schemas/agent-tool-schema.json'))
query = sys.argv[1].lower()
for name, op in schema['operations'].items():
    if query in name.lower() or query in op.get('description', '').lower():
        print(f'{name}: {op.get(\"description\", \"\")}')
" "filter"
```
