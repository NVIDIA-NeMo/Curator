---
name: validate
description: Validate generated NeMo Curator code against the Agent Tool Schema. Use when the user wants to check if their pipeline code is correct, uses valid operations, or has correct parameters.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  type: validation
---

# Code Validation Skill

Validate user's NeMo Curator code against the Agent Tool Schema to catch errors before runtime.

## When This Skill Applies

- User asks "is this code correct?"
- User gets import errors or parameter errors
- User wants to verify generated pipeline
- Before user runs expensive GPU operations

## What This Skill Validates

1. **Import paths** - Do the classes exist?
2. **Parameters** - Are parameter names and types correct?
3. **Type compatibility** - Can these stages connect?
4. **Resource requirements** - Does user have needed GPU memory?

## Skill Workflow

### Step 1: Parse User's Code

Extract:
- Import statements
- Class instantiations with parameters
- Pipeline composition order

### Step 2: Load Schema

```python
import json
schema = json.load(open("skills/shared/schemas/agent-tool-schema.json"))
```

### Step 3: Validate Each Component

#### Check Imports

```python
# User code has: from nemo_curator.stages.text.filters import WordCountFilter
# Check against schema:
op = schema["operations"].get("WordCountFilter")
if op:
    expected_import = op["import"]
    # Verify it matches
```

#### Check Parameters

```python
# User code has: WordCountFilter(min_words=50, max_word=100)
# Note: "max_word" should be "max_words"

op_params = schema["operations"]["WordCountFilter"]["parameters"]
for user_param in user_params:
    if user_param not in op_params:
        print(f"Unknown parameter: {user_param}")
        print(f"Did you mean: {suggest_similar(user_param, op_params)}")
```

#### Check Types

```python
# User code has: stage1 -> stage2
# Check: stage1.output_type == stage2.input_type

stage1_output = schema["operations"]["Stage1"]["output_type"]
stage2_input = schema["operations"]["Stage2"]["input_type"]

if stage1_output != stage2_input:
    print(f"Type mismatch: {stage1_output} -> {stage2_input}")
```

#### Check Resources

```python
# Sum GPU requirements
total_gpu = 0
for stage in pipeline_stages:
    op = schema["operations"][stage]
    if op["resources"]["requires_gpu"]:
        total_gpu += op["resources"].get("gpu_memory_gb", 8)

print(f"Estimated GPU memory: {total_gpu} GB")
```

### Step 4: Report Issues

Format clearly:

```
## Validation Results

✅ Imports: All valid
⚠️ Parameters: 1 issue
   - WordCountFilter: Unknown parameter 'max_word'
     Did you mean 'max_words'?
✅ Type flow: Compatible
📊 Resources: 12 GB GPU memory required

### Corrected Code

```python
# Fixed parameter name
WordCountFilter(min_words=50, max_words=100)  # was: max_word
```
```

## Example Interactions

### Example 1: Parameter typo
```
User: Is this code correct?
```python
from nemo_curator.stages.text.filters import WordCountFilter
filter = WordCountFilter(min_word=50)
```

Agent: [Validates against schema]

## Validation Results

❌ **Parameter Error**

`WordCountFilter` has no parameter `min_word`.

Did you mean `min_words`?

**Available parameters:**
- `min_words` (integer, default: 50)
- `max_words` (integer, default: 100000)

**Corrected:**
```python
filter = WordCountFilter(min_words=50)
```
```

### Example 2: Type mismatch
```
User: Can I do this?
```python
video_reader = VideoReader(...)
word_filter = WordCountFilter(...)
pipeline = Pipeline(stages=[video_reader, word_filter])
```

Agent: [Checks type flow]

## Validation Results

❌ **Type Mismatch**

- `VideoReader` outputs `VideoTask`
- `WordCountFilter` expects `DocumentBatch`

These stages are incompatible. `WordCountFilter` is for text data, not video.

**For video filtering**, use:
- `MotionFilterStage` - filter by motion
- `ClipAestheticFilterStage` - filter by aesthetic quality
```

### Example 3: Resource check
```
User: Will this fit on my 24GB GPU?
```python
classifier = QualityClassifier(...)
edu_classifier = FineWebEduClassifier(...)
aegis = AegisClassifier(...)
```

Agent: [Sums GPU requirements from schema]

## Resource Estimate

| Stage | GPU Memory |
|-------|------------|
| QualityClassifier | 4 GB |
| FineWebEduClassifier | 8 GB |
| AegisClassifier | 16 GB |
| **Total** | **28 GB** |

⚠️ **Warning**: Total exceeds 24GB.

**Options:**
1. Run stages sequentially (not in parallel)
2. Reduce `model_inference_batch_size` to lower memory
3. Drop one classifier
```

## Utility: Validate Script

Run the validation script directly:

```bash
python skills/shared/scripts/validate_agent_tool_schema.py \
    skills/shared/schemas/agent-tool-schema.json \
    --verbose
```

For CI integration:
```bash
python skills/shared/scripts/validate_agent_tool_schema.py \
    skills/shared/schemas/agent-tool-schema.json \
    --strict --json
```
