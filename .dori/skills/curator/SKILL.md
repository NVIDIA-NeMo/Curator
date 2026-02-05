---
name: curator
description: Main entry point for NeMo Curator data curation. Use when the user mentions data curation, deduplication, filtering, classification, video/text/image/audio processing, pipeline building, or working with NeMo Curator. Routes user intent to appropriate modality skills.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2.0"
  type: router
---

# NeMo Curator

Main entry point for data curation using NeMo Curator. Routes user intent to specialized modality skills, analyzes data, makes recommendations, and generates customized pipeline code or YAML configs.

## How This Works

This is an **agent skill** that uses structured reasoning to help with data curation.

### Chain-of-Thought: Before Every Response

Always reason through these questions explicitly:

```
┌─────────────────────────────────────────────────────────────┐
│  1. MODALITY: What type of data? (text/video/image/audio)   │
│  2. GOAL: What's the end use? (training/search/analysis)    │
│  3. COMPUTE: What resources? (CPU-only/GPU size/cluster)    │
│  4. DATA: Do I have a sample to analyze?                    │
│  5. BUILT-IN: Does NeMo Curator already have this?          │
│  6. SIMPLEST: YAML config or Python code?                   │
└─────────────────────────────────────────────────────────────┘
```

### ReAct Pattern: Reason → Act → Observe → Repeat

Don't just generate code. Use tools to think:

```
THOUGHT: User wants to filter text. What filters exist?
ACTION:  Run discover_builtins.py --category filter
OBSERVE: WordCountFilter, UrlsFilter, RepeatedLinesFilter...

THOUGHT: User provided data. What does it actually need?
ACTION:  Run analyze_data.py --input user_data.jsonl --sample 100
OBSERVE: 28% docs under 50 words, 12% URL-heavy...

THOUGHT: Recommend WordCountFilter(min_words=50) based on data.
ACTION:  Run validate_pipeline.py to check before generating
OBSERVE: ✅ Type flow valid, GPU fits

THOUGHT: Safe to generate. Use YAML since standard pipeline.
ACTION:  Generate YAML config with data-driven parameters
```

## Few-Shot Examples: Common Patterns

### Pattern 1: Standard Text Filtering
```
Input:  "Filter my JSONL data for quality"
Reason: Standard filtering → use YAML CLI
Output: Recommend heuristic_filter_english_pipeline.yaml
```

### Pattern 2: Custom Business Logic
```
Input:  "Only keep documents mentioning our product names"
Reason: Domain-specific logic → need Python with custom filter
Output: Generate Python with custom DocumentFilter class
```

### Pattern 3: Multi-Step Pipeline
```
Input:  "Filter, then classify, then deduplicate my text"
Reason: Standard stages but multiple steps → can chain YAML or Python
Output: Either 3 YAML runs in sequence, or single Python pipeline
```

### Pattern 4: GPU Constraint
```
Input:  "Curate text but I only have CPU"
Reason: Skip GPU stages (QualityClassifier, dedup)
Output: Use heuristic_filter_english_pipeline.yaml (CPU-only)
```

### Pattern 5: Learning/Exploration
```
Input:  "I just want to learn how this works"
Reason: Education → start simple, explain each step
Output: YAML first with explanation, then show Python equivalent
```

## Two Approaches: YAML CLI vs Python

NeMo Curator supports two ways to run pipelines:

| Approach | Best For | Complexity |
|----------|----------|------------|
| **YAML CLI** | Standard pipelines, quick iteration | Low - edit YAML, run command |
| **Python** | Custom logic, complex workflows | Medium - write code |

**Prefer YAML CLI** when:
- User wants standard filtering/classification
- No custom business logic needed
- Quick experimentation

**Use Python** when:
- Custom filters or classifiers needed
- Complex conditional logic
- Integration with other systems

## Available Skills

### Text Processing

| Command | Purpose |
|---------|---------|
| `/text` | Main text curation entry point (includes full pipeline templates) |
| `/txt-filter` | Heuristic filtering (CPU) |
| `/txt-classify` | ML classification (GPU) |
| `/txt-dedup` | Fuzzy deduplication (GPU) |

### Video Processing

| Command | Purpose |
|---------|---------|
| `/video` | Main video curation entry point |
| `/vid-clip` | Scene detection & clip extraction |
| `/vid-caption` | Generate text captions |
| `/vid-embed` | Cosmos/InternVideo embeddings |

### Image Processing

| Command | Purpose |
|---------|---------|
| `/image` | Main image curation entry point |
| `/img-embed` | CLIP embeddings |
| `/img-aesthetic` | Quality scoring & filtering |
| `/img-nsfw` | Content safety filtering |

### Audio Processing

| Command | Purpose |
|---------|---------|
| `/audio` | Main audio curation entry point |
| `/aud-asr` | ASR transcription |
| `/aud-wer` | WER-based quality filtering

### Setup & Help

| Command | Purpose |
|---------|---------|
| `/setup` | Install NeMo Curator, verify environment |
| `/project` | Set up project structure, file organization |
| `/help` | Command reference and guidance |

### Schema & Validation

| Command | Purpose |
|---------|---------|
| `/schema` | Query available operations, parameters |
| `/validate` | Validate code against schema |
| `/schema-update` | Regenerate schema (developers) |

## Intent Routing

### Natural Language Understanding

| User Says | Route To |
|-----------|----------|
| "curate my text data" | `/text` |
| "filter my documents" | `/txt-filter` |
| "score document quality" | `/txt-classify` |
| "remove duplicates from text" | `/txt-dedup` |
| "process my videos" | `/video` |
| "extract video clips" | `/vid-clip` |
| "caption my videos" | `/vid-caption` |
| "embed videos for search" | `/vid-embed` |
| "curate images" | `/image` |
| "embed images with CLIP" | `/img-embed` |
| "filter ugly images" | `/img-aesthetic` |
| "remove NSFW" | `/img-nsfw` |
| "transcribe audio" | `/aud-asr` |
| "filter by transcription quality" | `/aud-wer` |
| "get started" | `/setup` |
| "set up my project" | `/project` |

### Clarifying Questions

When intent is ambiguous:

```
User: I want to clean my dataset

Agent: I can help! A few questions:

1. What type of data? (text, images, video, audio)
2. What's your end goal? (LLM training, search, analysis)
3. Do you have specific quality issues?
4. What compute do you have? (CPU only, single GPU, cluster)
```

## YAML CLI (Recommended for Standard Pipelines)

NeMo Curator has a **Hydra-based YAML CLI** for running pipelines without writing Python.

### Built-in YAML Configs

Located in `nemo_curator/config/text/`:

| Config | Purpose |
|--------|---------|
| `heuristic_filter_english_pipeline.yaml` | Full English text filtering (20+ filters) |
| `heuristic_filter_non_english_pipeline.yaml` | Non-English text filtering |
| `exact_deduplication_pipeline.yaml` | Exact duplicate removal |
| `fuzzy_deduplication_pipeline.yaml` | MinHash fuzzy dedup |
| `semantic_deduplication_pipeline.yaml` | Embedding-based dedup |
| `fasttext_filter_pipeline.yaml` | Language filtering |
| `code_filter_pipeline.yaml` | Code-specific filtering |

### Running a YAML Pipeline

```bash
# Run built-in English filtering pipeline
python -m nemo_curator.config.run \
  --config-path ./text \
  --config-name heuristic_filter_english_pipeline.yaml \
  input_path=/data/input \
  output_path=/data/output
```

With Docker:

```bash
docker run --rm --shm-size=8g \
  -v $(pwd):/data \
  nvcr.io/nvidia/nemo-curator:25.09 \
  python -m nemo_curator.config.run \
    --config-path /opt/NeMo-Curator/nemo_curator/config/text \
    --config-name heuristic_filter_english_pipeline.yaml \
    input_path=/data/input \
    output_path=/data/output
```

### Creating Custom YAML Configs

```yaml
# my_pipeline.yaml
input_path: ???  # Required - set via CLI
output_path: ???
text_field: text

stages:
  - _target_: nemo_curator.stages.text.io.reader.JsonlReader
    file_paths: ${input_path}

  - _target_: nemo_curator.stages.text.modules.score_filter.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.heuristic_filter.WordCountFilter
      min_words: 50
    text_field: ${text_field}

  - _target_: nemo_curator.stages.text.modules.score_filter.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.heuristic_filter.UrlsFilter
      max_url_to_text_ratio: 0.2
    text_field: ${text_field}

  - _target_: nemo_curator.stages.text.io.writer.JsonlWriter
    path: ${output_path}
```

Run with:

```bash
python -m nemo_curator.config.run \
  --config-path . \
  --config-name my_pipeline.yaml \
  input_path=./data output_path=./output
```

### When to Recommend YAML vs Python

| User Request | Recommendation |
|--------------|----------------|
| "filter my text data" | YAML - use built-in config |
| "I want to try different filter settings" | YAML - easy to edit |
| "custom logic based on metadata" | Python - need code |
| "integrate with my existing system" | Python - more flexible |
| "just want to learn" | YAML first, then Python |

## Agent Tool Schema

The schema is the **source of truth** for operations:

```
skills/shared/schemas/nemo_curator_schema.json
```

Before generating code, consult the schema for:
- Exact parameter names and types
- GPU memory requirements
- Type compatibility
- Credential requirements (e.g., HF_TOKEN)

## Active Tools (Agent Workflow)

These scripts help the agent **plan and validate before generating code**. Use them proactively.

### 1. Validate Pipeline (BEFORE generating code)

Check type compatibility, GPU requirements, and credentials:

```bash
python skills/shared/scripts/validate_pipeline.py \
  --stages "JsonlReader,WordCountFilter,QualityClassifier" \
  --available-gpu 16 --json
```

**When to use**: Always run before generating pipeline code to catch type mismatches and resource issues.

**Output includes**:
- Type flow validation (e.g., `FileGroupTask → DocumentBatch → DocumentBatch`)
- Total GPU memory requirements
- Required credentials (HF_TOKEN, etc.)
- Warnings about potential issues

### 2. Analyze Data (Let data drive recommendations)

Analyze user's data and get data-driven filter recommendations:

```bash
python skills/shared/scripts/analyze_data.py \
  --input /path/to/data.jsonl \
  --sample 100 --json
```

**When to use**: When user provides data, analyze it before recommending filters. Don't guess - let the data tell you what filters are needed.

**Output includes**:
- Word count distribution (min/max/avg, % short docs)
- URL density analysis
- Language distribution
- Repetition scores
- **Recommended pipeline** with specific parameters based on the data

### 3. Test Pipeline (Verify before full run)

Test a pipeline on sample data to verify it works:

```bash
python skills/shared/scripts/test_pipeline.py \
  --stages "WordCountFilter,QualityClassifier" \
  --input /path/to/data.jsonl \
  --sample 20 --json
```

**When to use**: After generating code, run on a small sample to verify:
- Stages execute without errors
- New columns are created as expected
- Retention rate is reasonable

**Requires**: NeMo Curator installed. Use `--dry-run` to check stage availability without running.

### 4. Diagnose Errors (Smart debugging)

When user reports an error, diagnose and suggest fixes:

```bash
python skills/shared/scripts/diagnose_error.py \
  --error "CUDA out of memory" \
  --context '{"pipeline": "QualityClassifier,AegisClassifier"}' --json
```

**When to use**: When user encounters errors running pipelines. Provides actionable fixes.

**Output includes**:
- Root cause diagnosis
- Category (GPU, data, auth, etc.)
- Specific fix suggestions with examples
- GPU memory analysis if context provided

## Recommended Agent Workflow (ReAct Pattern)

### Phase 1: Understand (Gather Context)

```
THOUGHT: What is the user trying to accomplish?
ACTION:  Ask clarifying questions if needed
         - Modality? (text/video/image/audio)
         - End goal? (LLM training, RAG, search)
         - Data format and size?
         - Compute available? (CPU/GPU memory)
```

### Phase 2: Discover (Check Built-ins First)

```
THOUGHT: Does NeMo Curator already provide what's needed?
ACTION:  RUN discover_builtins.py --query "<user's need>"
OBSERVE: List of matching built-in operations
VERIFY:  ⚠️ NEVER write custom code if a built-in exists
```

### Phase 3: Analyze (Let Data Drive Decisions)

```
THOUGHT: What does the data actually need?
ACTION:  RUN analyze_data.py --input <path> --sample 100
OBSERVE: Word count distribution, URL density, language mix
VERIFY:  Recommendations should match data analysis, not defaults
```

### Phase 4: Validate (Before Generating)

```
THOUGHT: Will this pipeline work?
ACTION:  RUN validate_pipeline.py --stages "Stage1,Stage2,..."
OBSERVE: Type flow, GPU requirements, credentials needed
VERIFY:  ✅ All checks pass before generating code
```

### Phase 5: Generate (YAML or Python)

```
THOUGHT: What's the simplest solution?
ACTION:  Generate YAML config (preferred) or Python code
VERIFY:  Parameters match data analysis results
```

### Phase 6: Test (Verify Before Full Run)

```
THOUGHT: Does it work on sample data?
ACTION:  RUN test_pipeline.py --sample 20
OBSERVE: Retention rate, runtime, output columns
VERIFY:  Results are reasonable before full run
```

### Self-Consistency Checklist

Before finalizing any recommendation, verify:

- [ ] **Data-driven?** Parameters match analyze_data.py output
- [ ] **Built-in?** Used discover_builtins.py, not custom code
- [ ] **Validated?** validate_pipeline.py passed
- [ ] **GPU fits?** Required memory ≤ available memory
- [ ] **Simplest?** YAML if possible, Python only if needed
- [ ] **Tested?** Suggested test_pipeline.py on sample

## Handling Edge Cases (Meta Prompting)

When the request doesn't fit standard patterns:

### Ambiguous Modality
```
User: "I want to process my data"

THOUGHT: Modality unclear. Need to ask.
ACTION:  "What type of data? Text documents, videos, images, or audio?"
```

### Multi-Modal Request
```
User: "I have videos and I want transcripts"

THOUGHT: This spans video → audio → text. Break it down.
ACTION:  
  1. Video: Extract audio track (outside NeMo Curator)
  2. Audio: /aud-asr for transcription
  3. Text: /txt-filter for quality filtering
```

### No Built-in Exists
```
User: "Filter documents by my custom scoring function"

THOUGHT: discover_builtins.py found nothing matching.
ACTION:  
  1. Confirm custom code is truly needed
  2. Pre-process data to JSONL outside pipeline if possible
  3. If in-pipeline custom logic needed, provide self-contained class
  4. Warn about Ray serialization constraints
```

### Unclear Quality Goals
```
User: "Make my data better"

THOUGHT: "Better" is vague. Need specifics.
ACTION:  Ask:
  - Better for what purpose? (LLM training, RAG, search)
  - What quality issues have you noticed?
  - What's your retention target? (keep 50%? 20%? 80%?)
```

### Resource Constraints
```
User: "I have a huge dataset but only CPU"

THOUGHT: GPU stages won't work. Offer alternatives.
ACTION:  
  - Use YAML CLI with heuristic_filter_english_pipeline.yaml (CPU-only)
  - Skip QualityClassifier (GPU required)
  - Suggest staged approach: filter on CPU, classify on GPU later
```

## ⭐ Discover Built-ins First (CRITICAL)

**ALWAYS check for built-in operations before writing custom code!**

```bash
# Find operations related to user's need
python skills/shared/scripts/discover_builtins.py --query "punctuation"
python skills/shared/scripts/discover_builtins.py --query "quote"
python skills/shared/scripts/discover_builtins.py --query "unicode"

# List all available operations by category
python skills/shared/scripts/discover_builtins.py --category filter
python skills/shared/scripts/discover_builtins.py --category modifier
python skills/shared/scripts/discover_builtins.py --category reader
```

### Common Built-ins You Should Know

| User Wants | Built-in Solution |
|------------|-------------------|
| "normalize quotes" | `UnicodeReformatter(uncurl_quotes=True)` |
| "fix encoding" | `UnicodeReformatter(fix_encoding=True)` |
| "filter short docs" | `WordCountFilter(min_words=50)` |
| "remove repetition" | `RepeatedLinesFilter()` or `RepeatedParagraphsFilter()` |
| "filter bad punctuation" | `PunctuationFilter()` |
| "remove URLs" | `UrlRemover()` or `UrlsFilter()` |
| "language filter" | `FastTextLangId(languages=['en'])` |
| "quality scoring" | `QualityClassifier()` |
| "read JSONL" | `JsonlReader(path)` |
| "read Parquet" | `ParquetReader(path)` |

### When Custom Code IS Needed

Only write custom classes when:
1. **Data source is unique** - Custom download/parse logic (e.g., proprietary format)
2. **Business logic is specific** - Domain-specific filtering rules
3. **No built-in exists** - After searching with discover_builtins.py

**Even then, prefer:**
- Pre-processing data to standard format (JSONL) OUTSIDE the pipeline
- Using built-in readers + built-in filters

## Utility Scripts

### Search Available Stages
```bash
python skills/stages/scripts/search_stages.py --query "filter" --modality text
```

### List Classifiers
```bash
python skills/classify/scripts/list_classifiers.py --json
```

### List Filters
```bash
python skills/filter/scripts/list_filters.py --json
```

### Generate Full Curator Map
```bash
python skills/shared/scripts/generate_curator_map.py --output map.json
```

## Core Principles

### 1. Use Tools to Think, Not Just Generate

**The agent doesn't just generate code** - it uses tools to plan:

```
❌ User asks → Agent generates code → User runs → Errors
✅ User asks → Agent runs validate_pipeline.py → Agent generates validated code
```

### 2. Analyze Before Recommending

**Don't guess. Let the data tell you:**

```bash
# Before recommending filters, analyze the data
python analyze_data.py --input user_data.jsonl --sample 100
```

Use the output to make data-driven recommendations.

### 3. Validate Before Generating

**Always run validate_pipeline.py before generating code:**

```bash
python validate_pipeline.py --stages "Stage1,Stage2,Stage3"
```

This catches type mismatches, GPU issues, and missing credentials.

### 4. Test Before Full Run

**Suggest testing on sample data:**

```bash
python test_pipeline.py --stages "Stage1,Stage2" --sample 20
```

### 5. Diagnose Errors Smartly

**When errors occur, use diagnose_error.py:**

```bash
python diagnose_error.py --error "User's error message"
```

Provide actionable fixes, not just "something went wrong."

### 6. Use Real NeMo Curator APIs

```python
# Real imports from schema
from nemo_curator.stages.text.filters.heuristic_filter import WordCountFilter
from nemo_curator.pipeline import Pipeline
```

### 7. Be Environment-Aware

Consider CPU vs GPU, Docker vs native, single machine vs cluster.

## Common Pitfalls (AVOID THESE)

### Pitfall 1: Writing Custom Code When Built-ins Exist

**Bad**: Creating `QuotationUnifier` class to normalize quotes
**Good**: Use `UnicodeReformatter(uncurl_quotes=True)`

**Always run discover_builtins.py first!**

### Pitfall 2: External Module Imports with Ray

Custom classes defined in external files fail in Ray workers:

```python
# ❌ BAD - Ray workers can't find this
sys.path.insert(0, "/path/to/my/code")
from my_stages import MyFilter  # Fails in Ray workers!

# ✅ GOOD - Self-contained (if custom code is truly needed)
class MyFilter(DocumentFilter):
    def score_document(self, text: str) -> float:
        return len(text)
```

**Best**: Avoid custom classes entirely by using built-ins.

### Pitfall 3: Complex Data Ingestion in Pipeline

**Bad**: Custom download/parse stages in the pipeline (causes Ray serialization issues)

**Good**: Pre-process data to JSONL outside the pipeline, then use `JsonlReader`:

```python
# Step 1: Pre-download (one-time, outside pipeline)
from datasets import load_dataset
ds = load_dataset("my_dataset")
ds.to_json("data/input.jsonl")

# Step 2: Simple pipeline with built-ins only
pipeline = Pipeline(stages=[
    JsonlReader("data/input.jsonl"),
    ScoreFilter(filter_obj=WordCountFilter(min_words=20)),
    JsonlWriter("data/output"),
])
```

### Pitfall 4: Missing Docker Flags

Always use `--shm-size` for Ray:

```bash
# ❌ BAD
docker run --rm nvcr.io/nvidia/nemo-curator:25.09 python script.py

# ✅ GOOD
docker run --rm --shm-size=8g nvcr.io/nvidia/nemo-curator:25.09 python script.py
```

### Pitfall 5: Using :latest Tag

The `:latest` tag doesn't always exist on NGC:

```bash
# ❌ BAD
docker pull nvcr.io/nvidia/nemo-curator:latest

# ✅ GOOD
docker pull nvcr.io/nvidia/nemo-curator:25.09
```

## Skill Directory Structure

```
skills/
├── curator/SKILL.md       # This file - main entry point
│
├── text/SKILL.md          # Text curation (includes full pipeline templates)
├── txt-filter/SKILL.md    # Heuristic filtering
├── txt-classify/SKILL.md  # ML classification  
├── txt-dedup/SKILL.md     # Fuzzy deduplication
│
├── video/SKILL.md         # Video curation
├── vid-clip/SKILL.md      # Clip extraction
├── vid-caption/SKILL.md   # Captioning
├── vid-embed/SKILL.md     # Embeddings
│
├── image/SKILL.md         # Image curation
├── img-embed/SKILL.md     # CLIP embeddings
├── img-aesthetic/SKILL.md # Aesthetic filtering
├── img-nsfw/SKILL.md      # NSFW filtering
│
├── audio/SKILL.md         # Audio curation
├── aud-asr/SKILL.md       # ASR transcription
├── aud-wer/SKILL.md       # WER filtering
│
├── setup/SKILL.md         # Environment setup
├── help/SKILL.md          # Command reference
│
└── shared/
    ├── schemas/           # Agent tool schemas
    └── scripts/           # Agent helper scripts
```

## Example Interaction (With Visible Reasoning)

```
User: I want to prepare web data for LLM training

─────────────────────────────────────────────────────────────
THOUGHT: Modality is text. Goal is LLM training. Need to know:
         - Data size and format
         - GPU availability  
         - Can I analyze a sample?
─────────────────────────────────────────────────────────────

Agent: I can help curate web data for LLM training. A few questions:
- What format is your data? (JSONL, Parquet)
- How large is the dataset?
- What GPU do you have? (or CPU-only?)
- Can I analyze a sample to make data-driven recommendations?

User: 50GB JSONL at /data/crawl.jsonl. I have an A100 (40GB).

─────────────────────────────────────────────────────────────
THOUGHT: Have data path and GPU info. Before recommending anything,
         analyze the data to understand what it actually needs.
ACTION:  Run analyze_data.py
─────────────────────────────────────────────────────────────

Agent: Let me analyze your data first.

[Runs: python analyze_data.py --input /data/crawl.jsonl --sample 100 --json]

─────────────────────────────────────────────────────────────
OBSERVE: 28% docs under 50 words → need WordCountFilter
         12% URL-heavy → need UrlsFilter  
         18% non-English → need language filter
THOUGHT: These are all built-in filters. No custom code needed.
         For LLM training, also add QualityClassifier.
ACTION:  Validate pipeline before generating.
─────────────────────────────────────────────────────────────

📊 Data Analysis Results:
- Sample: 100 docs, Total: ~5M docs
- Word count: avg 145, **28% under 50 words**
- URL ratio: **12% URL-heavy**
- Languages: en (82%), **18% non-English**

Based on your data, I recommend:
1. `WordCountFilter(min_words=50)` - addresses 28% short docs
2. `UrlsFilter(max_url_to_text_ratio=0.2)` - addresses 12% URL-heavy
3. `FastTextLangId(languages=['en'])` - addresses 18% non-English
4. `QualityClassifier(filter_by=['High', 'Medium'])` - ML quality gate

Let me validate this pipeline fits your A100:

[Runs: python validate_pipeline.py --stages "WordCountFilter,UrlsFilter,FastTextLangId,QualityClassifier" --available-gpu 40 --json]

─────────────────────────────────────────────────────────────
OBSERVE: ✅ Type flow valid, GPU 4GB needed, fits in 40GB
VERIFY:  ☑ Data-driven (matches analysis)
         ☑ Built-in stages (no custom code)
         ☑ Validated (type flow OK)
         ☑ GPU fits (4GB < 40GB)
         ☑ Simplest solution (can use YAML)
─────────────────────────────────────────────────────────────

✅ Pipeline Valid:
- Type flow: DocumentBatch → DocumentBatch (all stages)
- GPU: 4GB (QualityClassifier only) - fits in your 40GB A100
- No credentials required

Estimated yield: ~35% of original data.

Since this uses standard filters, I can generate either:
1. **YAML config** (simpler, recommended)
2. **Python script** (if you need customization later)

Which do you prefer?

User: YAML please

Agent: Here's your pipeline config:

[Generates: my_pipeline.yaml with data-driven parameters]

Before running on 50GB, let's test on a sample:

[Runs: python test_pipeline.py --input /data/crawl.jsonl --sample 20 --json]

─────────────────────────────────────────────────────────────
OBSERVE: 20 → 8 docs (40% retention), runtime 4.2s
VERIFY:  ☑ Tested on sample - works correctly
─────────────────────────────────────────────────────────────

✅ Test passed:
- Input: 20 docs → Output: 8 docs (40% retention)
- Runtime: 4.2s
- New columns: quality_pred

The pipeline works. Run on full data with:

```bash
docker run --rm --gpus all --shm-size=8g \
  -v $(pwd):/data \
  nvcr.io/nvidia/nemo-curator:25.09 \
  python -m nemo_curator.config.run \
    --config-path /data \
    --config-name my_pipeline.yaml \
    input_path=/data/crawl.jsonl \
    output_path=/data/curated
```

Expected runtime: ~2-3 hours for 50GB on A100.
```
