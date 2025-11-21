# Math Data Curation Pipeline

This example demonstrates a complete pipeline for curating mathematical content, including text preprocessing, quality classification, deduplication, and LLM-based cleanup.

## Install
Use uv to create the project environment and install Curator with the math extra:

```bash
uv sync --extra math_cuda12
source .venv/bin/activate
```

**Note:** GPU detection - if `nvidia-smi` shows GPUs but examples log "No gpus found", `pynvml` may need to be reinstalled:
```bash
uv pip install --force-reinstall pynvml
```

## Prerequisites
- GPU(s) with CUDA for the HF model and vLLM
- Python environment with `nemo-curator[math_cuda12]` installed (uv sync above)
- Lynx system dependency for HTML rendering to text:
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y lynx`
  - RHEL/Fedora: `sudo dnf install -y lynx` (or `sudo yum install -y lynx`)
  - Conda: `conda install -c conda-forge lynx`

## Complete Pipeline Flow

This example provides a complete data curation pipeline where outputs from one step feed into the next:

```
1. Text Preprocessing → 2. Quality Classification → 3. Deduplication → 4. LLM Cleanup
```

### Working Directory Setup

```bash
# Create working directories
export MATH_DATA_DIR=/tmp/math_pipeline
mkdir -p $MATH_DATA_DIR/{preprocessed,classified,dedup_cache,dedup_ids,deduplicated,cleaned}
```

## Step 1: Text Preprocessing (decode → type-detect → extract)

Extract and preprocess text from raw web data:

```bash
python examples/math/run_text_preprocess.py \
  --input "examples/math/data/*.parquet" \
  --output $MATH_DATA_DIR/preprocessed

# Optional: Add --report-stats to see extraction statistics
python examples/math/run_text_preprocess.py \
  --input "examples/math/data/*.parquet" \
  --output $MATH_DATA_DIR/preprocessed \
  --report-stats
```

**Input**: Parquet files with columns: `binary_content` (bytes), `url`, `mime_type`

**Output**: JSONL files with columns: `text`, `url`, `type`

## Step 2: Quality Classification

Classify mathematical content quality using the FineMath model:

```bash
python examples/math/run_quality_classifier.py \
  --input "$MATH_DATA_DIR/preprocessed/*.jsonl" \
  --output $MATH_DATA_DIR/classified
```

**Input**: JSONL files from Step 1

**Output**: JSONL files with additional columns:
- `finemath_scores`: float scores (0..5)
- `finemath_int_scores`: integer scores (0..5)

**Example Output**:
```json
{"id":0,"text":"The derivative of x^2 is 2x.","finemath_scores":1.6865234375,"finemath_int_scores":2}
{"id":1,"text":"This is plain English without math.","finemath_scores":0.9130859375,"finemath_int_scores":1}
{"id":2,"text":"Let $f(x)=x^2$. Then $f'(x)=2x.","finemath_scores":2.291015625,"finemath_int_scores":2}
{"id":3,"text":"We have $$\\int_0^1 x^2 dx = 1/3.$$.","finemath_scores":1.9150390625,"finemath_int_scores":2}
```

## Step 3: Deduplication

Remove duplicate content using fuzzy deduplication:

```bash
python examples/math/run_deduplication.py \
  --input $MATH_DATA_DIR/classified \
  --cache_dir $MATH_DATA_DIR/dedup_cache \
  --duplicate_ids_dir $MATH_DATA_DIR/dedup_ids \
  --output $MATH_DATA_DIR/deduplicated \
  --input_filetype jsonl
```

**Input**: JSONL files from Step 2

**Output**: Deduplicated JSONL files

**Process**: Deduplication takes place in two stages:
1. First stage: Duplicate IDs are identified and saved to `duplicate_ids_dir`
2. Second stage: Duplicates are removed from the dataset

**Note**: The `cache_dir` must be empty between runs.

## Step 4: LLM Cleanup

Clean and refine text using a large language model (optional chunking for long texts):

### Option 1: Clean with chunking (for long texts)

For long texts that exceed model context limits, chunk first then clean each chunk:

```bash
python examples/math/run_cleanup_webpages_with_llm.py \
  --input $MATH_DATA_DIR/deduplicated \
  --output $MATH_DATA_DIR/cleaned \
  --model microsoft/phi-4 \
  --prompt HTML_TO_TEXT_PROMPT \
  --chunk_data \
  --chunk_length 5000 \
  --input_filetype parquet
```

**Input**: JSONL files from Step 3

**Output**: JSONL files with additional columns:
- `cleaned_text`: LLM-processed text (or `label` if `--classification` is used)
- `chunk_id`: Sequential chunk identifier (if chunking was used)
- `n_tokens`: Number of tokens in the chunk (if chunking was used)
- All original metadata fields preserved

### Option 2: Clean without chunking (for short texts)

For texts that fit within model context limits, clean directly:

```bash
python examples/math/run_cleanup_webpages_with_llm.py \
  --input $MATH_DATA_DIR/deduplicated \
  --output $MATH_DATA_DIR/cleaned \
  --model microsoft/phi-4 \
  --prompt HTML_TO_TEXT_PROMPT \
  --input_filetype parquet
```

**Additional options**:
- `--classification`: Output classification labels instead of cleaned text
- `--max_model_len`: Maximum model context length (auto-detected if not specified)
- `--filter_by_n_tokens`: Filter chunks by token count (requires `--chunk_data`)
- `--temperature`, `--top_p`, `--top_k`, `--min_p`: Sampling parameters

## Alternative Prompts and Use Cases

The LLM cleanup step supports various specialized prompts for different mathematical content processing needs:

### Content Cleaning Prompts

**`HTML_TO_TEXT_PROMPT`** (default): Extract main content, preserve math, standardize equations to LaTeX `$...$`, remove boilerplate

**`HTML_TO_TEXT_PROMPT_CODE`**: For pages mixing math and significant code (e.g., computational math tutorials)

```bash
python examples/math/run_cleanup_webpages_with_llm.py \
  --input $MATH_DATA_DIR/deduplicated \
  --output $MATH_DATA_DIR/cleaned_code \
  --model microsoft/phi-4 \
  --prompt HTML_TO_TEXT_PROMPT_CODE \
  --chunk_data \
  --chunk_length 5000 \
  --input_filetype parquet
```
---

## Running Individual Steps

You can also run individual steps independently with custom input/output directories. Just ensure the input format matches what each script expects.
