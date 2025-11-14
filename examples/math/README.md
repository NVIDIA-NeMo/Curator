# Math Quality Classifier

This example demonstrates running a model-based math classifier (FineMath)

## Install
Use uv to create the project environment and install Curator with the text extra:

```bash
uv sync --extra text
source .venv/bin/activate
```

- GPU detection: if `nvidia-smi` shows GPUs but the example logs "No gpus found", install `pynvml` so the backend can discover GPUs:

```bash
uv pip install pynvml
```

- For LLM cleanup pipeline, install vLLM:

```bash
uv pip install vllm
```

## Prerequisites
- GPU(s) with CUDA for the HF model
- Python environment with `nemo-curator` installed (uv sync above)
- Lynx system dependency for HTML rendering to text:
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y lynx`
  - RHEL/Fedora: `sudo dnf install -y lynx` (or `sudo yum install -y lynx`)
  - Conda: `conda install -c conda-forge lynx`

## Text preprocessing (decode → type-detect → extract)

```bash
python examples/math/run_text_preprocess.py \
  --input "examples/math/data/*.parquet" \
  --output /tmp/math_mock/preprocessed_jsonl

# Optional: Add --report-stats to see extraction statistics
python examples/math/run_text_preprocess.py \
  --input "examples/math/data/*.parquet" \
  --output /tmp/math_mock/preprocessed_jsonl \
  --report-stats
```

- Parquet files include columns: `binary_content` (bytes), `url`, `mime_type`.
- Output JSONL will include `text`, `url`, and `type`.

## Run the classifier pipeline
Run the pipeline that reads JSONL, classifies with the FineMath model, and writes JSONL outputs:

```bash
python examples/math/run_quality_classifier.py \
  --input "/tmp/math_mock/preprocessed_jsonl/*.jsonl" \
  --output /tmp/math_mock/classified_output
```

Outputs will be written as JSONL files under `/tmp/math_mock/classified_output/` with columns:
- `finemath_scores`: float scores (0..5)
- `finemath_int_scores`: integer scores (0..5)

Output
```
{"id":0,"text":"The derivative of x^2 is 2x.","finemath_scores":1.6865234375,"finemath_int_scores":2}
{"id":1,"text":"This is plain English without math.","finemath_scores":0.9130859375,"finemath_int_scores":1}
{"id":2,"text":"Let $f(x)=x^2$. Then $f'(x)=2x.","finemath_scores":2.291015625,"finemath_int_scores":2}
{"id":3,"text":"We have $$\\int_0^1 x^2 dx = 1\/3.$$.","finemath_scores":1.9150390625,"finemath_int_scores":2}
{"id":4,"text":"Using \\(a^2+b^2=c^2\\) we derive the relation.","finemath_scores":1.93359375,"finemath_int_scores":2}
{"id":5,"text":"Consider the set A \\subseteq B and A \\in \\mathbb{R}^n.","finemath_scores":1.5458984375,"finemath_int_scores":2}
```

## Deduplication Pipeline

Run fuzzy deduplication on Parquet or JSONL files:

```bash
python examples/math/run_deduplication.py \
  --input DATA_DIR \
  --cache_dir CACHE_DIR \
  --duplicate_ids_dir DUPLICATE_IDS_DIR \
  --output_path OUTPUT_PATH \
  --input_filetype jsonl
```

Deduplication takes place in two stages.
1. In the first stage, the duplicate ids are identified and saved to disk.
2. In the second stage, the duplicate ids are removed from the dataset

- `--input`: Input directory path for Parquet/JSONL files
- `--cache_dir`: Cache directory for deduplication intermediates (must be empty between runs)
- `--duplicate_ids_dir`: Output directory for duplicate IDs and id generator mapping
- `--output_path`: Output directory for deduplicated data
- `--input_filetype`: Input file type (`jsonl` or `parquet`)

## LLM Cleanup Pipeline

The LLM cleanup pipeline always runs LLM cleanup. Optionally, you can chunk long texts before cleaning using the `--chunk_data` flag.

### Option 1: Clean with chunking (for long texts)

For long texts that exceed model context limits, chunk first then clean each chunk:

```bash
python examples/math/run_cleanup_webpages_with_llm.py \
  --input DATA_DIR \
  --output OUTPUT_DIR \
  --model microsoft/phi-4 \
  --prompt HTML_TO_TEXT_PROMPT \
  --chunk_data \
  --chunk_length 5000 \
  --input_filetype parquet
```

This will chunk the data and clean each chunk, creating output in `OUTPUT_CHUNK_DIR/cleanup_*/` with:
- `cleaned_text`: LLM-processed text (or `label` if `--classification` is used)
- `chunk_id`: Sequential chunk identifier (if chunking was used)
- `n_tokens`: Number of tokens in the chunk (if chunking was used)
- All original metadata fields preserved

### Option 2: Clean without chunking (for short texts)

For texts that fit within model context limits, clean directly:

```bash
python examples/math/run_cleanup_webpages_with_llm.py \
  --input DATA_DIR \
  --output OUTPUT_DIR \
  --model microsoft/phi-4 \
  --prompt HTML_TO_TEXT_PROMPT \
  --input_filetype parquet
```

Outputs will be written as JSONL files with:
- `cleaned_text`: LLM-processed text (or `label` if `--classification` is used)
- All original metadata fields preserved

Additional options:
- `--classification`: Output classification labels instead of cleaned text
- `--max_model_len`: Maximum model context length (auto-detected if not specified)
- `--filter_by_n_tokens`: Filter chunks by token count (requires `--chunk_data`)
- `--temperature`, `--top_p`, `--top_k`, `--min_p`: Sampling parameters
