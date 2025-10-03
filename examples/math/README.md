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
pip install pynvml
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
  --output /tmp/math_mock/preprocessed_parquet
```

- Parquet files include columns: `binary_content` (bytes), `url`, `mime_type`.
- Output JSONL will include `text`, `url`, and `type`.

## Run the classifier pipeline
Run the pipeline that reads JSONL, classifies with the FineMath model, and writes JSONL outputs:

```bash
python examples/math/run_quality_classifier.py \
  --input "examples/math/data/*.jsonl" \
  --output /tmp/math_mock/out
```

Outputs will be written as JSONL files under `/tmp/math_mock/out/` with columns:
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
