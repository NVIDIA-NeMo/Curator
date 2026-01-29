# Code Annotation Integration Plan for NeMo Curator

## Overview

This document describes the code annotation library integrated into NeMo Curator for processing and filtering code data.

---

## ✅ Phase 1: Core Library (COMPLETED)

### Directory Structure

```
nemo_curator/code_annotation/
├── rust/
│   ├── Cargo.toml                    # Dependencies: pyo3, hyperpolyglot, software-metrics
│   ├── src/
│   │   ├── lib.rs                    # PyO3 module entry point
│   │   ├── annotations.rs            # Core annotation functions
│   │   └── annotations/
│   │       └── vendored_loc.rs       # Comment fraction calculation
│   └── macros/                       # Proc macros (if needed)
├── external/
│   ├── hyperpolyglot/                # Language detection
│   └── software_metrics/             # Code complexity metrics
├── python/
│   └── code_annotation/
│       └── __init__.py               # Python API wrapper
├── pyproject.toml                    # Maturin build config
└── README.md
```

### Building the Library

```bash
cd /raid/vjawa/NeMo-Curator
source .venv/bin/activate
cd nemo_curator/code_annotation
maturin develop  # Dev build
# OR
maturin build --release  # Release wheel
```

### Annotation Functions Available

| Function | Input Columns | Output Columns |
|----------|---------------|----------------|
| `detect_language` | content, representative_filename | language, language_detector |
| `basic` | content | basic_num_bytes, basic_num_lines, basic_alpha_percent, etc. |
| `software_metrics` | content, language | software_metrics_cyclomatic_complexity, etc. |
| `opencoder_rs` | content, language | ors_comment_lines_frac, ors_comment_chars_frac |
| `tokenize` | content | num_tokens_{tokenizer_name} |

### Basic Usage

```python
import pandas as pd
from code_annotation import annotate

df = pd.DataFrame({
    "content": ["def hello(): pass", "fn main() {}"],
    "representative_filename": ["test.py", "main.rs"],
})

# Run annotations
result = annotate({
    "detect_language": {},
    "basic": {},
    "software_metrics": {},
    "opencoder_rs": {},
    "tokenize": {"tokenizer_name": "github_o200k_base"}
}, df)
```

---

## ✅ Phase 2: NeMo Curator Integration (COMPLETED)

### Modifiers

Location: `nemo_curator/stages/code/modifiers.py`

```python
from nemo_curator.stages.code import (
    CodeLanguageDetector,   # Detect programming language
    CodeBasicStats,         # Basic statistics
    CodeSoftwareMetrics,    # Complexity metrics
    CodeOpenCoderMetrics,   # Comment fractions
    CodeTokenizer,          # BPE tokenization
    CodeAnnotator,          # All-in-one convenience modifier
)
```

#### Modifier Details

| Modifier | Description | Output Columns |
|----------|-------------|----------------|
| `CodeLanguageDetector` | Detects programming language | language, language_detector |
| `CodeBasicStats` | Basic file statistics | basic_num_bytes, basic_num_lines, basic_alpha_percent, etc. |
| `CodeSoftwareMetrics` | Code complexity | software_metrics_cyclomatic_complexity, etc. |
| `CodeOpenCoderMetrics` | Comment fractions | ors_comment_lines_frac, ors_comment_chars_frac |
| `CodeTokenizer` | Token counts | num_tokens_{tokenizer_name} |
| `CodeAnnotator` | All of the above | All columns combined |

### Filters

Location: `nemo_curator/stages/text/filters/code.py`

```python
from nemo_curator.stages.text.filters.code import (
    CommentFractionFilter,      # Filter by comment ratio
    MaxLineLengthFilter,        # Filter by max line length
    AverageLineLengthFilter,    # Filter by avg line length
    AlphaPercentFilter,         # Filter by alphabetic %
    HexContentFilter,           # Filter by hex content %
    Base64ContentFilter,        # Filter by base64 content %
    TokenCountFilter,           # Filter by token count
    CyclomaticComplexityFilter, # Filter by complexity
)
```

#### Filter Details

| Filter | Required Column | Default Thresholds |
|--------|-----------------|-------------------|
| `CommentFractionFilter` | ors_comment_lines_frac | 0.01 - 0.80 |
| `MaxLineLengthFilter` | basic_max_line_length | ≤ 1000 |
| `AverageLineLengthFilter` | basic_average_line_length | 5 - 100 |
| `AlphaPercentFilter` | basic_alpha_percent | ≥ 0.25 |
| `HexContentFilter` | basic_hex_percent | ≤ 0.40 |
| `Base64ContentFilter` | basic_base64_percent | ≤ 0.40 |
| `TokenCountFilter` | num_tokens_* | 10 - 100000 |
| `CyclomaticComplexityFilter` | software_metrics_cyclomatic_complexity | ≤ 50 |

---

## ✅ Phase 3: Tests (COMPLETED)

### Test Location

`tests/code_annotation/test_annotate.py` - 21 tests

### Running Tests

```bash
cd /raid/vjawa/NeMo-Curator
source .venv/bin/activate

# Run all code_annotation tests
python -m pytest tests/code_annotation/test_annotate.py -v

# Run specific test class
python -m pytest tests/code_annotation/test_annotate.py::TestDetectLanguage -v

# Quick sanity test
python -c "
import pandas as pd
from nemo_curator.stages.code import CodeAnnotator

df = pd.DataFrame({
    'content': ['def hello(): pass'],
    'representative_filename': ['test.py'],
})
result = CodeAnnotator().modify_document(df)
print('Columns:', result.columns.tolist())
print('Language:', result['language'][0])
"
```

### Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestDetectLanguage | 5 | Python, Rust, Java, batch, no filename |
| TestBasicAnnotation | 4 | Stats, XML detection, alpha percent |
| TestSoftwareMetrics | 2 | Python, Rust code metrics |
| TestOpenCoderRs | 2 | Comment fractions |
| TestTokenize | 3 | Default, tiktoken, longer code |
| TestMultipleAnnotations | 1 | Chain all annotations |
| TestEdgeCases | 4 | Empty, missing column, None values, large batch |

---

## ✅ Phase 4: Examples (COMPLETED)

### Example Scripts

Location: `examples/code_annotation/`

#### Annotate Code

```bash
# With sample data
python examples/code_annotation/annotate_code.py \
    --use_sample_data \
    --output_file output.parquet

# With real data
python examples/code_annotation/annotate_code.py \
    --input_dir /path/to/code \
    --output_file annotated.parquet
```

#### Filter Code

```bash
# With sample data
python examples/code_annotation/filter_code.py \
    --use_sample_data \
    --output_file filtered.parquet

# With pre-annotated data
python examples/code_annotation/filter_code.py \
    --input_file annotated.parquet \
    --output_file filtered.parquet

# Keep filtered files with reason columns
python examples/code_annotation/filter_code.py \
    --input_file annotated.parquet \
    --output_file filtered.parquet \
    --keep_filtered
```

---

## 🔄 Future Work / Next Steps

### 1. Integration with NeMo Curator Pipeline

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.code import CodeAnnotator
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

# Example pipeline (to be implemented)
pipeline = Pipeline(
    name="code_curation",
    stages=[
        ParquetReader(input_path="..."),
        CodeAnnotator(...),
        # Add filters...
        ParquetWriter(output_path="..."),
    ]
)
```

### 2. Add More Languages

The rust-code-analysis library supports additional languages that could be enabled.

### 3. Add OpenCoder Python Quality Signals

The full OpenCoder quality signals include many more metrics that could be added as Python-based modifiers.

### 4. Add Decontamination

The Rust library has decontamination and n-gram matching functions that could be exposed.

### 5. Performance Optimization

- Build with `--release` for production
- Consider batch processing for large datasets
- Profile and optimize hot paths

---

## Quick Reference

### File Locations

| Component | Path |
|-----------|------|
| Rust source | `nemo_curator/code_annotation/rust/src/` |
| Python API | `nemo_curator/code_annotation/python/code_annotation/` |
| Modifiers | `nemo_curator/stages/code/modifiers.py` |
| Filters | `nemo_curator/stages/text/filters/code.py` |
| Tests | `tests/code_annotation/test_annotate.py` |
| Examples | `examples/code_annotation/` |

### Dependencies

```bash
# Python dependencies
uv pip install pandas pyarrow maturin ftfy sentencepiece fasttext-wheel comment_parser

# Rust dependencies (in Cargo.toml)
# pyo3, hyperpolyglot, software-metrics, tiktoken-rs, bpe-openai, smallvec
```

### Rebuild After Changes

```bash
cd /raid/vjawa/NeMo-Curator/nemo_curator/code_annotation
maturin develop  # Rebuilds and reinstalls the Python package
```

---

## Troubleshooting

### "Module not found: code_annotation"

```bash
cd /raid/vjawa/NeMo-Curator/nemo_curator/code_annotation
maturin develop
```

### Rust compilation errors

```bash
cd /raid/vjawa/NeMo-Curator/nemo_curator/code_annotation/rust
cargo build  # Check for detailed errors
```

### Test failures

```bash
python -m pytest tests/code_annotation/test_annotate.py -v --tb=short
```

### Missing Python dependencies

```bash
cd /raid/vjawa/NeMo-Curator
source .venv/bin/activate
uv pip install ftfy sentencepiece fasttext-wheel comment_parser
```

---

## Changelog

- **2025-01-07**: Initial implementation completed
  - Rust library with PyO3 bindings
  - 5 annotation functions (detect_language, basic, software_metrics, opencoder_rs, tokenize)
  - 6 document modifiers
  - 8 document filters
  - 21 unit tests
  - 2 example scripts

