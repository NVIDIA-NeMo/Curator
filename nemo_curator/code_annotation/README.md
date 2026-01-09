# Code Annotation Library

Minimal Rust-based code annotation library for NeMo Curator.

## Functions

- `detect_language`: Programming language detection via hyperpolyglot
- `basic`: Basic statistics (byte count, UTF-8 validation, line stats, patterns)
- `tokenize`: BPE tokenization
- `software_metrics`: Code complexity metrics via rust-code-analysis
- `opencoder_rs`: Comment line/character fractions

## Usage

```python
import pandas as pd
from code_annotation import annotate

df = pd.DataFrame({
    "content": ["def hello(): pass", "fn main() {}"],
    "representative_filename": ["test.py", "main.rs"],
})

# Run language detection
result = annotate({"detect_language": {}}, df)
print(result["language"])
```

## Building

```bash
cd nemo_curator/code_annotation
maturin develop
```
