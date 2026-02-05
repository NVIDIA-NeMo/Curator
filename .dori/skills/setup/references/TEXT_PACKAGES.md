# Text Curation Packages

Reference for text-specific NeMo Curator dependencies.

## Package Extras

| Extra | Use Case |
|-------|----------|
| `text_cpu` | Text filtering, basic processing (no GPU) |
| `text_cuda12` | Full text curation with GPU deduplication |

## Dependencies

### text_cpu

| Package | Purpose |
|---------|---------|
| `beautifulsoup4` | HTML parsing |
| `justext` | Boilerplate removal |
| `lxml` | XML/HTML processing |
| `pycld2` | Language detection |
| `resiliparse` | Fast HTML parsing |
| `trafilatura` | Web content extraction |
| `warcio` | WARC file processing |
| `fasttext` | Language ID, quality filtering |
| `sentencepiece` | Tokenization |
| `peft` | Aegis classifier support |
| `ftfy` | Unicode fixing |
| `sentence-transformers` | Embeddings |

### text_cuda12 (additional)

| Package | Purpose |
|---------|---------|
| `cudf-cu12` | GPU DataFrames |
| `cuml-cu12` | GPU ML algorithms |
| `pylibcugraph-cu12` | GPU graph algorithms |
| `raft-dask-cu12` | Distributed GPU |
| `vllm` | Fast LLM inference |

## Capabilities

### Filters (25+)

- **Length**: WordCount, TokenCount, MeanWordLength
- **Composition**: NonAlphaNumeric, Symbols, Numbers, Punctuation
- **Content**: URLs, Bullets, Ellipsis, PornographicURLs
- **Repetition**: RepeatedLines, RepeatedParagraphs, NGrams
- **Language**: CommonEnglishWords, FastTextLangId

### Classifiers (10+)

- **Quality**: QualityClassifier, FineWebEduClassifier
- **Domain**: DomainClassifier, MultilingualDomainClassifier
- **Safety**: AegisClassifier, InstructionDataGuardClassifier

### Deduplication

- **Exact**: Hash-based exact matching
- **Fuzzy**: MinHash + LSH (80% similarity)
- **Semantic**: Embedding-based similarity

## GPU Requirements

| Stage | GPU Memory |
|-------|------------|
| Fuzzy deduplication | 16+ GB |
| Quality classifier | 8+ GB |
| vLLM embeddings | 16+ GB |

## Common Issues

### FastText Installation

```bash
# If build fails
sudo apt-get install build-essential
uv pip install fasttext==0.9.3
```

### cuDF/RAPIDS

RAPIDS requires CUDA 12.x:

```bash
# Check CUDA version
nvidia-smi

# If CUDA 11.x, GPU deduplication unavailable
```

### vLLM Build Failures

```bash
uv pip install --no-build-isolation vllm
```
