# Modality-Specific Pipeline Patterns

Common pipeline patterns for each data modality in NeMo Curator.

---

## Text Curation Patterns

### Pattern 1: Basic Quality Filtering

```yaml
stages:
  - reader: ParquetReader
  - filters:
      - WordCountFilter: {min: 50, max: 100000}
      - NonAlphaNumericFilter: {max: 0.25}
      - SymbolsToWordsFilter: {max: 0.1}
  - writer: ParquetWriter
```

**Use when**: Quick quality pass on web-scraped text

### Pattern 2: Full Text Curation

```yaml
stages:
  - reader: ParquetReader
  - filters: [25 heuristic filters]
  - classifier: QualityClassifier
  - dedup: FuzzyDeduplicationWorkflow
  - writer: ParquetWriter
```

**Use when**: Training data preparation for LLMs

### Pattern 3: Domain-Specific Filtering

```yaml
stages:
  - reader: JsonlReader
  - classifier: DomainClassifier
  - filter: ScoreFilter (by domain)
  - classifier: FineWebEduClassifier
  - filter: ScoreFilter (by edu_score >= 3)
  - writer: ParquetWriter
```

**Use when**: Creating domain-specific training sets

### Pattern 4: Safety Filtering

```yaml
stages:
  - reader: ParquetReader
  - classifier: AegisClassifier
  - filter: Remove unsafe content
  - classifier: InstructionDataGuardClassifier  
  - writer: ParquetWriter
```

**Use when**: Filtering for safety-critical applications

---

## Video Curation Patterns

### Pattern 1: Scene Detection + Captioning

```yaml
stages:
  - reader: VideoReader
  - clipping: TransNetV2ClipExtractionStage  # GPU
  - caption_prep: CaptionPreparationStage
  - captioning: CaptionGenerationStage  # GPU
  - writer: ClipWriterStage
```

**Use when**: Creating captioned video datasets

### Pattern 2: Fixed-Stride Embedding

```yaml
stages:
  - reader: VideoReader
  - clipping: FixedStrideExtractorStage  # CPU
  - embedding: CosmosEmbed1EmbeddingStage  # GPU
  - writer: ClipWriterStage
```

**Use when**: Uniform sampling for embedding generation

### Pattern 3: Full Video Pipeline

```yaml
stages:
  - reader: VideoReader
  - clipping: TransNetV2ClipExtractionStage
  - motion_decode: MotionVectorDecodeStage
  - motion_filter: MotionFilterStage
  - caption_prep: CaptionPreparationStage
  - captioning: CaptionGenerationStage
  - embedding: CosmosEmbed1EmbeddingStage
  - aesthetic_filter: ClipAestheticFilterStage
  - writer: ClipWriterStage
```

**Use when**: High-quality video training data

---

## Image Curation Patterns

### Pattern 1: CLIP Embedding + Dedup

```yaml
stages:
  - reader: ImageReader
  - embedding: ImageEmbeddingStage  # CLIP
  - dedup: SemanticDeduplicationWorkflow
  - writer: ImageWriter
```

**Use when**: Removing duplicate/near-duplicate images

### Pattern 2: Quality Filtering

```yaml
stages:
  - reader: ImageReader
  - embedding: ImageEmbeddingStage
  - aesthetic_filter: AestheticFilterStage
  - nsfw_filter: NSFWFilterStage
  - writer: ImageWriter
```

**Use when**: Filtering for quality and safety

---

## Audio Curation Patterns

### Pattern 1: ASR Transcription

```yaml
stages:
  - reader: AudioReader
  - asr: InferenceAsrNemoStage  # GPU
  - writer: AudioWriter (with transcripts)
```

**Use when**: Converting speech to text

### Pattern 2: Quality Assessment

```yaml
stages:
  - reader: AudioReader
  - asr: InferenceAsrNemoStage
  - wer: WERCalculationStage
  - filter: WER threshold filter
  - writer: AudioWriter
```

**Use when**: Filtering by transcription quality

---

## Common Crawl Patterns

### Pattern 1: Full CC Pipeline

```yaml
workflow:
  - download: CommonCrawlDownloadExtractStage
  - filter: [All 25 heuristic filters]
  - classifier: QualityClassifier
  - classifier: FineWebEduClassifier
  - dedup: FuzzyDeduplicationWorkflow
  - writer: ParquetWriter
```

**Estimated reduction**: 70-90% (raw CC → curated)

### Pattern 2: Language-Specific CC

```yaml
workflow:
  - download: CommonCrawlDownloadExtractStage
  - lang_filter: FastTextLangId (keep target language)
  - filter: Language-specific heuristics
  - dedup: FuzzyDeduplicationWorkflow
  - writer: ParquetWriter
```

**Use when**: Building non-English datasets

---

## Deduplication Patterns

### Exact Deduplication

```python
# Fast, low memory, catches exact matches only
ExactDuplicateIdentification(
    input_path=input_path,
    output_path=output_path,
    text_field="text",
)
```

**Best for**: Quick pass, small datasets, exact copies

### Fuzzy Deduplication

```python
# MinHash + LSH, catches near-duplicates
FuzzyDeduplicationWorkflow(
    input_path=input_path,
    output_path=output_path,
    cache_path=cache_path,
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13,
)
```

**Best for**: Web data, templated content, ~80% similarity threshold

### Semantic Deduplication

```python
# Embedding-based, catches paraphrases
SemanticDeduplicationWorkflow(
    input_path=input_path,
    output_path=output_path,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.9,
)
```

**Best for**: High-quality filtering, catching paraphrases

---

## Pipeline Composition Tips

### 1. Order Matters

```
Cheap filters first → Expensive classifiers → Deduplication last
```

This reduces data volume before expensive operations.

### 2. Batch Appropriately

| Stage Type | Recommended Batch |
|------------|-------------------|
| Filters | 1000+ documents |
| Classifiers | 32-128 documents |
| Dedup | Full dataset |

### 3. Checkpoint Frequently

For long pipelines, write intermediate outputs:

```yaml
pipeline:
  - stage: filters
    checkpoint: /data/checkpoint/filtered
  - stage: classifiers
    checkpoint: /data/checkpoint/classified
  - stage: dedup
    output: /data/final
```

### 4. Monitor GPU Utilization

If GPU stages are <80% utilized:
- Increase batch size
- Add more CPU workers feeding GPU stages
- Check for IO bottlenecks
