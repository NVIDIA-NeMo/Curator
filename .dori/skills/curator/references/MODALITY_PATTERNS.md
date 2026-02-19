# Modality-Specific Pipeline Patterns

Common pipeline patterns for each data modality in NeMo Curator.

**Note**: The diagrams below show conceptual pipeline flows. For actual YAML configurations, use the `/curate` skill or see the template files in `skills/curate/assets/`.

---

## Text Curation Patterns

### Pattern 1: Basic Quality Filtering

```
Read → Heuristic Filters → Write
       (WordCount, NonAlphaNumeric, etc.)
```

**Use when**: Quick quality pass on web-scraped text

**Key stages**:
- `ParquetReader` / `JsonlReader`
- `ScoreFilter` with heuristic filters (CPU)
- `ParquetWriter` / `JsonlWriter`

### Pattern 2: Full Text Curation

```
Read → Filters → Classify → Deduplicate → Write
       (25+)     (Quality)   (Fuzzy)
```

**Use when**: Training data preparation for LLMs

**Key stages**:
- Heuristic filters (CPU-only)
- `QualityClassifier` (GPU) - outputs "high"/"medium"/"low" in `quality_pred`
- `FuzzyDeduplicationWorkflow` (run separately)

### Pattern 3: Domain-Specific Filtering

```
Read → Domain Classify → Filter → Edu Score → Write
                         (by domain)
```

**Use when**: Creating domain-specific training sets

**Key stages**:
- `DomainClassifier` - outputs category in `domain_pred`
- Use `filter_by` parameter to keep specific domains
- `FineWebEduClassifier` - outputs scores in `fineweb-edu-score-*` fields

### Pattern 4: Safety Filtering

```
Read → Aegis Classify → Filter Safe → Write
                        (keep "safe")
```

**Use when**: Filtering for safety-critical applications

**Key stages**:
- `AegisClassifier` - outputs "safe" or "O1"-"O13" in `aegis_pred`
- Use `filter_by: ["safe"]` to keep only safe content

---

## Video Curation Patterns

### Pattern 1: Scene Detection + Captioning

```
Read → Scene Detect → Caption Prep → Caption Gen → Write
       (TransNetV2)                   (Qwen VL)
```

**Use when**: Creating captioned video datasets

**Key stages**:
- `VideoReader` (CompositeStage)
- `TransNetV2ClipExtractionStage` (GPU)
- `CaptionPreparationStage` (CPU)
- `CaptionGenerationStage` (GPU)
- `ClipWriterStage`

### Pattern 2: Fixed-Stride Embedding

```
Read → Fixed Clips → Embed → Write
       (uniform)     (Cosmos)
```

**Use when**: Uniform sampling for embedding generation

**Key stages**:
- `FixedStrideExtractorStage` (CPU)
- `CosmosEmbed1EmbeddingStage` (GPU)

### Pattern 3: Full Video Pipeline

```
Read → Clip → Motion Filter → Caption → Embed → Aesthetic Filter → Write
```

**Use when**: High-quality video training data

---

## Image Curation Patterns

### Pattern 1: CLIP Embedding + Dedup

```
Read → CLIP Embed → Semantic Dedup → Write
```

**Use when**: Removing duplicate/near-duplicate images

**Key stages**:
- `ImageEmbeddingStage` (GPU, CLIP-based)
- `SemanticDeduplicationWorkflow`

### Pattern 2: Quality Filtering

```
Read → Embed → Aesthetic Filter → NSFW Filter → Write
```

**Use when**: Filtering for quality and safety

**Key stages**:
- `ImageAestheticFilterStage` (GPU)
- `ImageNSFWFilterStage` (GPU)

---

## Audio Curation Patterns

### Pattern 1: ASR Transcription

```
Read → ASR Transcribe → Write (with transcripts)
       (NeMo ASR)
```

**Use when**: Converting speech to text

**Key stages**:
- `InferenceAsrNemoStage` (GPU)

### Pattern 2: Quality Assessment

```
Read → ASR → WER Calculate → Filter by WER → Write
```

**Use when**: Filtering by transcription quality

**Key stages**:
- `InferenceAsrNemoStage` (GPU)
- `WERCalculationStage` (CPU)

---

## Deduplication Patterns

### Exact Deduplication

```python
ExactDuplicateIdentification(
    input_path=input_path,
    output_path=output_path,
    text_field="text",
)
```

**Best for**: Quick pass, small datasets, exact copies

### Fuzzy Deduplication

```python
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

### 2. Use Built-in Filtering

Classifiers have a `filter_by` parameter for filtering by output labels:

```yaml
- _target_: nemo_curator.stages.text.classifiers.QualityClassifier
  model_inference_batch_size: 256
  filter_by: ["high", "medium"]  # Keep only these labels
```

### 3. Checkpoint Frequently

For long pipelines, write intermediate outputs after each major step.

### 4. Monitor GPU Utilization

If GPU stages are underutilized:
- Increase `model_inference_batch_size`
- Add more CPU workers feeding GPU stages
- Check for I/O bottlenecks
