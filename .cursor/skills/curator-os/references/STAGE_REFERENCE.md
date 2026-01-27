# NeMo Curator Stage Reference

Complete catalog of processing stages organized by modality and category.

## Execution Hierarchy

```
WorkflowBase (orchestrates multiple pipelines)
    └── Pipeline (container for stages)
        └── CompositeStage (decomposes to ProcessingStages)
            └── ProcessingStage (single transformation)
```

**Important**: `FuzzyDeduplicationWorkflow` and `SemanticDeduplicationWorkflow` are `WorkflowBase` classes, not stages.

---

## Text Stages

### IO

| Stage | Type | Purpose | Source |
|-------|------|---------|--------|
| `JsonlReader` | ProcessingStage | Read JSONL files | `stages/text/io/reader/` |
| `ParquetReader` | ProcessingStage | Read Parquet files | `stages/text/io/reader/` |
| `JsonlWriter` | ProcessingStage | Write JSONL files | `stages/text/io/writer/` |
| `ParquetWriter` | ProcessingStage | Write Parquet files | `stages/text/io/writer/` |
| `MegatronTokenizerWriter` | ProcessingStage | Write for Megatron | `stages/text/io/writer/` |

### Classifiers

All classifiers are **CompositeStages** that decompose into tokenizer and model inference stages.

| Stage | Type | Purpose | GPU |
|-------|------|---------|-----|
| `QualityClassifier` | **CompositeStage** | General quality scoring | Yes |
| `DomainClassifier` | **CompositeStage** | Domain classification | Yes |
| `MultilingualDomainClassifier` | **CompositeStage** | Multi-language domains | Yes |
| `ContentTypeClassifier` | **CompositeStage** | Content type detection | Yes |
| `FineWebEduClassifier` | **CompositeStage** | Educational quality | Yes |
| `FineWebMixtralEduClassifier` | **CompositeStage** | Mixtral-based edu quality | Yes |
| `FineWebNemotronEduClassifier` | **CompositeStage** | Nemotron-based edu quality | Yes |
| `AegisClassifier` | **CompositeStage** | Safety classification (LlamaGuard) | Yes |
| `PromptTaskComplexityClassifier` | **CompositeStage** | Prompt complexity | Yes |
| `InstructionDataGuardClassifier` | **CompositeStage** | Instruction safety | Yes |

**Classifier Output Fields** (defaults):

| Classifier | Output Field | Output Type |
|------------|--------------|-------------|
| `QualityClassifier` | `quality_pred` | "high", "medium", "low" |
| `FineWebEduClassifier` | `fineweb-edu-score-label`, `-float`, `-int` | label, float, int |
| `DomainClassifier` | `domain_pred` | Domain category string |
| `AegisClassifier` | `aegis_pred` | "safe" or "O1"-"O13" |

### Heuristic Filters

| Filter | Purpose | Default Threshold |
|--------|---------|-------------------|
| `BoilerPlateStringFilter` | Remove boilerplate strings | - |
| `WordCountFilter` | Filter by word count | 50-100000 |
| `NonAlphaNumericFilter` | Non-alphanumeric ratio | max 0.25 |
| `SymbolsToWordsFilter` | Symbol to word ratio | max 0.1 |
| `NumbersFilter` | Number ratio | max 0.15 |
| `UrlsFilter` | URL ratio | max 0.2 |
| `BulletsFilter` | Bullet point ratio | max 0.9 |
| `WhiteSpaceFilter` | Whitespace ratio | max 0.25 |
| `ParenthesesFilter` | Parentheses ratio | max 0.1 |
| `LongWordFilter` | Long word ratio | max 0.1 |
| `MeanWordLengthFilter` | Average word length | 3-10 chars |
| `RepeatedLinesFilter` | Duplicate line ratio | max 0.3 |
| `RepeatedParagraphsFilter` | Duplicate paragraph ratio | max 0.3 |
| `RepeatedLinesByCharFilter` | Duplicate lines by char | max 0.2 |
| `RepeatedParagraphsByCharFilter` | Duplicate paragraphs by char | max 0.2 |
| `RepeatingTopNGramsFilter` | Top n-gram repetition | varies |
| `RepeatingDuplicateNGramsFilter` | Duplicate n-gram ratio | varies |
| `PunctuationFilter` | Punctuation ratio | max 0.1 |
| `EllipsisFilter` | Ellipsis ratio | max 0.3 |
| `CommonEnglishWordsFilter` | Common words presence | min 2 |
| `WordsWithoutAlphabetsFilter` | Words without letters | max 0.5 |
| `PornographicUrlsFilter` | Adult URL detection | threshold 0 |
| `TokenCountFilter` | Token count range | varies |
| `SubstringFilter` | Substring presence | - |
| `HistogramFilter` | Character histogram | - |

### Code Filters

| Filter | Purpose |
|--------|---------|
| `AlphaFilter` | Alphabetic character ratio |
| `GeneralCommentToCodeFilter` | Comment to code ratio |
| `HTMLBoilerplateFilter` | HTML boilerplate detection |
| `NumberOfLinesOfCodeFilter` | Lines of code count |
| `PerExtensionFilter` | Per-extension filtering |
| `PythonCommentToCodeFilter` | Python comment ratio |
| `TokenizerFertilityFilter` | Tokenizer efficiency |
| `XMLHeaderFilter` | XML header detection |

### FastText Filters

| Filter | Purpose |
|--------|---------|
| `FastTextLangId` | Language identification |
| `FastTextQualityFilter` | Quality scoring |

### Modifiers

| Modifier | Purpose |
|----------|---------|
| `BoilerPlateStringModifier` | Remove boilerplate |
| `MarkdownRemover` | Strip markdown |
| `NewlineNormalizer` | Normalize newlines |
| `QuotationRemover` | Remove quotes |
| `UnicodeReformatter` | Unicode normalization |
| `UrlRemover` | Remove URLs |
| `DocumentModifier` | Generic modifier |
| `FastTextLabelModifier` | Add FastText labels |
| `LineRemover` | Remove specific lines |
| `Slicer` | Slice documents |

### Modules

| Module | Purpose |
|--------|---------|
| `AddId` | Add document IDs |
| `DocumentJoiner` | Join documents |
| `DocumentSplitter` | Split documents |
| `Filter` | Apply filter logic |
| `Modify` | Apply modifier |
| `Score` | Add scores |
| `ScoreFilter` | Score + filter |

### Deduplication

| Component | Type | Purpose |
|-----------|------|---------|
| `FuzzyDeduplicationWorkflow` | **WorkflowBase** | Full fuzzy dedup pipeline |
| `SemanticDeduplicationWorkflow` | **WorkflowBase** | Full semantic dedup pipeline |
| `ExactDuplicateIdentification` | ProcessingStage | Hash-based exact matching |
| `MinHashStage` | ProcessingStage | Generate MinHash signatures |
| `LSHStage` | ProcessingStage | Locality-sensitive hashing |
| `BucketsToEdgesStage` | ProcessingStage | Convert buckets to graph |
| `ConnectedComponentsStage` | ProcessingStage | Find duplicate clusters |
| `IdentifyDuplicatesStage` | ProcessingStage | Mark duplicates for removal |

---

## Video Stages

### IO

| Stage | Type | Purpose |
|-------|------|---------|
| `VideoReader` | CompositeStage | Read videos (decomposes to VideoReaderStage) |
| `VideoReaderStage` | ProcessingStage | Download + extract metadata |
| `ClipWriterStage` | ProcessingStage | Write clips to storage |

### Clipping

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `TransNetV2ClipExtractionStage` | ProcessingStage | Yes (16GB) | ML scene detection |
| `FixedStrideExtractorStage` | ProcessingStage | No | Fixed-duration clips |
| `ClipTranscodingStage` | ProcessingStage | Optional | FFmpeg encoding |

### Embedding

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `CosmosEmbed1FrameCreationStage` | ProcessingStage | Yes | Prepare frames |
| `CosmosEmbed1EmbeddingStage` | ProcessingStage | Yes | NVIDIA Cosmos embeddings |
| `InternVideo2EmbeddingStage` | ProcessingStage | Yes | InternVideo2 embeddings |

### Captioning

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `CaptionPreparationStage` | ProcessingStage | No | Prepare video windows |
| `CaptionGenerationStage` | ProcessingStage | Yes (1 GPU) | Qwen VL captioning |
| `CaptionEnhancementStage` | ProcessingStage | Yes | Refine with Qwen LM |

### Filtering

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `MotionVectorDecodeStage` | ProcessingStage | No | Decode motion vectors |
| `MotionFilterStage` | ProcessingStage | No | Filter static clips |
| `ClipAestheticFilterStage` | ProcessingStage | Yes | Aesthetic quality |

---

## Image Stages

### Embedding

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `ImageEmbeddingStage` | ProcessingStage | Yes (0.25 GPU) | CLIP embeddings |

### Filtering

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `ImageAestheticFilterStage` | ProcessingStage | Yes | Aesthetic quality |
| `ImageNSFWFilterStage` | ProcessingStage | Yes | NSFW detection |

---

## Audio Stages

### Inference

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `InferenceAsrNemoStage` | ProcessingStage | Yes | NeMo ASR transcription |

### Metrics

| Stage | Type | GPU | Purpose |
|-------|------|-----|---------|
| `GetPairwiseWerStage` | ProcessingStage | No | Word Error Rate |

---

## Download Stages

| Stage | Type | Purpose |
|-------|------|---------|
| `CommonCrawlDownloadExtractStage` | CompositeStage | Download + extract CC |
| `WikipediaDownloadExtractStage` | CompositeStage | Download Wikipedia |
| `ArxivDownloadExtractStage` | CompositeStage | Download ArXiv |

---

## Task Types

| Modality | Task Type | Data Format |
|----------|-----------|-------------|
| Text | `DocumentBatch` | `pd.DataFrame` or `pa.Table` |
| Video | `VideoTask` | `Video` object |
| Image | `ImageBatch` | `list[ImageObject]` |
| Audio | `AudioBatch` | `dict` or `list[dict]` |
| Files | `FileGroupTask` | `list[str]` |
