---
description: "Concepts for integrating audio processing with text curation workflows in multimodal applications"
categories: ["concepts-architecture"]
tags: ["text-integration", "multimodal", "workflow-integration", "format-conversion", "cross-modal"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "audio-text"
---

(about-concepts-audio-text-integration)=

# Audio-Text Integration Concepts

This guide covers how audio processing integrates with text curation workflows in NeMo Curator, enabling seamless multi-modal data preparation and cross-modal quality assessment.

## Integration Architecture

Audio-text integration in NeMo Curator operates on several levels:

### Data Structure Integration

**Format Conversion**: `AudioBatch` to `DocumentBatch`

- All fields remain intact without remapping (`text` and `pred_text` remain unchanged)
- Audio metadata remains available as extra fields for downstream processing

**Metadata Preservation**: Audio characteristics remain intact during conversion

- File paths remain available for traceability and debugging
- Quality metrics (WER, duration) remain available for filtering operations
- Audio-specific metadata remains available for downstream processing stages

### Pipeline Integration

**Sequential Processing**: Audio to Text to Multi-Modal

```{mermaid}
flowchart LR
    A[Audio Files] --> B[InferenceAsrNemoStage] 
    B --> C[AudioToDocumentStage]
    C --> D[ScoreFilter<br/>Text Processing]
    D --> E[Integrated Output]
    
    style A fill:#e1f5fe
    style C fill:#ffcc02
    style E fill:#fff3e0
```

The `AudioToDocumentStage` provides the conversion bridge between audio and text processing workflows.

**Parallel Processing**: Simultaneous audio and text analysis

```{mermaid}
flowchart LR
    A[Audio Files] --> B[InferenceAsrNemoStage]
    C[Text Data] --> D[ScoreFilter<br/>Text Processing]
    B --> E[Cross-Modal<br/>Quality Assessment]
    D --> E
    E --> F[Filtered Output]
    
    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style F fill:#fff3e0
```

## Cross-Modal Quality Assessment

### Audio-Informed Text Quality

Use audio characteristics to enhance text quality assessment:

**Speech Rate Analysis**: Detect unnaturally fast or slow speech patterns using the `get_wordrate()` function

**Duration-Text Consistency**: Ensure transcription length matches audio duration

- Short audio with long text: Potential transcription errors
- Long audio with short text: Potential missing content
- Optimal ratio: ~3-5 characters per second of audio

### Text-Informed Audio Quality

Use text characteristics to assess audio quality:

**Transcription Completeness**: Detect incomplete or truncated speech

- Sentence fragments without proper endings
- Unusual punctuation patterns
- Incomplete words or phrases

**Content Coherence**: Assess semantic consistency

- Logical flow and coherence in transcriptions
- Domain-appropriate vocabulary usage
- Language consistency throughout sample

## Workflow Patterns

### Audio-First Workflows

Start with audio processing, then apply text curation:

```{mermaid}
flowchart TD
    A[Audio Files] --> B[InferenceAsrNemoStage<br/>ASR Transcription]
    B --> C[GetPairwiseWerStage<br/>Calculate WER Metrics]
    C --> D[ScoreFilter<br/>WER-based Filtering]
    D --> E[AudioToDocumentStage<br/>Convert to DocumentBatch]
    E --> F[ScoreFilter<br/>Text Quality Assessment]
    F --> G[Filter<br/>Metadata-based Filtering]
    G --> H[Text Enhancement Stages]
    H --> I[Processed Dataset]
    
    style A fill:#e1f5fe
    style E fill:#fff3e0
    style I fill:#e8f5e8
    
    classDef audioStage fill:#bbdefb
    classDef conversionStage fill:#ffcc02
    classDef textStage fill:#c8e6c9
    
    class B,C,D audioStage
    class E conversionStage
    class F,G,H textStage
```

**Use Cases**:

- Speech dataset curation for ASR training
- Podcast transcription and processing
- Lecture and educational content preparation

### Text-First Workflows

Start with text processing, then verify with audio:

```{mermaid}
flowchart TD
    A[Text Corpus] --> B[ScoreFilter<br/>Text Quality Assessment]
    B --> C[Filter<br/>Initial Text Filtering]
    C --> D[Audio File Matching<br/>Custom Stage]
    D --> E[InferenceAsrNemoStage<br/>ASR Validation]
    E --> F[GetPairwiseWerStage<br/>Cross-Modal Metrics]
    F --> G[ScoreFilter<br/>Consistency Filtering]
    G --> H[Validated Dataset]
    
    style A fill:#e8f5e8
    style D fill:#fff3e0
    style H fill:#e1f5fe
    
    classDef textStage fill:#c8e6c9
    classDef matchingStage fill:#ffcc02
    classDef audioStage fill:#bbdefb
    
    class B,C textStage
    class D matchingStage
    class E,F,G audioStage
```

**Use Cases**:

- Validating existing transcriptions with audio
- Creating audio-text pairs from separate sources
- Quality control for crowd-sourced transcriptions

## Data Flow Concepts

### Conversion Mechanisms

**AudioBatch to DocumentBatch**:

NeMo Curator provides the `AudioToDocumentStage` for converting audio processing results to text processing format:

```python
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

# Create the conversion stage
converter = AudioToDocumentStage()

# Example input AudioBatch data
audio_data = {
    "audio_filepath": "/audio.wav",
    "text": "ground truth", 
    "pred_text": "asr prediction",
    "wer": 15.2,
    "duration": 3.4
}

# The stage returns a list containing one DocumentBatch
# with the same fields preserved as pandas DataFrame
document_batches = converter.process(audio_batch)
document_batch = document_batches[0]  # Extract the single DocumentBatch

# All fields are preserved in the DocumentBatch:
# - audio_filepath, text, pred_text, wer, duration
```

**Note**: A built-in `DocumentBatch` to `AudioBatch` conversion stage is not provided. Create a custom stage if you need reverse conversion.

For practical usage examples and step-by-step implementation, refer to {doc}`/curate-audio/process-data/text-integration/index`.

### Metadata Flow

**Additive Processing**: Processing stages typically add metadata without removing existing fields

```python
# Stage 1: Initial loading
stage1_output = {"audio_filepath": "/audio.wav", "text": "transcription"}

# Stage 2: ASR inference  
stage2_output = {**stage1_output, "pred_text": "asr result"}

# Stage 3: Quality assessment
stage3_output = {**stage2_output, "wer": 15.2, "duration": 3.4}

# Stage 4: Text processing (after conversion)
stage4_output = {**stage3_output, "word_count": 6, "language": "en"}
```

## Quality Assessment Integration

### Available Quality Metrics

NeMo Curator provides these audio quality assessment capabilities:

**Word Error Rate (WER) Analysis**:

- WER correlates with transcription accuracy
- Available through `GetPairwiseWerStage`
- Measures percentage of incorrect words between ground truth and ASR predictions

**Duration and Speech Rate Analysis**:

- Duration validation using `GetAudioDurationStage`  
- Speech rate calculation using `get_wordrate()` function
- Character rate calculation using `get_charrate()` function

**Individual Quality Dimensions**:

- **Technical Quality**: File integrity, format compliance, duration validation
- **Content Quality**: Transcription accuracy via WER/CER metrics
- **Speech Rate Quality**: Words/characters per second analysis

## Performance and Scaling

### Memory Considerations

**AudioBatch Memory Usage**:

- Metadata storage scales linearly with batch size
- Audio files loaded on-demand, not cached in memory
- Large batches increase processing efficiency but consume more RAM

**Conversion Overhead**:

- AudioBatch → DocumentBatch conversion is lightweight
- Metadata copying has minimal performance impact
- Batch size affects conversion performance

### Processing Efficiency

**Sequential vs. Parallel Integration**:

**Sequential Processing**: Audio to Text (lower memory, slower)

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.modules.score_filter import ScoreFilter
from nemo_curator.filters import WordCountFilter  # Example filter

# Define a text quality filter
text_quality_filter = WordCountFilter(min_words=10)

# Process audio completely first
audio_pipeline = Pipeline(
    name="audio_processing",
    stages=[
        InferenceAsrNemoStage(model_name="stt_en_fastconformer_transducer_large"),
        AudioToDocumentStage()
    ]
)
audio_results = audio_pipeline.run(executor)

# Then process text
text_pipeline = Pipeline(
    name="text_processing", 
    stages=[
        ScoreFilter(filter_obj=text_quality_filter)
    ]
)
final_results = text_pipeline.run(executor, initial_tasks=audio_results)
```

**Parallel Processing**: Audio and Text (higher memory, faster)

```python
# Run a single pipeline that includes both audio and text stages using an executor
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(name="audio_text", stages=[
    InferenceAsrNemoStage(model_name="stt_en_fastconformer_transducer_large"),
    GetPairwiseWerStage(),
    AudioToDocumentStage(),
    ScoreFilter(filter_obj=text_quality_filter)
])
results = pipeline.run()
```

### Scaling Strategies

**Horizontal Scaling**: Distribute across several workers

- Partition audio files across workers
- Independent processing with final aggregation
- Load balancing based on audio duration

**Vertical Scaling**: Optimize single-machine performance

- GPU acceleration for ASR inference
- Batch size optimization for hardware
- Memory management for large datasets

## Design Principles

### Modularity

**Separation of Concerns**: Audio and text processing remain independent

- Audio stages focus on speech-specific operations
- Text stages handle language processing
- Integration stages manage cross-modal operations

**Modular Architecture**: Mix and match audio and text processing stages

- Flexible pipeline construction
- Reusable stage components
- Configurable integration points

### Extensibility

**Custom Integration Patterns**: Support for domain-specific workflows

- Custom conversion logic
- Specialized quality metrics
- Domain-specific filtering rules

**Plugin Architecture**: Easy addition of new integration methods

- Custom stage implementations
- External tool integration
- Specialized format support
