---
description: "Convert processed audio data to text processing workflows for multimodal applications and downstream text curation"
categories: ["audio-processing"]
tags: ["text-integration", "multimodal", "format-conversion", "pipeline-integration", "transcription-processing"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "audio-text"
---

# Text Integration for Audio Data

Convert processed audio data to text processing workflows, enabling seamless integration between audio curation and NeMo Curator's comprehensive text processing capabilities.

## How it Works

Text integration bridges audio and text modalities by:

1. **Format Conversion**: Transform `AudioBatch` objects to `DocumentBatch` format
2. **Transcription Processing**: Apply text curation filters to ASR-generated transcriptions
3. **Metadata Preservation**: Maintain audio-specific metadata during conversion
4. **Pipeline Integration**: Enable mixed audio-text processing workflows

## Basic Conversion

### AudioBatch to DocumentBatch

```python
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

# Convert audio data to text format
converter = AudioToDocumentStage()

# Input: AudioBatch with transcriptions
audio_batch = AudioBatch(data=[
    {
        "audio_filepath": "/data/audio/sample.wav",
        "text": "ground truth text",
        "pred_text": "asr predicted text", 
        "wer": 12.5,
        "duration": 3.2
    }
])

# Output: DocumentBatch compatible with text processing
batches = converter.process(audio_batch)
document_batch = batches[0]
```

### Preserving Audio Metadata

```python
# Audio metadata is preserved during conversion
converted_data = {
    "text": "ground truth text",                 # Original transcription
    "audio_filepath": "/data/audio/sample.wav",  # Original audio reference
    "pred_text": "asr predicted text",           # ASR prediction
    "wer": 12.5,                                   # Quality metrics
    "duration": 3.2,                               # Audio characteristics
}
```

## Text Processing Integration

### Apply Text Filters to Transcriptions

```python
from nemo_curator.stages.text.filters import WordCountFilter, NonAlphaNumericFilter
from nemo_curator.stages.text.modules import ScoreFilter

# Create integrated audio-text pipeline
pipeline = Pipeline(name="audio_text_integration")

# Audio processing stages
pipeline.add_stage(asr_inference_stage)
pipeline.add_stage(wer_calculation_stage)

# Convert to text format
pipeline.add_stage(AudioToDocumentStage())

# Apply text quality filters to transcriptions
pipeline.add_stage(
    ScoreFilter(
        WordCountFilter(min_words=3, max_words=100),
        text_field="pred_text",  # Filter ASR predictions
        score_field="word_count_score"
    )
)

pipeline.add_stage(
    ScoreFilter(
        NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.3),
        text_field="pred_text",
        score_field="alpha_numeric_score"
    )
)
```

### Language Detection on Transcriptions

```python
from nemo_curator.stages.text.filters import FastTextLangId
from nemo_curator.stages.text.modules import ScoreFilter

# Detect language of ASR predictions
pipeline.add_stage(
    ScoreFilter(
        FastTextLangId(model_path="/path/to/lid.176.bin", min_langid_score=0.3),
        text_field="pred_text",
        score_field="language",
    )
)
```

## Cross-Modal Workflows

### Audio-Text Paired Processing

```python
def create_multimodal_pipeline(audio_manifest: str) -> Pipeline:
    """Create pipeline processing both audio and text content."""
    
    pipeline = Pipeline(name="multimodal_audio_text")
    
    # Load audio data
    pipeline.add_stage(JsonlReader(file_paths=audio_manifest))
    
    # Audio processing branch
    audio_branch = [
        InferenceAsrNemoStage(model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"),
        GetPairwiseWerStage(),
        GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
    ]
    
    for stage in audio_branch:
        pipeline.add_stage(stage)
    
    # Convert to text processing
    pipeline.add_stage(AudioToDocumentStage())
    
    # Text processing branch
    text_branch = [
        # Apply text filters to both ground truth and predictions
        ScoreFilter(WordCountFilter(min_words=5), text_field="text"),
        ScoreFilter(WordCountFilter(min_words=5), text_field="pred_text"),
        
        # Language consistency check
        ScoreFilter(FastTextLangId(model_path="/path/to/lid.176.bin", min_langid_score=0.3), text_field="text", score_field="gt_language"),
        ScoreFilter(FastTextLangId(model_path="/path/to/lid.176.bin", min_langid_score=0.3), text_field="pred_text", score_field="pred_language"),
    ]
    
    for stage in text_branch:
        pipeline.add_stage(stage)
    
    return pipeline
```

### Cross-Modal Quality Assessment

```python
@dataclass
class AudioTextQualityStage(LegacySpeechStage):
    """Assess quality using both audio and text features."""
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        # Audio quality factors
        duration = data_entry.get("duration", 0)
        wer = data_entry.get("wer", 100)
        
        # Text quality factors
        text = data_entry.get("text", "")
        pred_text = data_entry.get("pred_text", "")
        
        # Combined quality score
        audio_score = max(0, 100 - wer)  # Higher score for lower WER
        
        text_length_score = min(100, len(text.split()) * 10)  # Prefer longer texts
        
        duration_score = 100 if 1.0 <= duration <= 15.0 else 50  # Optimal duration range
        
        # Weighted combination
        combined_quality = (
            0.5 * audio_score +      # 50% weight on transcription accuracy
            0.3 * text_length_score + # 30% weight on text content
            0.2 * duration_score      # 20% weight on duration
        )
        
        data_entry["multimodal_quality"] = combined_quality
        return [AudioBatch(data=data_entry)]
```

## Advanced Integration Patterns

### Conditional Text Processing

```python
def create_conditional_text_pipeline() -> Pipeline:
    """Apply different text processing based on audio quality."""
    
    pipeline = Pipeline(name="conditional_text_processing")
    
    # Audio processing
    pipeline.add_stage(asr_stage)
    pipeline.add_stage(wer_stage)
    pipeline.add_stage(AudioToDocumentStage())
    
    # Conditional text processing based on WER
    @processing_stage(name="conditional_text_filter")
    def conditional_filter(document_batch: DocumentBatch) -> DocumentBatch:
        filtered_data = []
        
        for doc in document_batch.data:
            wer = doc.get("wer", 100)
            
            if wer <= 20:  # High quality: minimal filtering
                if len(doc["pred_text"].split()) >= 3:
                    filtered_data.append(doc)
            elif wer <= 50:  # Medium quality: strict filtering  
                if len(doc["pred_text"].split()) >= 5:
                    # Additional text quality checks
                    text = doc["pred_text"]
                    if text.count(" ") >= 2:  # At least 3 words
                        filtered_data.append(doc)
            # wer > 50: Skip low quality samples
        
        return DocumentBatch(data=filtered_data)
    
    pipeline.add_stage(conditional_filter)
    return pipeline
```

## Output Formats

### Integrated Manifests

Combined audio-text manifests preserve both modalities:

```json
{
    "audio_filepath": "/data/audio/sample.wav",
    "text": "ground truth transcription",
    "pred_text": "asr predicted transcription",
    "wer": 15.2,
    "duration": 3.4,
    "word_count_score": 6,
    "language": "en",
    "multimodal_quality": 87.5
}
```

### Separated Outputs

Export audio and text data separately while maintaining relationships:

```python
# Export audio metadata
audio_metadata = {
    "audio_id": "sample_001", 
    "audio_filepath": "/data/audio/sample.wav",
    "duration": 3.4,
    "wer": 15.2
}

# Export text content
text_content = {
    "audio_id": "sample_001",  # Link to audio
    "text": "asr predicted transcription",
    "word_count_score": 6,
    "language": "en"
}
```

## Performance Considerations

### Memory Efficiency

```python
# Stream large datasets through conversion
@processing_stage(name="streaming_audio_to_text")
def streaming_converter(audio_batch: AudioBatch) -> DocumentBatch:
    """Memory-efficient conversion for large datasets."""
    
    # Process in smaller chunks
    chunk_size = 100
    converted_data = []
    
    for i in range(0, len(audio_batch.data), chunk_size):
        chunk = audio_batch.data[i:i + chunk_size]
        
        # Convert chunk to document format
        for item in chunk:
            converted_item = {
                "text": item["pred_text"],
                "metadata": {
                    "audio_filepath": item["audio_filepath"],
                    "wer": item.get("wer"),
                    "duration": item.get("duration")
                }
            }
            converted_data.append(converted_item)
    
    return DocumentBatch(data=converted_data)
```

## Related Topics

- **[Audio Processing Overview](../index.md)** - Complete audio processing workflow
- **[Quality Assessment](../quality-assessment/index.md)** - Audio quality metrics
- **[Text Curation](../../../curate-text/index.md)** - Text processing capabilities
- **[Cross-Modal Concepts](../../../about/concepts/index.md)** - Cross-modal processing concepts
