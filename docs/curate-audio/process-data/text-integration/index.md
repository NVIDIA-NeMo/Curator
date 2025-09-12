---
description: "Convert processed audio data to DocumentBatch format for downstream processing"
categories: ["audio-processing"]
tags: ["format-conversion", "audio-to-text", "documentbatch"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-text"
---

(audio-process-data-text-integration)=
# Text Integration for Audio Data

Convert processed audio data from `AudioBatch` to `DocumentBatch` format using the built-in `AudioToDocumentStage`. This enables you to export audio processing results or integrate with custom text processing workflows.

## How it Works

The `AudioToDocumentStage` provides basic format conversion:

1. **Format Conversion**: Transform `AudioBatch` objects to `DocumentBatch` format
2. **Metadata Preservation**: All fields from the audio data are preserved in the conversion
3. **Export Ready**: Convert audio processing results to pandas DataFrame format for analysis or export

## Basic Conversion

### AudioBatch to DocumentBatch

```python
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.tasks import AudioBatch

# Convert audio data to DocumentBatch format
converter = AudioToDocumentStage()

# Input: AudioBatch with audio processing results
audio_batch = AudioBatch(data=[
    {
        "audio_filepath": "/data/audio/sample.wav",
        "text": "ground truth text",
        "pred_text": "asr predicted text", 
        "wer": 12.5,
        "duration": 3.2
    }
])

# Output: DocumentBatch with pandas DataFrame
document_batches = converter.process(audio_batch)
document_batch = document_batches[0]

# Access the converted data
print(f"Converted {len(document_batch.data)} audio records to DocumentBatch")
```

### What Gets Preserved

The conversion preserves all fields from your audio processing pipeline:

```python
# All audio processing results are maintained:
# - audio_filepath: Original audio file reference
# - text: Ground truth transcription (if available)  
# - pred_text: ASR prediction
# - wer: Word Error Rate (if calculated)
# - duration: Audio duration (if calculated)
# - Any other metadata fields you've added
```

## Integration in Pipelines

### Complete Audio Processing with Export

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter

# Create pipeline that processes audio and exports results
pipeline = Pipeline(name="audio_processing_with_export")

# Audio processing stages
pipeline.add_stage(InferenceAsrNemoStage(model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"))
pipeline.add_stage(GetPairwiseWerStage(text_key="text", pred_text_key="pred_text"))
pipeline.add_stage(GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration"))

# Convert to DocumentBatch for export
pipeline.add_stage(AudioToDocumentStage())

# Export results
pipeline.add_stage(JsonlWriter(path="/output/processed_audio_results"))
```

## Custom Integration

If you need to apply text processing to your ASR transcriptions, you will need to implement custom stages. The `AudioToDocumentStage` provides the foundation for this by converting to the standard `DocumentBatch` format.

### Example: Custom Text Processing

```python
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.tasks import DocumentBatch
import pandas as pd

@processing_stage(name="custom_transcription_filter")
def filter_transcriptions(document_batch: DocumentBatch) -> DocumentBatch:
    """Custom filtering of ASR transcriptions."""
    
    # Access the pandas DataFrame
    df = document_batch.data
    
    # Example: Filter by transcription length
    df = df[df['pred_text'].str.len() > 10]  # Keep transcriptions > 10 chars
    
    # Example: Filter by WER if available
    if 'wer' in df.columns:
        df = df[df['wer'] < 50.0]  # Keep WER < 50%
    
    return DocumentBatch(data=df)
```

## Output Format

After conversion, your data will be in `DocumentBatch` format with a pandas DataFrame:

```python
# Example output structure
document_batch.data  # pandas DataFrame with columns:
# - audio_filepath: "/path/to/audio.wav"
# - text: "ground truth transcription" 
# - pred_text: "asr prediction"
# - wer: 15.2
# - duration: 3.4
# - [any other fields from your audio processing]
```

## Limitations

:::{note}
**Text Processing Integration**: NeMo Curator's text processing stages are designed for `DocumentBatch` inputs, but they may not be optimized for audio-derived transcriptions. You may need to implement custom processing for audio-specific workflows.
:::

## Related Topics

- **[Audio Processing Overview](../index.md)** - Complete audio processing workflow
- **[Quality Assessment](../quality-assessment/index.md)** - Audio quality metrics and filtering
- **[ASR Inference](../asr-inference/index.md)** - Speech recognition processing