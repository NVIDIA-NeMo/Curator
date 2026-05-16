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

The `AudioToDocumentStage` provides straightforward format conversion between NeMo Curator's audio and text data structures:

1. **Format Conversion**: Transform `AudioBatch` objects to `DocumentBatch` format
2. **Metadata Preservation**: All fields from the audio data are preserved in the conversion
3. **Export Ready**: Convert audio processing results to pandas DataFrame format for analysis or export

**Common use cases:**
- Export ASR results and quality metrics for analysis
- Save filtered audio datasets with transcriptions
- Integrate audio processing outputs with downstream text workflows

## Basic Conversion

### AudioBatch to DocumentBatch

Use `AudioToDocumentStage` to convert audio processing results to document format:

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

**Parameters:**
- `AudioToDocumentStage()` has no configuration parameters; it performs direct format conversion

**Returns:**
- List of `DocumentBatch` objects containing a pandas DataFrame with all original audio fields

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

:::{note}
Field names and values are preserved exactly as they appear in the `AudioBatch`. No data transformation or cleaning is performed during conversion.
:::

## Integration in Pipelines

### Complete Audio Processing with Export

The most common use case is adding `AudioToDocumentStage` at the end of your audio pipeline to enable result export:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources

# Create pipeline that processes audio and exports results
pipeline = Pipeline(name="audio_processing_with_export")

# 1. Load audio data
pipeline.add_stage(CreateInitialManifestFleursStage(
    lang="en_us",
    split="test",
    raw_data_dir="./audio_data"
).with_(batch_size=8))

# 2. Run ASR inference
pipeline.add_stage(InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
    pred_text_key="pred_text"
).with_(resources=Resources(gpus=1.0)))

# 3. Calculate quality metrics
pipeline.add_stage(GetPairwiseWerStage(
    text_key="text",
    pred_text_key="pred_text",
    wer_key="wer"
))
pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# 4. Convert to DocumentBatch for export
pipeline.add_stage(AudioToDocumentStage())

# 5. Export to JSONL format
pipeline.add_stage(JsonlWriter(path="/output/processed_audio_results"))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)
```

**Output format:** The `JsonlWriter` creates a JSONL file where each line contains one audio sample with all fields:

```json
{"audio_filepath": "/data/audio/sample1.wav", "text": "hello world", "pred_text": "hello world", "wer": 0.0, "duration": 1.5}
{"audio_filepath": "/data/audio/sample2.wav", "text": "test audio", "pred_text": "test odio", "wer": 50.0, "duration": 2.1}
```

## Custom Integration

While `AudioToDocumentStage` converts audio data to `DocumentBatch` format, NeMo Curator's built-in text processing stages (filters, classifiers, etc.) are designed for text documents, not audio transcriptions. For audio-specific text processing, implement custom stages that operate on the converted `DocumentBatch` data.

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

    return DocumentBatch(
        data=df,
        task_id=document_batch.task_id,
        dataset_name=document_batch.dataset_name
    )
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

:::{important}
**Text Processing Integration**: NeMo Curator's text processing stages are designed for `DocumentBatch` inputs (text documents such as articles, web pages), but they are not designed for audio-derived transcriptions. You should implement custom processing stages for audio-specific workflows.

**Reasons for incompatibility:**
- Text filters assume document-level content (e.g., paragraph structure, word count thresholds designed for articles)
- ASR transcriptions have different characteristics (shorter, can contain recognition errors, conversational language)
- Audio-specific metrics (WER, duration, speech rate) require custom filtering logic

**Recommendation:** Use `PreserveByValueStage` for audio quality filtering, or create custom stages for transcription-specific processing.
:::

## Related Topics

- **[Audio Processing Overview](../index.md)** - Complete audio processing workflow
- **[Quality Assessment](../quality-assessment/index.md)** - Audio quality metrics and filtering
- **[ASR Inference](../asr-inference/index.md)** - Speech recognition processing
