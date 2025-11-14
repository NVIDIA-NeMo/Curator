---
description: "Assess and filter audio quality using transcription accuracy metrics, duration analysis, and custom quality measures"
categories: ["audio-processing"]
tags: ["quality-assessment", "wer-filtering", "duration-filtering", "quality-metrics", "speech-quality"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "audio-only"
---

(audio-process-data-quality-assessment)=
# Quality Assessment for Audio Data

Filter audio quality using transcription accuracy metrics, duration analysis, and custom quality measures to ensure high-quality speech datasets for ASR training.

## How it Works

Audio quality assessment in NeMo Curator focuses on speech-specific metrics that correlate with training data quality:

1. **Transcription Accuracy**: Word Error Rate (WER) and Character Error Rate (CER) between ground truth and ASR predictions
2. **Duration Analysis**: Audio length validation and speech rate calculations  
3. **Value-based Filtering**: Configurable filtering using comparison operators

## Quality Metrics

### Word Error Rate (WER)

The primary metric for assessing transcription quality:

```python
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage

# Calculate WER for each audio sample
wer_stage = GetPairwiseWerStage(
    text_key="text",           # Ground truth transcription
    pred_text_key="pred_text", # ASR prediction
    wer_key="wer"             # Output WER field
)
```

WER measures the percentage of words that differ between ground truth and predicted transcriptions:

- **WER = 0%**: Perfect transcription match
- **WER = 25%**: Good quality (1 in 4 words incorrect)
- **WER = 50%**: Moderate quality
- **WER > 75%**: Poor quality (consider filtering)

### Character Error Rate (CER)

More granular accuracy measurement at the character level:

```python
from nemo_curator.stages.audio.metrics.get_wer import get_cer

# Calculate CER programmatically
cer_value = get_cer("hello world", "helo world")  # Returns 9.09
```

:::{note} The WER and CER utilities depend on the `editdistance` package.
:::

### Speech Rate Metrics

Analyze speaking speed and content density:

```python
from nemo_curator.stages.audio.metrics.get_wer import get_wordrate, get_charrate

# Calculate words per second
word_rate = get_wordrate("hello world example", 2.5)  # 1.2 words/second

# Calculate characters per second  
char_rate = get_charrate("hello world", 2.0)  # 5.5 chars/second
```

## Filtering Strategies

### WER-based Filtering

Filter audio samples based on transcription accuracy:

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Keep samples with WER <= 30% (high quality)
high_quality_filter = PreserveByValueStage(
    input_value_key="wer",
    target_value=30.0,
    operator="le"  # less than or equal
)

# Remove samples with WER >= 80% (very poor quality)
poor_quality_filter = PreserveByValueStage(
    input_value_key="wer", 
    target_value=80.0,
    operator="lt"  # less than
)
# Preserves only entries with WER < 80%
```

### Duration-based Filtering

Filter by audio length to remove short or long samples:

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Keep samples between 1-30 seconds
duration_min_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"  # greater than or equal
)

duration_max_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=30.0, 
    operator="le"  # less than or equal
)
```

### Combined Quality Filtering

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage

# Create multi-stage quality pipeline
quality_pipeline = Pipeline(name="audio_quality_assessment")

# Calculate all metrics
quality_pipeline.add_stage(GetPairwiseWerStage())
quality_pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# Apply filters in sequence
filters = [
    PreserveByValueStage("wer", 50.0, "le"),        # WER <= 50%
    PreserveByValueStage("duration", 1.0, "ge"),     # Duration >= 1s
    PreserveByValueStage("duration", 20.0, "le"),    # Duration <= 20s
]

for filter_stage in filters:
    quality_pipeline.add_stage(filter_stage)
```

## Operator Options

The `PreserveByValueStage` supports several comparison operators:

| Operator | Description | Example Use Case |
|----------|-------------|------------------|
| `"eq"` | Equal to | Exact duration matching |
| `"ne"` | Not equal to | Exclude specific values |
| `"lt"` | Less than | Max thresholds |
| `"le"` | Less than or equal | Quality thresholds |
| `"gt"` | Greater than | Min thresholds |
| `"ge"` | Greater than or equal | Min requirements |

## Complete Quality Assessment Pipeline

Here's a complete working example that demonstrates quality assessment:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter

# Create complete quality assessment pipeline
pipeline = Pipeline(name="audio_quality_assessment")

# 1. Load data
pipeline.add_stage(CreateInitialManifestFleursStage(
    lang="hy_am", 
    split="dev", 
    raw_data_dir="./audio_data"
).with_(batch_size=4))

# 2. ASR inference
pipeline.add_stage(InferenceAsrNemoStage(
    model_name="nvidia/stt_hy_fastconformer_hybrid_large_pc"
))

# 3. Calculate quality metrics
pipeline.add_stage(GetPairwiseWerStage())
pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# 4. Apply quality filters
pipeline.add_stage(PreserveByValueStage(
    input_value_key="wer",
    target_value=75.0,
    operator="le"  # Keep WER <= 75%
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"  # Keep duration >= 1s
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration", 
    target_value=30.0,
    operator="le"  # Keep duration <= 30s
))

# 5. Export high-quality results
pipeline.add_stage(AudioToDocumentStage())
pipeline.add_stage(JsonlWriter(path="./high_quality_audio"))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)
```

## Related Topics

- **[WER Filtering](wer-filtering.md)** - Detailed guide to Word Error Rate filtering
- **[Duration Filtering](duration-filtering.md)** - Audio length and speech rate filtering
- **[Audio Analysis](../audio-analysis/index.md)** - Audio file analysis and validation

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

WER Filtering <wer-filtering.md>
Duration Filtering <duration-filtering.md>
```
