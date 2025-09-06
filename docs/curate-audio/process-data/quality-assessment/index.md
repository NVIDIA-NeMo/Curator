---
description: "Assess and filter audio quality using transcription accuracy metrics, duration analysis, and custom quality measures"
categories: ["audio-processing"]
tags: ["quality-assessment", "wer-filtering", "duration-filtering", "quality-metrics", "speech-quality"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "audio-only"
---

# Quality Assessment for Audio Data

Evaluate and filter audio quality using transcription accuracy metrics, duration analysis, and custom quality measures to ensure high-quality speech datasets for ASR training.

## How it Works

Audio quality assessment in NeMo Curator focuses on speech-specific metrics that correlate with training data quality:

1. **Transcription Accuracy**: Word Error Rate (WER) and Character Error Rate (CER) between ground truth and ASR predictions
2. **Duration Analysis**: Audio length validation and speech rate calculations  
3. **Custom Metrics**: Domain-specific quality measures for specialized datasets
4. **Value-based Filtering**: Configurable filtering using comparison operators

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

Filter by audio length to remove very short or long samples:

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

The `PreserveByValueStage` supports multiple comparison operators:

| Operator | Description | Example Use Case |
|----------|-------------|------------------|
| `"eq"` | Equal to | Exact duration matching |
| `"ne"` | Not equal to | Exclude specific values |
| `"lt"` | Less than | Maximum thresholds |
| `"le"` | Less than or equal | Quality thresholds |
| `"gt"` | Greater than | Minimum thresholds |
| `"ge"` | Greater than or equal | Minimum requirements |

## Custom Quality Metrics

### Implementing Custom Filters

Create domain-specific quality assessments:

```python
from dataclasses import dataclass, field
from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

@dataclass
class CustomAudioQualityStage(LegacySpeechStage):
    """Custom quality assessment for domain-specific audio."""
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        # Calculate custom quality score
        text = data_entry.get("text", "")
        duration = data_entry.get("duration", 0)
        
        # Example: Penalize very fast or slow speech
        word_count = len(text.split())
        speech_rate = word_count / duration if duration > 0 else 0
        
        # Ideal speech rate: 2-4 words per second
        if 2.0 <= speech_rate <= 4.0:
            quality_score = 100.0
        else:
            quality_score = max(0, 100 - abs(speech_rate - 3.0) * 20)
        
        data_entry["speech_rate_quality"] = quality_score
        return [AudioBatch(data=data_entry)]
```

### Language-specific Quality

```python
from dataclasses import dataclass, field
from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

@dataclass  
class LanguageQualityStage(LegacySpeechStage):
    """Language-specific quality assessment."""
    
    language_thresholds: dict = field(default_factory=lambda: {
        "en_us": {"max_wer": 30.0, "min_duration": 0.5},
        "hy_am": {"max_wer": 50.0, "min_duration": 1.0},  # More lenient for low-resource languages
        "zh_cn": {"max_wer": 40.0, "min_duration": 0.8},
    })
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        language = data_entry.get("language", "unknown")
        thresholds = self.language_thresholds.get(language, {"max_wer": 50.0, "min_duration": 1.0})
        
        wer = data_entry.get("wer", 100.0)
        duration = data_entry.get("duration", 0.0)
        
        # Apply language-specific quality criteria
        is_quality = (
            wer <= thresholds["max_wer"] and 
            duration >= thresholds["min_duration"]
        )
        
        data_entry["language_quality_pass"] = is_quality
        return [AudioBatch(data=data_entry)]
```

## Performance Optimization

### Batch Processing

```python
# Optimize quality assessment for large datasets
wer_stage = GetPairwiseWerStage().with_(batch_size=100)
duration_stage = GetAudioDurationStage().with_(batch_size=50)
```

Parallelism is managed by the pipeline executor. Configure resources per stage using `.with_(resources=...)` and run with your chosen executor.

## Quality Analysis

### Dataset Quality Reports

Generate comprehensive quality reports after processing:

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_audio_quality(manifest_path: str):
    """Analyze quality distribution of processed audio dataset."""
    
    # Load processed data
    df = pd.read_json(manifest_path, lines=True)
    
    # Quality statistics
    print("Audio Dataset Quality Report")
    print("=" * 40)
    print(f"Total samples: {len(df)}")
    print(f"Average WER: {df['wer'].mean():.2f}%")
    print(f"WER std dev: {df['wer'].std():.2f}%")
    print(f"Average duration: {df['duration'].mean():.2f}s")
    
    # Quality distribution
    quality_bins = pd.cut(df['wer'], bins=[0, 10, 25, 50, 75, 100], 
                         labels=['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'])
    print(f"\nQuality Distribution:")
    print(quality_bins.value_counts())
    
    # Duration analysis
    print(f"\nDuration Statistics:")
    print(df['duration'].describe())
```

## Related Topics

- **[WER Filtering](wer-filtering.md)** - Detailed guide to Word Error Rate filtering
- **[Duration Filtering](duration-filtering.md)** - Audio length and speech rate filtering
- **[Custom Metrics](custom-metrics.md)** - Implementing domain-specific quality measures
- **[Audio Analysis](../audio-analysis/index.md)** - Audio file analysis and validation

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

WER Filtering <wer-filtering.md>
Duration Filtering <duration-filtering.md>
Custom Metrics <custom-metrics.md>
```

