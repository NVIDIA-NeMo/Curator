---
description: "Filter audio samples based on Word Error Rate thresholds to ensure high-quality transcription accuracy"
categories: ["audio-processing"]
tags: ["wer-filtering", "quality-filtering", "transcription-accuracy", "threshold-based", "speech-quality"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---

# WER Filtering

Filter audio samples based on Word Error Rate (WER) thresholds to ensure high-quality transcription accuracy in your speech datasets. WER filtering is the primary quality control mechanism for ASR-based audio curation.

## Understanding WER

### What is Word Error Rate?

Word Error Rate (WER) measures transcription accuracy by calculating the percentage of words that differ between ground truth and ASR predictions:

```text
WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words × 100
```

**Components:**

- **Substitutions**: Words incorrectly replaced (for example, "cat" → "hat")
- **Deletions**: Words omitted from the prediction
- **Insertions**: Extra words added to the prediction
- **Total_Reference_Words**: Total word count in ground truth transcription

A lower WER indicates higher transcription accuracy.

### WER Quality Levels

The following table provides general guidelines for interpreting WER values. Adjust thresholds based on your specific domain requirements and use case:

| WER Range | Quality Level | Recommended Use |
|-----------|---------------|-----------------|
| 0-10% | Excellent | Production ASR training, high-quality datasets |
| 10-25% | Good | General ASR training, most applications |
| 25-50% | Moderate | Supplementary training data, domain adaptation |
| 50-75% | Poor | Review required, potential filtering |
| 75%+ | Poor | Strong candidate for removal |

## Basic WER Filtering

Follow these steps to calculate WER values and apply threshold-based filtering to your audio dataset:

### Step 1: Calculate WER

Use `GetPairwiseWerStage` to compute WER between ground truth transcriptions and ASR model predictions:

```python
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage

# Calculate WER for audio samples
wer_stage = GetPairwiseWerStage(
    text_key="text",           # Ground truth transcription
    pred_text_key="pred_text", # ASR model prediction
    wer_key="wer"             # Output field for WER value
)

# Add to pipeline
pipeline.add_stage(wer_stage)
```

**Parameters:**

- `text_key`: Field name containing ground truth transcriptions in your manifest
- `pred_text_key`: Field name containing ASR predictions (from `InferenceAsrNemoStage` or similar)
- `wer_key`: Field name to store calculated WER values (default: `"wer"`)

**Prerequisites:** Your audio manifest must contain both ground truth transcriptions and ASR predictions before calculating WER.

### Step 2: Apply WER Threshold

Use `PreserveByValueStage` to filter audio samples based on the calculated WER values:

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Keep samples with WER ≤ 30% (good quality)
wer_filter = PreserveByValueStage(
    input_value_key="wer",
    target_value=30.0,
    operator="le"  # less than or equal
)

pipeline.add_stage(wer_filter)
```

**Parameters:**

- `input_value_key`: Field containing WER values (matches `wer_key` from previous stage)
- `target_value`: WER threshold (percentage as float, e.g., `30.0` for 30%)
- `operator`: Comparison operator (`"le"` for ≤, `"lt"` for <, `"ge"` for ≥, `"gt"` for >)

The stage preserves samples meeting the threshold criteria and filters out others.

## Advanced WER Filtering

### Statistical WER Filtering

Rather than using fixed thresholds, you can analyze your dataset's WER distribution to determine optimal filtering thresholds. This approach is useful when working with domain-specific data or evaluating data quality.

**Workflow:**

1. Calculate WER for all samples using `GetPairwiseWerStage`
2. Export results and analyze WER distribution (mean, median, percentiles)
3. Determine threshold based on your quality requirements (for example, keep samples below 75th percentile)
4. Apply the calculated threshold using `PreserveByValueStage`

**Example:**

```python
# Apply calculated statistical threshold
statistical_filter = PreserveByValueStage(
    input_value_key="wer",
    target_value=calculated_threshold,  # From your statistical analysis
    operator="le"
)

pipeline.add_stage(statistical_filter)
```

:::{tip}
Use `AudioToDocumentStage` and `JsonlWriter` to export WER values for analysis in tools like pandas, numpy, or visualization libraries.
:::

## Domain-Specific WER Filtering

Different speech domains have varying acoustic characteristics and transcription complexity. Adjust WER thresholds based on your specific domain:

### Conversational Speech

Conversational speech typically has higher WER due to informal language, disfluencies, overlapping speech, and background noise. Use more lenient thresholds:

```python
# More lenient thresholds for conversational speech
conversational_wer_config = {
    "excellent_threshold": 15.0,  # compared to 10.0 for read speech
    "good_threshold": 35.0,       # compared to 25.0 for read speech  
    "acceptable_threshold": 60.0   # compared to 50.0 for read speech
}

conversational_filter = PreserveByValueStage(
    input_value_key="wer",
    target_value=conversational_wer_config["good_threshold"],
    operator="le"
)

pipeline.add_stage(conversational_filter)
```

**Use cases:** Call center recordings, meeting transcriptions, casual interviews, social media audio

### Broadcast and News

Broadcast speech features professional speakers, controlled environments, and clear articulation, enabling stricter quality standards:

```python
# Stricter thresholds for high-quality broadcast speech
broadcast_wer_config = {
    "excellent_threshold": 5.0,   # Very strict
    "good_threshold": 15.0,       # Stricter than general
    "acceptable_threshold": 25.0   # Maximum for broadcast quality
}

broadcast_filter = PreserveByValueStage(
    input_value_key="wer", 
    target_value=broadcast_wer_config["good_threshold"],
    operator="le"
)

pipeline.add_stage(broadcast_filter)
```

**Use cases:** News broadcasts, audiobooks, podcasts, prepared presentations, voiceovers

## Complete WER Filtering Example

Here's a complete pipeline demonstrating WER calculation and filtering:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources

# Create WER filtering pipeline
pipeline = Pipeline(name="wer_filtering")

# 1. Load audio data with ground truth transcriptions
pipeline.add_stage(CreateInitialManifestFleursStage(
    lang="en_us", 
    split="validation", 
    raw_data_dir="./audio_data"
).with_(batch_size=8))

# 2. Run ASR inference to generate predictions
pipeline.add_stage(InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
    pred_text_key="pred_text"
).with_(resources=Resources(gpus=1.0)))

# 3. Calculate WER
pipeline.add_stage(GetPairwiseWerStage(
    text_key="text",
    pred_text_key="pred_text",
    wer_key="wer"
))

# 4. Filter by WER threshold (keep WER ≤ 30%)
pipeline.add_stage(PreserveByValueStage(
    input_value_key="wer",
    target_value=30.0,
    operator="le"
))

# 5. Export filtered results
pipeline.add_stage(AudioToDocumentStage())
pipeline.add_stage(JsonlWriter(path="./filtered_audio"))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)
```

## Best Practices

- **Start with lenient thresholds**: Begin with higher WER thresholds (for example, 50%) and progressively tighten based on dataset size and quality requirements.
- **Consider domain characteristics**: Adjust thresholds based on speech type (conversational compared to broadcast compared to read speech).
- **Analyze before filtering**: Export WER distributions to understand your data before applying aggressive filters.
- **Balance quality and quantity**: Stricter thresholds improve data quality but reduce dataset size; find the right balance for your use case.
- **Check ASR model**: Ensure your ASR model is appropriate for the language and domain before using WER for filtering.

## Related Topics

- **[Quality Assessment Overview](index.md)** - Complete guide to audio quality assessment
- **[Duration Filtering](duration-filtering.md)** - Filter by audio length and speech rate
- **[ASR Inference](../asr-inference/index.md)** - Generate ASR predictions for WER calculation
