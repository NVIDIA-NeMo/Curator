---
description: "Core concepts for evaluating speech transcription quality using WER, CER, duration analysis, and speech rate metrics"
categories: ["concepts-architecture"]
tags: ["quality-metrics", "wer", "cer", "speech-quality", "transcription-accuracy"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "audio-only"
---

(about-concepts-audio-quality-metrics)=
# Audio Quality Metrics

This guide covers the quality metrics used in NeMo Curator for evaluating speech transcription accuracy, audio characteristics, and overall dataset quality.

## Transcription Accuracy Metrics

### Word Error Rate (WER)

The primary metric for measuring ASR transcription quality:

**Definition**: Percentage of words that differ between ground truth and predicted transcriptions.

**Calculation**: 
```
WER = (Substitutions + Deletions + Insertions) / Total_Words × 100
```

**Interpretation**:
- **WER = 0%**: Perfect transcription match
- **WER ≤ 10%**: Excellent quality (production-ready)
- **WER ≤ 25%**: Good quality (suitable for most training)
- **WER ≤ 50%**: Moderate quality (may need review)
- **WER > 75%**: Poor quality (consider filtering)

**Example**:
```python
# Ground truth: "hello world example"
# Prediction:   "hello word example"
# WER = 1/3 × 100 = 33.33% (1 substitution out of 3 words)
```

```{note}
WER and CER utilities depend on the `editdistance` package.
```

### Character Error Rate (CER)

More granular accuracy measurement at the character level:

**Definition**: Percentage of characters that differ between ground truth and predicted transcriptions.

**Calculation**:
```
CER = (Character_Substitutions + Character_Deletions + Character_Insertions) / Total_Characters × 100
```

**Use Cases**:
- Languages with complex morphology
- Detailed accuracy analysis
- Character-level model evaluation

**Example**:
```python
# Ground truth: "hello"
# Prediction:   "helo" 
# CER = 1/5 × 100 = 20% (1 deletion out of 5 characters)
```

## Audio Characteristic Metrics

### Duration Analysis

**Audio Duration**: Precise measurement of audio file length in seconds.

**Speech Rate Metrics**:
- **Words per Second**: `word_count / duration`
- **Characters per Second**: `character_count / duration`

```{note}
To enforce duration thresholds in a pipeline, use `PreserveByValueStage`.
```

### Format and Technical Metrics

**Sample Rate**: Audio sampling frequency (typically 16 kHz for ASR)
**Bit Depth**: Audio resolution (16-bit or 24-bit)
**Channels**: Mono (preferred) or stereo audio
**Encoding format**: Compression format (WAV, FLAC preferred for quality)

## Quality Assessment Strategies

### Threshold-Based Filtering

**Conservative Filtering** (High Quality):
```python
quality_thresholds = {
    "max_wer": 15.0,        # WER ≤ 15%
    "min_duration": 1.0,     # Duration ≥ 1 second
    "max_duration": 20.0,    # Duration ≤ 20 seconds
    "min_words": 3,          # At least 3 words
}
```

**Balanced Filtering** (Good Quality):
```python
quality_thresholds = {
    "max_wer": 30.0,        # WER ≤ 30%
    "min_duration": 0.5,     # Duration ≥ 0.5 seconds
    "max_duration": 30.0,    # Duration ≤ 30 seconds
    "min_words": 2,          # At least 2 words
}
```

**Lenient Filtering** (Acceptable Quality):
```python
quality_thresholds = {
    "max_wer": 50.0,        # WER ≤ 50%
    "min_duration": 0.3,     # Duration ≥ 0.3 seconds
    "max_duration": 60.0,    # Duration ≤ 60 seconds
    "min_words": 1,          # At least 1 word
}
```

Filtering mechanism reference: `nemo_curator/stages/audio/common.py:71-116` (`PreserveByValueStage` supports `lt`, `le`, `eq`, `ne`, `ge`, `gt` over a value key)

### Language-Specific Considerations

Different languages require different quality thresholds:

**High-Resource Languages** (English, Spanish, French):
- Lower WER thresholds (≤ 20%)
- Standard duration ranges
- Extensive ASR model availability

**Medium-Resource Languages** (German, Italian, Portuguese):
- Moderate WER thresholds (≤ 30%) 
- Slightly more lenient filtering
- Good ASR model availability

**Low-Resource Languages** (Armenian, Estonian, Maltese):
- Higher WER thresholds (≤ 50%)
- More lenient duration filtering
- Limited ASR model options

## Composite Quality Scores

### Weighted Quality Scoring

Combine multiple metrics for overall quality assessment:

```python
def calculate_composite_quality(wer: float, duration: float, text: str) -> float:
    """Calculate composite quality score (0-100)."""
    
    # WER component (50% weight)
    wer_score = max(0, 100 - wer)
    
    # Duration component (30% weight) 
    if 1.0 <= duration <= 15.0:
        duration_score = 100
    elif 0.5 <= duration < 1.0 or 15.0 < duration <= 30.0:
        duration_score = 75
    else:
        duration_score = 25
    
    # Text length component (20% weight)
    word_count = len(text.split())
    if word_count >= 5:
        length_score = 100
    elif word_count >= 3:
        length_score = 75
    else:
        length_score = 50
    
    # Weighted combination
    composite_score = (
        0.5 * wer_score +
        0.3 * duration_score + 
        0.2 * length_score
    )
    
    return round(composite_score, 2)
```

```{note}
This function is an example-only snippet to illustrate a possible scoring approach. It is not a built-in utility. To use it in a pipeline, implement a custom stage that writes a `composite_quality` field. For end-to-end examples, refer to the custom metrics guidance.
```

### Domain-Specific Scoring

**Conversational Speech**:
- Emphasis on natural speech patterns
- Tolerance for pauses and filler words
- Speaker change detection importance

**Broadcast Speech**:
- High accuracy requirements
- Clear pronunciation expectations
- Background noise considerations

**Telephony Speech**:
- Bandwidth limitations consideration
- Compression artifact tolerance
- Channel-specific quality factors

## Quality Monitoring

### Dataset Quality Distribution

Monitor quality across your dataset:

```python
def analyze_quality_distribution(manifest_data: list) -> dict:
    """Analyze quality distribution across dataset."""
    
    wer_values = [item["wer"] for item in manifest_data]
    duration_values = [item["duration"] for item in manifest_data]
    
    return {
        "total_samples": len(manifest_data),
        "wer_stats": {
            "mean": np.mean(wer_values),
            "median": np.median(wer_values), 
            "std": np.std(wer_values),
            "percentiles": np.percentile(wer_values, [25, 50, 75, 90, 95])
        },
        "duration_stats": {
            "mean": np.mean(duration_values),
            "median": np.median(duration_values),
            "total_hours": sum(duration_values) / 3600
        },
        "quality_bins": {
            "excellent": sum(1 for wer in wer_values if wer <= 10),
            "good": sum(1 for wer in wer_values if 10 < wer <= 25),
            "fair": sum(1 for wer in wer_values if 25 < wer <= 50),
            "poor": sum(1 for wer in wer_values if wer > 50)
        }
    }
```

```{note}
This distribution function is a documentation example, not part of the shipped API. It requires `numpy` (such as `import numpy as np`). Consider integrating it in analysis notebooks or a custom stage.
```

## Best Practices

### Quality Threshold Selection

1. **Start Conservative**: Begin with strict thresholds (WER ≤ 20%)
2. **Analyze Distribution**: Examine quality distribution of your dataset
3. **Adjust Iteratively**: Relax thresholds based on data availability
4. **Domain Adaptation**: Customize thresholds for your specific use case

### Metric Combination

1. **Primary Metric**: Use WER as the main quality indicator
2. **Secondary Filters**: Apply duration and text length filters
3. **Value-based Filtering**: Apply configurable threshold filtering
4. **Validation**: Cross-validate quality with human evaluation

### Quality-Performance Trade-offs

**High Quality (Strict Filtering)**:
- Pros: Better model training, higher accuracy
- Cons: Reduced dataset size, potential bias

**Balanced Quality (Moderate Filtering)**:
- Pros: Good quality with reasonable dataset size
- Cons: Some noise in training data

**High Coverage (Lenient Filtering)**:
- Pros: Maximum data utilization, diverse content
- Cons: Lower average quality, potential model degradation
