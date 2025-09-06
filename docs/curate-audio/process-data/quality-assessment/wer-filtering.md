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

Word Error Rate measures transcription accuracy by calculating the percentage of words that differ between ground truth and ASR predictions:

```
WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words × 100
```

### WER Quality Levels

| WER Range | Quality Level | Recommended Use |
|-----------|---------------|-----------------|
| 0-10% | Excellent | Production ASR training, high-quality datasets |
| 10-25% | Good | General ASR training, most applications |
| 25-50% | Moderate | Supplementary training data, domain adaptation |
| 50-75% | Poor | Review required, potential filtering |
| 75%+ | Very Poor | Strong candidate for removal |

## Basic WER Filtering

### Calculate WER

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

### Apply WER Threshold

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

## Advanced WER Filtering

### Multi-Tier Filtering

```python
def create_multi_tier_wer_pipeline() -> Pipeline:
    """Create pipeline with multiple WER quality tiers."""
    
    pipeline = Pipeline(name="multi_tier_wer")
    
    # Calculate WER
    pipeline.add_stage(GetPairwiseWerStage())
    
    # Tier 1: Excellent quality (WER ≤ 10%)
    @processing_stage(name="tier1_filter")
    def excellent_quality_filter(audio_batch: AudioBatch) -> AudioBatch:
        excellent_data = [
            item for item in audio_batch.data 
            if item.get("wer", 100) <= 10.0
        ]
        
        # Mark as tier 1
        for item in excellent_data:
            item["quality_tier"] = "excellent"
        
        return AudioBatch(data=excellent_data, filepath_key=audio_batch.filepath_key)
    
    # Tier 2: Good quality (10% < WER ≤ 25%)  
    @processing_stage(name="tier2_filter")
    def good_quality_filter(audio_batch: AudioBatch) -> AudioBatch:
        good_data = [
            item for item in audio_batch.data
            if 10.0 < item.get("wer", 100) <= 25.0
        ]
        
        for item in good_data:
            item["quality_tier"] = "good"
        
        return AudioBatch(data=good_data, filepath_key=audio_batch.filepath_key)
    
    # Add both tier filters to the pipeline
    pipeline.add_stage(excellent_quality_filter)
    pipeline.add_stage(good_quality_filter)
    
    return pipeline
```

### Adaptive WER Thresholds

```python
@dataclass
class AdaptiveWerFilterStage(LegacySpeechStage):
    """Apply WER filtering with adaptive thresholds based on dataset characteristics."""
    
    base_threshold: float = 30.0
    language_adjustments: dict = field(default_factory=lambda: {
        "en": 0.0,      # No adjustment for English
        "es": 5.0,      # Slightly more lenient for Spanish  
        "hy": 15.0,     # More lenient for Armenian (low-resource)
        "zh": 10.0      # Moderate adjustment for Chinese
    })
    domain_adjustments: dict = field(default_factory=lambda: {
        "telephony": 10.0,     # More lenient for telephony
        "broadcast": -5.0,     # Stricter for broadcast
        "conversational": 5.0   # Slightly more lenient for conversation
    })
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        wer = data_entry.get("wer", 100.0)
        language = data_entry.get("language", "en")
        domain = data_entry.get("domain", "general")
        
        # Calculate adaptive threshold
        threshold = self.base_threshold
        threshold += self.language_adjustments.get(language, 0)
        threshold += self.domain_adjustments.get(domain, 0)
        
        # Apply filter
        if wer <= threshold:
            data_entry["wer_threshold_used"] = threshold
            data_entry["wer_filter_passed"] = True
            return [AudioBatch(data=data_entry)]
        else:
            return []  # Filter out
```

### Statistical WER Filtering

```python
import numpy as np

def create_statistical_wer_filter(audio_data: list[dict], percentile: float = 75) -> float:
    """Calculate WER threshold based on dataset statistics."""
    
    wer_values = [item["wer"] for item in audio_data if "wer" in item]
    
    if not wer_values:
        return 50.0  # Default threshold
    
    # Use percentile-based threshold
    threshold = np.percentile(wer_values, percentile)
    
    # Analysis
    stats = {
        "mean_wer": np.mean(wer_values),
        "median_wer": np.median(wer_values),
        "std_wer": np.std(wer_values),
        "suggested_threshold": threshold,
        "samples_kept": sum(1 for wer in wer_values if wer <= threshold),
        "samples_total": len(wer_values),
        "retention_rate": sum(1 for wer in wer_values if wer <= threshold) / len(wer_values)
    }
    
    print(f"Statistical WER Analysis:")
    print(f"  Mean WER: {stats['mean_wer']:.2f}%")
    print(f"  Suggested threshold (P{percentile}): {threshold:.2f}%")
    print(f"  Retention rate: {stats['retention_rate']:.1%}")
    
    return threshold

# Usage
threshold = create_statistical_wer_filter(processed_audio_data, percentile=80)

# Apply statistical threshold
statistical_filter = PreserveByValueStage(
    input_value_key="wer",
    target_value=threshold,
    operator="le"
)
```

## Domain-Specific WER Filtering

### Conversational Speech

```python
# More lenient thresholds for conversational speech
conversational_wer_config = {
    "excellent_threshold": 15.0,  # vs. 10.0 for read speech
    "good_threshold": 35.0,       # vs. 25.0 for read speech  
    "acceptable_threshold": 60.0   # vs. 50.0 for read speech
}

conversational_filter = PreserveByValueStage(
    input_value_key="wer",
    target_value=conversational_wer_config["good_threshold"],
    operator="le"
)
```

### Broadcast and News

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
```

### Multilingual Considerations

```python
def create_language_aware_wer_filter(language_thresholds: dict) -> Pipeline:
    """Create WER filter with language-specific thresholds."""
    
    pipeline = Pipeline(name="language_aware_wer")
    
    @processing_stage(name="language_wer_filter")
    def language_specific_filter(audio_batch: AudioBatch) -> AudioBatch:
        filtered_data = []
        
        for item in audio_batch.data:
            language = item.get("language", "unknown")
            wer = item.get("wer", 100.0)
            
            # Get language-specific threshold
            threshold = language_thresholds.get(language, 50.0)  # Default threshold
            
            if wer <= threshold:
                item["language_wer_threshold"] = threshold
                filtered_data.append(item)
        
        return AudioBatch(data=filtered_data, filepath_key=audio_batch.filepath_key)
    
    pipeline.add_stage(language_specific_filter)
    return pipeline

# Language-specific thresholds
language_thresholds = {
    "en": 20.0,    # Strict for high-resource English
    "es": 25.0,    # Moderate for Spanish
    "fr": 25.0,    # Moderate for French
    "de": 30.0,    # Slightly lenient for German
    "hy": 45.0,    # Very lenient for low-resource Armenian
    "zh": 35.0     # Moderate-lenient for Chinese
}

language_filter = create_language_aware_wer_filter(language_thresholds)
```

## WER Analysis and Insights

### WER Distribution Analysis

```python
import numpy as np

def analyze_wer_distribution(audio_data: list[dict]) -> dict:
    """Analyze WER distribution to inform filtering decisions."""
    
    wer_values = [item["wer"] for item in audio_data if "wer" in item]
    
    analysis = {
        "basic_stats": {
            "count": len(wer_values),
            "mean": np.mean(wer_values),
            "median": np.median(wer_values),
            "std": np.std(wer_values),
            "min": min(wer_values),
            "max": max(wer_values)
        },
        
        "percentiles": {
            f"p{p}": np.percentile(wer_values, p)
            for p in [10, 25, 50, 75, 90, 95, 99]
        },
        
        "quality_distribution": {
            "excellent": sum(1 for wer in wer_values if wer <= 10),
            "good": sum(1 for wer in wer_values if 10 < wer <= 25),
            "fair": sum(1 for wer in wer_values if 25 < wer <= 50),
            "poor": sum(1 for wer in wer_values if 50 < wer <= 75),
            "very_poor": sum(1 for wer in wer_values if wer > 75)
        },
        
        "threshold_impact": {}
    }
    
    # Calculate retention rates for different thresholds
    for threshold in [10, 15, 20, 25, 30, 40, 50]:
        retained = sum(1 for wer in wer_values if wer <= threshold)
        retention_rate = retained / len(wer_values)
        analysis["threshold_impact"][f"wer_{threshold}"] = {
            "threshold": threshold,
            "samples_retained": retained,
            "retention_rate": retention_rate
        }
    
    return analysis

# Generate recommendations
def recommend_wer_threshold(analysis: dict, target_retention: float = 0.8) -> float:
    """Recommend WER threshold based on desired retention rate."""
    
    threshold_impact = analysis["threshold_impact"]
    
    # Find threshold that achieves target retention
    for threshold_key in sorted(threshold_impact.keys()):
        threshold_data = threshold_impact[threshold_key]
        
        if threshold_data["retention_rate"] >= target_retention:
            return threshold_data["threshold"]
    
    # If no threshold achieves target retention, return highest
    return max(threshold_impact[key]["threshold"] for key in threshold_impact)
```

### Error Pattern Analysis

```python
import numpy as np

def analyze_wer_error_patterns(audio_data: list[dict]) -> dict:
    """Analyze common error patterns in high-WER samples."""
    
    high_wer_samples = [
        item for item in audio_data 
        if item.get("wer", 0) > 50.0
    ]
    
    error_patterns = {
        "common_characteristics": {},
        "duration_correlation": {},
        "text_length_correlation": {},
        "recommendations": []
    }
    
    if not high_wer_samples:
        return error_patterns
    
    # Analyze duration patterns
    durations = [item.get("duration", 0) for item in high_wer_samples]
    error_patterns["duration_correlation"] = {
        "mean_duration": np.mean(durations),
        "very_short": sum(1 for d in durations if d < 1.0),
        "very_long": sum(1 for d in durations if d > 30.0)
    }
    
    # Analyze text characteristics
    texts = [item.get("text", "") for item in high_wer_samples]
    text_lengths = [len(text.split()) for text in texts]
    
    error_patterns["text_length_correlation"] = {
        "mean_word_count": np.mean(text_lengths),
        "very_short_text": sum(1 for length in text_lengths if length < 3),
        "very_long_text": sum(1 for length in text_lengths if length > 50)
    }
    
    # Generate recommendations
    recommendations = []
    
    if error_patterns["duration_correlation"]["very_short"] > len(high_wer_samples) * 0.3:
        recommendations.append("Consider filtering very short audio files (< 1s)")
    
    if error_patterns["duration_correlation"]["very_long"] > len(high_wer_samples) * 0.2:
        recommendations.append("Consider segmenting long audio files (> 30s)")
    
    if error_patterns["text_length_correlation"]["very_short_text"] > len(high_wer_samples) * 0.4:
        recommendations.append("High WER correlated with short transcriptions - check data quality")
    
    error_patterns["recommendations"] = recommendations
    
    return error_patterns
```

## Filtering Strategies

### Progressive Filtering

```python
def create_progressive_wer_pipeline() -> Pipeline:
    """Create pipeline with progressive WER filtering stages."""
    
    pipeline = Pipeline(name="progressive_wer")
    
    # Calculate WER
    pipeline.add_stage(GetPairwiseWerStage())
    
    # Stage 1: Remove very poor quality (WER > 75%)
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="wer",
            target_value=75.0,
            operator="lt"  # strictly less than
        )
    )
    
    # Stage 2: Mark quality tiers
    @processing_stage(name="quality_tier_marking")
    def mark_quality_tiers(audio_batch: AudioBatch) -> AudioBatch:
        for item in audio_batch.data:
            wer = item["wer"]
            
            if wer <= 10:
                item["quality_tier"] = "tier1_excellent"
            elif wer <= 25:
                item["quality_tier"] = "tier2_good"
            elif wer <= 50:
                item["quality_tier"] = "tier3_moderate"
            else:
                item["quality_tier"] = "tier4_poor"
        
        return audio_batch
    
    pipeline.add_stage(mark_quality_tiers)
    
    # Stage 3: Apply final threshold based on data availability
    @processing_stage(name="adaptive_final_filter")
    def adaptive_final_filter(audio_batch: AudioBatch) -> AudioBatch:
        # Count samples in each tier
        tier_counts = {}
        for item in audio_batch.data:
            tier = item["quality_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Adaptive threshold based on data availability
        total_samples = len(audio_batch.data)
        tier1_ratio = tier_counts.get("tier1_excellent", 0) / total_samples
        tier2_ratio = tier_counts.get("tier2_good", 0) / total_samples
        
        if tier1_ratio >= 0.3:  # 30%+ excellent quality
            threshold = 10.0    # Use only excellent
        elif tier1_ratio + tier2_ratio >= 0.6:  # 60%+ good quality  
            threshold = 25.0    # Use excellent + good
        else:
            threshold = 50.0    # Include moderate quality
        
        # Apply adaptive threshold
        filtered_data = [
            item for item in audio_batch.data
            if item["wer"] <= threshold
        ]
        
        for item in filtered_data:
            item["adaptive_threshold_used"] = threshold
        
        return AudioBatch(data=filtered_data, filepath_key=audio_batch.filepath_key)
    
    pipeline.add_stage(adaptive_final_filter)
    
    return pipeline
```

### Context-Aware Filtering

```python
@dataclass
class ContextAwareWerFilterStage(LegacySpeechStage):
    """Filter based on WER with context awareness."""
    
    base_threshold: float = 30.0
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        wer = data_entry.get("wer", 100.0)
        text = data_entry.get("text", "")
        pred_text = data_entry.get("pred_text", "")
        duration = data_entry.get("duration", 0)
        
        # Context-aware threshold adjustment
        threshold = self.base_threshold
        
        # Adjust for text complexity
        word_count = len(text.split())
        if word_count < 3:
            threshold -= 10.0  # Stricter for short utterances
        elif word_count > 20:
            threshold += 5.0   # More lenient for long utterances
        
        # Adjust for duration
        if duration < 1.0:
            threshold -= 5.0   # Stricter for very short audio
        elif duration > 15.0:
            threshold += 5.0   # More lenient for long audio
        
        # Adjust for prediction length consistency
        pred_word_count = len(pred_text.split())
        length_ratio = pred_word_count / word_count if word_count > 0 else 0
        
        if length_ratio < 0.5 or length_ratio > 2.0:
            threshold -= 10.0  # Stricter for length mismatches
        
        # Apply adjusted threshold
        if wer <= threshold:
            data_entry["context_threshold_used"] = threshold
            data_entry["context_adjustments"] = {
                "word_count_adj": word_count,
                "duration_adj": duration,
                "length_ratio_adj": length_ratio
            }
            return [AudioBatch(data=data_entry)]
        else:
            return []
```

## WER Filtering Best Practices

### Threshold Selection Guidelines

1. **Start Strict**: Begin with conservative thresholds (WER ≤ 20%)
2. **Analyze Distribution**: Examine WER distribution of your dataset
3. **Consider Use Case**: Adjust based on training vs. evaluation needs
4. **Language Adaptation**: Use language-specific thresholds
5. **Iterative Refinement**: Adjust based on downstream model performance

### Quality vs. Quantity Trade-offs

```python
import numpy as np
import pandas as pd

def evaluate_filtering_trade_offs(audio_data: list[dict], 
                                thresholds: list[float]) -> pd.DataFrame:
    """Evaluate quality vs. quantity trade-offs for different WER thresholds."""
    
    results = []
    
    for threshold in thresholds:
        filtered_data = [
            item for item in audio_data 
            if item.get("wer", 100) <= threshold
        ]
        
        if filtered_data:
            wer_values = [item["wer"] for item in filtered_data]
            durations = [item.get("duration", 0) for item in filtered_data]
            
            result = {
                "threshold": threshold,
                "samples_retained": len(filtered_data),
                "retention_rate": len(filtered_data) / len(audio_data),
                "mean_wer": np.mean(wer_values),
                "total_hours": sum(durations) / 3600,
                "quality_score": (100 - np.mean(wer_values)) * (len(filtered_data) / len(audio_data))
            }
        else:
            result = {
                "threshold": threshold,
                "samples_retained": 0,
                "retention_rate": 0.0,
                "mean_wer": 0.0,
                "total_hours": 0.0,
                "quality_score": 0.0
            }
        
        results.append(result)
    
    return pd.DataFrame(results)

# Usage
trade_off_analysis = evaluate_filtering_trade_offs(
    audio_data=processed_samples,
    thresholds=[10, 15, 20, 25, 30, 40, 50]
)

print(trade_off_analysis)
```

### Validation Strategies

```python
import numpy as np

def validate_wer_filtering_effectiveness(original_data: list[dict], 
                                       filtered_data: list[dict]) -> dict:
    """Validate effectiveness of WER filtering."""
    
    original_wers = [item["wer"] for item in original_data]
    filtered_wers = [item["wer"] for item in filtered_data]
    
    validation_metrics = {
        "filtering_summary": {
            "original_count": len(original_data),
            "filtered_count": len(filtered_data),
            "retention_rate": len(filtered_data) / len(original_data)
        },
        
        "quality_improvement": {
            "original_mean_wer": np.mean(original_wers),
            "filtered_mean_wer": np.mean(filtered_wers),
            "wer_improvement": np.mean(original_wers) - np.mean(filtered_wers),
            "std_reduction": np.std(original_wers) - np.std(filtered_wers)
        },
        
        "distribution_shift": {
            "original_p95": np.percentile(original_wers, 95),
            "filtered_p95": np.percentile(filtered_wers, 95),
            "excellent_ratio_original": sum(1 for wer in original_wers if wer <= 10) / len(original_wers),
            "excellent_ratio_filtered": sum(1 for wer in filtered_wers if wer <= 10) / len(filtered_wers)
        }
    }
    
    return validation_metrics
```

## Troubleshooting

### Common WER Filtering Issues

**Too Aggressive Filtering**: Very high WER threshold rejection
```python
import numpy as np

# Diagnose: Check WER distribution
wer_dist = [item["wer"] for item in audio_data]
print(f"WER distribution: P50={np.percentile(wer_dist, 50):.1f}, P75={np.percentile(wer_dist, 75):.1f}")

# Solution: Relax threshold or check ASR model selection
```

**Inconsistent Quality**: High variance in WER values
```python
# Diagnose: Analyze error patterns
error_analysis = analyze_wer_error_patterns(audio_data)

# Solution: Add duration filtering, improve audio preprocessing
```

**Language-Specific Issues**: Poor performance for certain languages
```python
import numpy as np

# Diagnose: Check language-specific WER distribution
for lang in languages:
    lang_data = [item for item in audio_data if item.get("language") == lang]
    lang_wers = [item["wer"] for item in lang_data]
    print(f"{lang}: mean WER = {np.mean(lang_wers):.2f}%")

# Solution: Use language-specific models and thresholds
```

## Related Topics

- **[Quality Assessment Overview](index.md)** - Complete quality assessment workflow
- **[Duration Filtering](duration-filtering.md)** - Audio length-based filtering
- **[Custom Metrics](custom-metrics.md)** - Additional quality measures
- **[ASR Models](../asr-inference/nemo-models.md)** - Model selection for better WER
