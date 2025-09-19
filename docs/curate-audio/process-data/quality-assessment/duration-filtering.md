---
description: "Filter audio samples by duration ranges, speech rate metrics, and temporal characteristics for optimal dataset quality"
categories: ["audio-processing"]
tags: ["duration-filtering", "speech-rate", "temporal-filtering", "audio-length", "quality-control"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---

# Duration Filtering

Filter audio samples by duration ranges, speech rate metrics, and temporal characteristics to create optimal datasets for ASR training and speech processing applications.

## Duration-Based Quality Control

### Why Duration Matters

**Training Efficiency**: Optimal duration ranges improve ASR training:
- **Too Short** (< 0.5s): Incomplete utterances, insufficient context
- **Too Long** (> 60s): Multiple speakers, topic changes, memory constraints
- **Optimal Range** (1-15s): Complete utterances, manageable memory usage

**Processing Performance**: Duration affects computational requirements:
- Memory usage scales with audio length
- Batch processing efficiency varies with duration variance
- GPU utilization optimized for consistent lengths

## Basic Duration Filtering

### Simple Duration Range

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage

# Calculate duration for each audio file
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)

# Filter for optimal duration range (1-15 seconds)
min_duration_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"  # greater than or equal
)

max_duration_filter = PreserveByValueStage(
    input_value_key="duration", 
    target_value=15.0,
    operator="le"  # less than or equal
)

# Add to pipeline
pipeline.add_stage(duration_stage)
pipeline.add_stage(min_duration_filter) 
pipeline.add_stage(max_duration_filter)
```

### Use Case-Specific Ranges

```python
# Duration ranges for different applications
duration_configs = {
    "asr_training": {
        "min_duration": 1.0,
        "max_duration": 20.0,
        "optimal_range": (2.0, 10.0)
    },
    
    "voice_cloning": {
        "min_duration": 3.0,
        "max_duration": 10.0, 
        "optimal_range": (4.0, 8.0)
    },
    
    "speech_synthesis": {
        "min_duration": 2.0,
        "max_duration": 15.0,
        "optimal_range": (3.0, 12.0)
    },
    
    "keyword_spotting": {
        "min_duration": 0.5,
        "max_duration": 3.0,
        "optimal_range": (1.0, 2.0)
    }
}

def create_use_case_duration_filter(use_case: str) -> list[PreserveByValueStage]:
    """Create duration filters for specific use case."""
    
    config = duration_configs.get(use_case, duration_configs["asr_training"])
    
    return [
        PreserveByValueStage(
            input_value_key="duration",
            target_value=config["min_duration"],
            operator="ge"
        ),
        PreserveByValueStage(
            input_value_key="duration",
            target_value=config["max_duration"],
            operator="le"
        )
    ]
```

## Speech Rate Analysis

### Calculate Speech Rate Metrics

Use the built-in speech rate calculation functions to analyze speaking speed:

```python
from nemo_curator.stages.audio.metrics.get_wer import get_wordrate, get_charrate

# Calculate speech rate metrics for a sample
text = "Hello world this is a test"
duration = 2.5  # seconds

word_rate = get_wordrate(text, duration)  # Returns words per second
char_rate = get_charrate(text, duration)  # Returns characters per second

print(f"Word rate: {word_rate} words/second")
print(f"Character rate: {char_rate} chars/second")
```

:::{note}
The speech rate functions `get_wordrate()` and `get_charrate()` are utility functions. To use them in a processing pipeline, you would need to create a custom stage that calls these functions and adds the results to your data.
:::

### Speech Rate Filtering

If you have pre-calculated speech rate metrics in your data, you can filter based on them:

```python
from nemo_curator.stages.audio.common import PreserveByValueStage
from nemo_curator.pipeline import Pipeline

# Example: Filter by speech rate if you have word_rate field in your data
pipeline = Pipeline(name="speech_rate_filtering")

# Filter by speech rate (1.5-5 words per second)
pipeline.add_stage(
    PreserveByValueStage(
        input_value_key="word_rate",  # Assumes this field exists in your data
        target_value=1.5,
        operator="ge"
    )
)

pipeline.add_stage(
    PreserveByValueStage(
        input_value_key="word_rate",
        target_value=5.0,
        operator="le"
    )
)
```

:::{note}
This example assumes you have already calculated and stored speech rate metrics in your audio data. The built-in stages do not automatically calculate speech rates - you would need to create a custom stage for that functionality.
:::

## Advanced Duration Filtering

### Statistical Duration Filtering

```python
def create_statistical_duration_filter(audio_data: list[dict], 
                                     outlier_threshold: float = 2.0) -> tuple[float, float]:
    """Create duration filter based on statistical analysis."""
    
    durations = [item["duration"] for item in audio_data if item.get("duration", 0) > 0]
    
    # Statistical analysis
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    
    # Remove statistical outliers
    min_duration = max(0.5, mean_duration - outlier_threshold * std_duration)
    max_duration = min(60.0, mean_duration + outlier_threshold * std_duration)
    
    # Analysis report
    outliers_removed = sum(
        1 for d in durations 
        if d < min_duration or d > max_duration
    )
    
    print(f"Duration Analysis:")
    print(f"  Mean: {mean_duration:.2f}s, Std: {std_duration:.2f}s")
    print(f"  Suggested range: {min_duration:.2f}s - {max_duration:.2f}s")
    print(f"  Outliers removed: {outliers_removed}/{len(durations)} ({outliers_removed/len(durations):.1%})")
    
    return min_duration, max_duration

# Apply statistical filtering
min_dur, max_dur = create_statistical_duration_filter(audio_dataset)

duration_filters = [
    PreserveByValueStage("duration", min_dur, "ge"),
    PreserveByValueStage("duration", max_dur, "le")
]
```

### Percentile-Based Filtering

```python
def create_percentile_duration_filter(audio_data: list[dict],
                                    lower_percentile: float = 5,
                                    upper_percentile: float = 95) -> tuple[float, float]:
    """Create duration filter based on percentiles."""
    
    durations = [item["duration"] for item in audio_data if item.get("duration", 0) > 0]
    
    min_duration = np.percentile(durations, lower_percentile)
    max_duration = np.percentile(durations, upper_percentile)
    
    # Ensure reasonable bounds
    min_duration = max(0.1, min_duration)  # At least 0.1s
    max_duration = min(300.0, max_duration)  # At most 5 minutes
    
    retention_count = sum(
        1 for d in durations 
        if min_duration <= d <= max_duration
    )
    
    print(f"Percentile Duration Filter (P{lower_percentile}-P{upper_percentile}):")
    print(f"  Range: {min_duration:.2f}s - {max_duration:.2f}s") 
    print(f"  Retention: {retention_count}/{len(durations)} ({retention_count/len(durations):.1%})")
    
    return min_duration, max_duration
```

### Content-Length Consistency

```python
@dataclass
class ContentLengthConsistencyStage(LegacySpeechStage):
    """Filter based on consistency between audio duration and text length."""
    
    min_chars_per_second: float = 3.0
    max_chars_per_second: float = 25.0
    min_words_per_second: float = 0.5
    max_words_per_second: float = 8.0
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        duration = data_entry.get("duration", 0)
        text = data_entry.get("text", "")
        
        if duration <= 0 or not text:
            return []  # Filter out invalid entries
        
        # Calculate rates
        char_rate = len(text) / duration
        word_rate = len(text.split()) / duration
        
        # Check consistency bounds
        char_rate_valid = self.min_chars_per_second <= char_rate <= self.max_chars_per_second
        word_rate_valid = self.min_words_per_second <= word_rate <= self.max_words_per_second
        
        if char_rate_valid and word_rate_valid:
            data_entry["char_rate"] = char_rate
            data_entry["word_rate"] = word_rate
            data_entry["content_length_consistent"] = True
            return [AudioBatch(data=data_entry)]
        else:
            return []  # Filter out inconsistent samples
```

## Duration Distribution Analysis

### Dataset Characterization

```python
def analyze_duration_characteristics(audio_data: list[dict]) -> dict:
    """Comprehensive duration analysis for filtering strategy."""
    
    durations = [item["duration"] for item in audio_data if item.get("duration", 0) > 0]
    
    analysis = {
        "basic_statistics": {
            "count": len(durations),
            "total_hours": sum(durations) / 3600,
            "mean": np.mean(durations),
            "median": np.median(durations),
            "std": np.std(durations),
            "min": min(durations),
            "max": max(durations)
        },
        
        "distribution_analysis": {
            "percentiles": {
                f"p{p}": np.percentile(durations, p)
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            
            "duration_bins": {
                "very_short": sum(1 for d in durations if d < 0.5),
                "short": sum(1 for d in durations if 0.5 <= d < 2.0),
                "normal": sum(1 for d in durations if 2.0 <= d < 10.0),
                "long": sum(1 for d in durations if 10.0 <= d < 30.0),
                "very_long": sum(1 for d in durations if d >= 30.0)
            }
        },
        
        "filtering_recommendations": {}
    }
    
    # Generate filtering recommendations
    recommendations = []
    
    very_short_ratio = analysis["distribution_analysis"]["duration_bins"]["very_short"] / len(durations)
    if very_short_ratio > 0.1:  # > 10% very short
        recommendations.append(f"Consider filtering very short audio (< 0.5s): {very_short_ratio:.1%} of dataset")
    
    very_long_ratio = analysis["distribution_analysis"]["duration_bins"]["very_long"] / len(durations)
    if very_long_ratio > 0.05:  # > 5% very long
        recommendations.append(f"Consider segmenting very long audio (≥ 30s): {very_long_ratio:.1%} of dataset")
    
    # Suggest optimal range based on data distribution
    p10 = analysis["distribution_analysis"]["percentiles"]["p10"]
    p90 = analysis["distribution_analysis"]["percentiles"]["p90"]
    
    optimal_min = max(0.5, p10)
    optimal_max = min(30.0, p90)
    
    analysis["filtering_recommendations"] = {
        "suggested_min_duration": optimal_min,
        "suggested_max_duration": optimal_max,
        "retention_rate_estimate": sum(
            1 for d in durations if optimal_min <= d <= optimal_max
        ) / len(durations),
        "recommendations": recommendations
    }
    
    return analysis
```

### Visualization and Reporting

```python
import matplotlib.pyplot as plt

def visualize_duration_distribution(audio_data: list[dict], output_path: str = None):
    """Create duration distribution visualizations."""
    
    durations = [item["duration"] for item in audio_data if item.get("duration", 0) > 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(durations, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Duration Distribution")
    axes[0, 0].set_xlabel("Duration (seconds)")
    axes[0, 0].set_ylabel("Count")
    
    # Box plot
    axes[0, 1].boxplot(durations)
    axes[0, 1].set_title("Duration Box Plot")
    axes[0, 1].set_ylabel("Duration (seconds)")
    
    # Cumulative distribution
    sorted_durations = sorted(durations)
    cumulative_pct = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    axes[1, 0].plot(sorted_durations, cumulative_pct)
    axes[1, 0].set_title("Cumulative Duration Distribution")
    axes[1, 0].set_xlabel("Duration (seconds)")
    axes[1, 0].set_ylabel("Cumulative Percentage")
    
    # Duration categories
    categories = ["< 0.5s", "0.5-2s", "2-10s", "10-30s", "> 30s"]
    counts = [
        sum(1 for d in durations if d < 0.5),
        sum(1 for d in durations if 0.5 <= d < 2.0),
        sum(1 for d in durations if 2.0 <= d < 10.0),
        sum(1 for d in durations if 10.0 <= d < 30.0),
        sum(1 for d in durations if d >= 30.0)
    ]
    
    axes[1, 1].bar(categories, counts)
    axes[1, 1].set_title("Duration Categories")
    axes[1, 1].set_xlabel("Duration Range")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

## Speech Rate Filtering

### Normal Speech Rate Range

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Filter by word rate (assumes word_rate field exists in your data)
# Normal speech: 1.5-5.0 words per second
word_rate_min_filter = PreserveByValueStage(
    input_value_key="word_rate",
    target_value=1.5,
    operator="ge"
)

word_rate_max_filter = PreserveByValueStage(
    input_value_key="word_rate",
    target_value=5.0,
    operator="le"
)

# Filter by character rate (assumes char_rate field exists in your data)
# Normal speech: 8-30 characters per second
char_rate_min_filter = PreserveByValueStage(
    input_value_key="char_rate",
    target_value=8.0,
    operator="ge"
)

char_rate_max_filter = PreserveByValueStage(
    input_value_key="char_rate",
    target_value=30.0,
    operator="le"
)
```

:::{note}
These examples assume you have pre-calculated speech rate metrics in your audio data. Use the `get_wordrate()` and `get_charrate()` utility functions to calculate these values in a custom processing stage.
:::

### Language-Specific Speech Rates

```python
# Speech rate norms by language
language_speech_rates = {
    "en": {"min_wps": 1.8, "max_wps": 4.5, "typical": (2.5, 3.5)},
    "es": {"min_wps": 2.0, "max_wps": 5.0, "typical": (3.0, 4.0)},  # Spanish tends faster
    "de": {"min_wps": 1.5, "max_wps": 4.0, "typical": (2.0, 3.0)},  # German compound words
    "fr": {"min_wps": 2.0, "max_wps": 4.8, "typical": (2.8, 3.8)},
    "zh": {"min_wps": 1.0, "max_wps": 3.5, "typical": (1.5, 2.5)}   # Character-based measurement
}

# Example: Apply language-specific filtering (assumes language and word_rate fields exist)
from nemo_curator.stages.audio.common import PreserveByValueStage

# For English audio samples
en_word_rate_filter = PreserveByValueStage(
    input_value_key="word_rate",
    target_value=1.8,  # English minimum
    operator="ge"
)

# For Spanish audio samples  
es_word_rate_filter = PreserveByValueStage(
    input_value_key="word_rate",
    target_value=2.0,  # Spanish minimum
    operator="ge"
)
```

:::{note}
This shows reference speech rate values for different languages. To implement language-aware filtering, you would need to create custom logic that selects appropriate thresholds based on your data language field.
:::

## Combined Duration and Quality Filtering

### Integrated Quality-Duration Pipeline

```python
def create_integrated_quality_duration_pipeline() -> Pipeline:
    """Combine duration, speech rate, and WER filtering."""
    
    pipeline = Pipeline(name="integrated_quality_duration")
    
    # Step 1: Calculate all metrics
    pipeline.add_stage(GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    ))
    pipeline.add_stage(SpeechRateAnalysisStage())
    pipeline.add_stage(GetPairwiseWerStage())
    
    # Step 2: Apply comprehensive filtering
    @processing_stage(name="comprehensive_quality_filter")
    def comprehensive_filter(audio_batch: AudioBatch) -> AudioBatch:
        filtered_data = []
        
        for item in audio_batch.data:
            duration = item.get("duration", 0)
            wer = item.get("wer", 100)
            word_rate = item.get("words_per_second", 0)
            
            # Multi-criteria filtering
            duration_ok = 1.0 <= duration <= 20.0
            wer_ok = wer <= 35.0
            speech_rate_ok = 1.5 <= word_rate <= 5.0
            
            # Adaptive criteria based on other metrics
            if wer <= 15.0:  # Excellent WER - relax duration requirements
                duration_ok = 0.5 <= duration <= 30.0
            
            if duration >= 10.0:  # Long audio - relax WER requirements slightly
                wer_ok = wer <= 40.0
            
            # Apply combined filter
            if duration_ok and wer_ok and speech_rate_ok:
                item["comprehensive_filter_passed"] = True
                item["filter_criteria"] = {
                    "duration_range": [1.0, 20.0],
                    "wer_threshold": 35.0,
                    "speech_rate_range": [1.5, 5.0]
                }
                filtered_data.append(item)
        
        return AudioBatch(data=filtered_data, filepath_key=audio_batch.filepath_key)
    
    pipeline.add_stage(comprehensive_filter)
    
    return pipeline
```

## Performance Impact Analysis

### Filtering Impact Assessment

```python
def assess_filtering_impact(original_data: list[dict], 
                          filtered_data: list[dict]) -> dict:
    """Assess impact of duration filtering on dataset."""
    
    original_durations = [item["duration"] for item in original_data]
    filtered_durations = [item["duration"] for item in filtered_data]
    
    impact_analysis = {
        "dataset_changes": {
            "original_count": len(original_data),
            "filtered_count": len(filtered_data),
            "retention_rate": len(filtered_data) / len(original_data),
            "samples_removed": len(original_data) - len(filtered_data)
        },
        
        "duration_changes": {
            "original_total_hours": sum(original_durations) / 3600,
            "filtered_total_hours": sum(filtered_durations) / 3600,
            "hour_retention_rate": sum(filtered_durations) / sum(original_durations),
            "mean_duration_change": np.mean(filtered_durations) - np.mean(original_durations)
        },
        
        "quality_changes": {}
    }
    
    # Quality comparison if WER available
    if "wer" in original_data[0]:
        original_wers = [item["wer"] for item in original_data]
        filtered_wers = [item["wer"] for item in filtered_data]
        
        impact_analysis["quality_changes"] = {
            "original_mean_wer": np.mean(original_wers),
            "filtered_mean_wer": np.mean(filtered_wers),
            "wer_improvement": np.mean(original_wers) - np.mean(filtered_wers),
            "quality_variance_reduction": np.std(original_wers) - np.std(filtered_wers)
        }
    
    return impact_analysis
```

## Best Practices

### Duration Filtering Strategy

1. **Analyze First**: Understand your dataset's duration distribution
2. **Use Case Alignment**: Align duration ranges with intended use
3. **Progressive Filtering**: Apply duration filters before computationally expensive stages
4. **Quality Correlation**: Consider correlation between duration and other quality metrics

### Common Pitfalls

**Over-Filtering**: Removing too much data

```python
# Check retention rates before applying filters
retention_rate = filtered_count / original_count
if retention_rate < 0.5:  # Less than 50% retained
    print("Warning: Very aggressive filtering - consider relaxing thresholds")
```

**Under-Filtering**: Keeping problematic samples

```python
# Monitor quality distribution after filtering
post_filter_analysis = analyze_duration_characteristics(filtered_data)
if post_filter_analysis["distribution_analysis"]["duration_bins"]["very_long"] > 0.1:
    print("Warning: Still have many very long samples - consider stricter filtering")
```

### Validation and Testing

```python
def validate_duration_filtering_strategy(audio_data: list[dict],
                                       min_duration: float,
                                       max_duration: float) -> dict:
    """Validate proposed duration filtering strategy."""
    
    # Simulate filtering
    filtered_data = [
        item for item in audio_data
        if min_duration <= item.get("duration", 0) <= max_duration
    ]
    
    # Calculate impact
    impact = assess_filtering_impact(audio_data, filtered_data)
    
    # Generate validation report
    validation_report = {
        "filter_parameters": {
            "min_duration": min_duration,
            "max_duration": max_duration
        },
        "impact_summary": impact,
        "validation_status": "passed",
        "warnings": []
    }
    
    # Check for potential issues
    if impact["dataset_changes"]["retention_rate"] < 0.3:
        validation_report["warnings"].append("Very low retention rate - consider relaxing thresholds")
        validation_report["validation_status"] = "warning"
    
    if impact["duration_changes"]["hour_retention_rate"] < 0.5:
        validation_report["warnings"].append("Significant audio duration loss - verify thresholds")
    
    return validation_report
```

## Real Working Example

Here's a complete working example from the NeMo Curator tutorials showing actual duration filtering in practice:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.resources import Resources

def create_audio_pipeline(raw_data_dir: str, wer_threshold: float = 75.0) -> Pipeline:
    """Real working pipeline from NeMo Curator tutorials."""
    
    pipeline = Pipeline(name="audio_inference", description="Inference audio and filter by WER threshold.")
    
    # Load FLEURS dataset
    pipeline.add_stage(
        CreateInitialManifestFleursStage(
            lang="hy_am",
            split="dev", 
            raw_data_dir=raw_data_dir,
        ).with_(batch_size=4)
    )
    
    # ASR inference
    pipeline.add_stage(
        InferenceAsrNemoStage(
            model_name="nvidia/stt_hy_fastconformer_hybrid_large_pc"
        ).with_(resources=Resources(gpus=1.0))
    )
    
    # Calculate WER
    pipeline.add_stage(
        GetPairwiseWerStage(
            text_key="text", 
            pred_text_key="pred_text", 
            wer_key="wer"
        )
    )
    
    # Calculate duration
    pipeline.add_stage(
        GetAudioDurationStage(
            audio_filepath_key="audio_filepath", 
            duration_key="duration"
        )
    )
    
    # Filter by WER threshold
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="wer", 
            target_value=wer_threshold, 
            operator="le"
        )
    )
    
    # Convert to document format
    pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
    
    return pipeline
```

:::{note}
This example comes directly from `tutorials/audio/fleurs/pipeline.py` and shows the correct parameter names and usage patterns for the built-in stages.
:::

## Related Topics

- **[Quality Assessment Overview](index.md)**: Complete quality filtering workflow
- **[WER Filtering](wer-filtering.md)**: Transcription accuracy filtering
- **[Custom Metrics](custom-metrics.md)**: More quality measures
- **[Audio Analysis](../audio-analysis/index.md)**: Duration calculation and analysis
