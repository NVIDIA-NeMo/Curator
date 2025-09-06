---
description: "Extract and analyze audio file characteristics including duration calculation, format validation, and metadata extraction"
categories: ["audio-processing"]
tags: ["audio-analysis", "duration-calculation", "format-validation", "metadata-extraction", "file-validation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "audio-only"
---

# Audio Analysis

Extract and analyze audio file characteristics for quality control, metadata generation, and dataset validation. Audio analysis provides essential information about audio files before and during processing.

## How it Works

Audio analysis in NeMo Curator examines audio files to extract:

1. **Duration Information**: Precise timing measurements using `soundfile`
2. **Format Characteristics**: Sample rate, bit depth, channels, and format
3. **Quality Indicators**: File integrity, format compliance, technical quality
4. **Metadata Extraction**: Embedded metadata and file properties

:::{note} NeMo Curator provides duration extraction as a built-in stage (`GetAudioDurationStage`). The format and metadata examples below show how to build custom stages and are not built-in.
:::

## Duration Analysis

### Precise Duration Calculation

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage

# Calculate audio duration for each file
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)
```

The duration calculation:

- Uses the `soundfile` library; computes duration as frames ÷ sample rate
- Handles formats supported by `soundfile` (`libsndfile`)
- Returns -1.0 for corrupted or unreadable files
- Calculates: `duration = sample_count / sample_rate`

### Duration-Based Quality Assessment

```python
def analyze_duration_distribution(audio_data: list[dict]) -> dict:
    """Analyze duration characteristics of audio dataset."""
    
    durations = [item["duration"] for item in audio_data if item["duration"] > 0]
    
    return {
        "total_samples": len(durations),
        "total_hours": sum(durations) / 3600,
        "mean_duration": np.mean(durations),
        "median_duration": np.median(durations),
        "duration_range": (min(durations), max(durations)),
        "percentiles": {
            "p25": np.percentile(durations, 25),
            "p50": np.percentile(durations, 50), 
            "p75": np.percentile(durations, 75),
            "p90": np.percentile(durations, 90),
            "p95": np.percentile(durations, 95)
        },
        "quality_categories": {
            "very_short": sum(1 for d in durations if d < 0.5),      # < 0.5s
            "short": sum(1 for d in durations if 0.5 <= d < 2.0),    # 0.5-2s
            "normal": sum(1 for d in durations if 2.0 <= d < 10.0),  # 2-10s
            "long": sum(1 for d in durations if 10.0 <= d < 30.0),   # 10-30s
            "very_long": sum(1 for d in durations if d >= 30.0)      # 30s+
        }
    }
```

### Duration Filtering Example

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Keep samples between 1 and 15 seconds
min_duration_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"
)
max_duration_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=15.0,
    operator="le"
)
```

Refer to [Duration Filtering](../quality-assessment/duration-filtering.md) for end-to-end examples.

## Format Validation

:::{note} The following format validation examples illustrate a custom approach and are not built-in stages.
:::

### Audio Format Analysis

```python
import soundfile as sf
from pathlib import Path

def analyze_audio_format(audio_file: str) -> dict:
    """Extract comprehensive audio format information."""
    
    try:
        info = sf.info(audio_file)
        file_path = Path(audio_file)
        
        format_info = {
            "filepath": str(file_path.absolute()),
            "filename": file_path.name,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            
            # Audio characteristics
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
            
            # Quality indicators
            "bit_depth": _estimate_bit_depth(info),
            "compression_ratio": _calculate_compression_ratio(info, file_path),
            "is_mono": info.channels == 1,
            "is_standard_rate": info.samplerate in [16000, 22050, 44100, 48000],
            
            # Validation status
            "is_valid": True,
            "validation_errors": []
        }
        
        return format_info
        
    except Exception as e:
        return {
            "filepath": audio_file,
            "is_valid": False,
            "validation_errors": [str(e)],
            "error_type": type(e).__name__
        }

def _estimate_bit_depth(info) -> int:
    """Estimate bit depth from subtype."""
    subtype_mapping = {
        'PCM_16': 16,
        'PCM_24': 24,
        'PCM_32': 32,
        'FLOAT': 32,
        'DOUBLE': 64
    }
    return subtype_mapping.get(info.subtype, 16)  # Default to 16-bit

def _calculate_compression_ratio(info, file_path: Path) -> float:
    """Calculate compression ratio compared to uncompressed PCM."""
    uncompressed_size = info.frames * info.channels * 2  # Assume 16-bit PCM
    actual_size = file_path.stat().st_size
    return actual_size / uncompressed_size if uncompressed_size > 0 else 1.0
```

### Batch Format Validation

```python
from nemo_curator.stages.audio.common import LegacySpeechStage

@dataclass
class AudioFormatValidationStage(LegacySpeechStage):
    """Validate audio format characteristics across batch."""
    
    required_sample_rate: int = 16000
    required_channels: int = 1
    max_duration: float = 30.0
    min_duration: float = 0.5
    
    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        audio_filepath = data_entry["audio_filepath"]
        
        try:
            info = sf.info(audio_filepath)
            
            # Validation checks
            validation_results = {
                "format_valid": True,
                "validation_issues": []
            }
            
            # Check sample rate
            if info.samplerate != self.required_sample_rate:
                validation_results["validation_issues"].append(
                    f"Sample rate {info.samplerate} != required {self.required_sample_rate}"
                )
            
            # Check channels
            if info.channels != self.required_channels:
                validation_results["validation_issues"].append(
                    f"Channels {info.channels} != required {self.required_channels}"
                )
            
            # Check duration
            if not (self.min_duration <= info.duration <= self.max_duration):
                validation_results["validation_issues"].append(
                    f"Duration {info.duration} outside range [{self.min_duration}, {self.max_duration}]"
                )
            
            # Mark as invalid if issues found
            if validation_results["validation_issues"]:
                validation_results["format_valid"] = False
            
            # Add validation results to data entry
            data_entry.update(validation_results)
            data_entry.update({
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "format": info.format
            })
            
        except Exception as e:
            data_entry.update({
                "format_valid": False,
                "validation_issues": [f"File read error: {str(e)}"],
                "duration": -1.0
            })
        
        return [AudioBatch(data=data_entry)]
```

## Metadata Extraction

### Comprehensive Audio Metadata

Note: The following metadata extraction example illustrates a custom approach and is not a built-in stage.

```python
def extract_comprehensive_metadata(audio_file: str) -> dict:
    """Extract all available audio metadata."""
    
    file_path = Path(audio_file)
    metadata = {
        "file_info": {
            "filepath": str(file_path.absolute()),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "modification_time": file_path.stat().st_mtime,
            "extension": file_path.suffix.lower()
        }
    }
    
    try:
        # Audio technical information
        info = sf.info(audio_file)
        metadata["audio_info"] = {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype
        }
        
        # Derived characteristics
        metadata["characteristics"] = {
            "bit_rate": _estimate_bit_rate(info, file_path),
            "is_mono": info.channels == 1,
            "is_cd_quality": info.samplerate >= 44100,
            "is_speech_optimized": info.samplerate == 16000,
            "duration_category": _categorize_duration(info.duration)
        }
        
        # Quality indicators
        metadata["quality"] = {
            "file_integrity": True,
            "format_compliance": _check_format_compliance(info),
            "recommended_for_asr": _assess_asr_suitability(info)
        }
        
    except Exception as e:
        metadata["error"] = {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "file_accessible": False
        }
    
    return metadata

def _categorize_duration(duration: float) -> str:
    """Categorize audio duration for analysis."""
    if duration < 1.0:
        return "very_short"
    elif duration < 5.0:
        return "short"
    elif duration < 15.0:
        return "medium"
    elif duration < 60.0:
        return "long"
    else:
        return "very_long"

def _assess_asr_suitability(info) -> bool:
    """Assess if audio format is suitable for ASR processing."""
    return (
        info.samplerate >= 8000 and      # Minimum speech quality
        info.duration >= 0.5 and         # Minimum content length
        info.duration <= 60.0 and        # Maximum practical length
        info.channels <= 2               # Mono or stereo only
    )
```

## Statistical Analysis

### Dataset Characterization

```python
def characterize_audio_dataset(audio_files: list[dict]) -> dict:
    """Generate comprehensive dataset statistics."""
    
    valid_files = [f for f in audio_files if f.get("duration", -1) > 0]
    
    # Duration analysis
    durations = [f["duration"] for f in valid_files]
    
    # Sample rate analysis
    sample_rates = [f.get("sample_rate", 0) for f in valid_files]
    
    # Format analysis
    formats = [f.get("format", "unknown") for f in valid_files]
    
    statistics = {
        "dataset_overview": {
            "total_files": len(audio_files),
            "valid_files": len(valid_files),
            "total_duration_hours": sum(durations) / 3600,
            "average_duration": np.mean(durations),
            "median_duration": np.median(durations)
        },
        
        "duration_distribution": {
            "min": min(durations),
            "max": max(durations), 
            "std": np.std(durations),
            "percentiles": {
                f"p{p}": np.percentile(durations, p) 
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        },
        
        "format_distribution": {
            "sample_rates": dict(Counter(sample_rates)),
            "formats": dict(Counter(formats)),
            "mono_ratio": sum(1 for f in valid_files if f.get("channels") == 1) / len(valid_files)
        },
        
        "quality_assessment": {
            "asr_suitable": sum(1 for f in valid_files if _assess_asr_suitability_from_dict(f)),
            "standard_rate": sum(1 for f in valid_files if f.get("sample_rate") in [16000, 22050, 44100]),
            "optimal_duration": sum(1 for d in durations if 1.0 <= d <= 15.0)
        }
    }
    
    return statistics
```

## Related Topics

- **[Duration Calculation](duration-calculation.md)** - Detailed duration extraction methods
- **[Format Validation](format-validation.md)** - Audio format validation techniques
- **[Quality Assessment](../quality-assessment/index.md)** - Using analysis results for quality filtering
- **[Local File Loading](../../load-data/local-files.md)** - Loading files for analysis
