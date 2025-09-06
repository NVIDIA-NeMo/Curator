---
description: "Load audio files directly from local directories and file systems with custom manifest creation and batch processing"
categories: ["data-loading"]
tags: ["local-files", "file-discovery", "batch-processing", "custom-manifests", "directory-scanning"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---

(audio-load-data-local)=
# Load Local Audio Files

Load audio files directly from local directories and file systems, with support for automatic file discovery, custom manifest creation, and batch processing of large audio collections.

## Overview

Loading local audio files involves:

1. **File Discovery**: Scan directories for supported audio formats
2. **Manifest Creation**: Generate structured manifests from file collections
3. **Validation**: Verify audio file integrity and accessibility
4. **Batch Processing**: Organize files into efficient processing batches

## Supported Audio Formats

NeMo Curator supports all audio formats compatible with the soundfile library:

| Format | Extension | Description | Recommended Use |
|--------|-----------|-------------|-----------------|
| WAV | `.wav` | Uncompressed, high quality | ASR training, high-quality datasets |
| FLAC | `.flac` | Lossless compression | Archival, high-quality with compression |
| MP3 | `.mp3` | Lossy compression | Web content, podcasts |
| OGG | `.ogg` | Open-source compression | General purpose |
| M4A | `.m4a` | AAC compression | Mobile recordings |

```{note}
MP3 (`.mp3`) and M4A (`.m4a`) support depends on your system's libsndfile build. For the most reliable behavior across environments, prefer WAV (`.wav`) or FLAC (`.flac`).
```

## Directory Scanning

::::{tab-set}

:::{tab-item} Automatic File Discovery

```python
import os
from pathlib import Path
from nemo_curator.tasks import AudioBatch
import soundfile as sf

def discover_audio_files(directory: str, extensions: list = None) -> list[dict]:
    """Discover audio files in directory structure."""
    
    if extensions is None:
        extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    
    audio_files = []
    directory_path = Path(directory)
    
    for ext in extensions:
        # Find all files with this extension (recursive)
        for audio_file in directory_path.rglob(f"*{ext}"):
            try:
                # Validate audio file
                info = sf.info(str(audio_file))
                
                audio_files.append({
                    "audio_filepath": str(audio_file.absolute()),
                    "filename": audio_file.name,
                    "relative_path": str(audio_file.relative_to(directory_path)),
                    "duration": info.duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format
                })
                
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
                continue
    
    return audio_files

# Usage
audio_files = discover_audio_files("/data/my_audio_collection")
print(f"Found {len(audio_files)} valid audio files")
```

:::

:::{tab-item} Organized Directory Structures

Handle common audio dataset organizations:

```python
def load_organized_dataset(base_dir: str) -> dict[str, list[dict]]:
    """Load audio from organized directory structure."""
    
    # Common patterns:
    # /dataset/speaker_001/*.wav
    # /dataset/language/speaker/*.wav  
    # /dataset/train/audio/*.wav + /dataset/train/transcripts/*.txt
    
    datasets = {}
    base_path = Path(base_dir)
    
    # Pattern 1: Speaker-organized
    for speaker_dir in base_path.glob("speaker_*"):
        if speaker_dir.is_dir():
            speaker_files = discover_audio_files(str(speaker_dir))
            datasets[speaker_dir.name] = speaker_files
    
    # Pattern 2: Split-organized (train/dev/test)
    for split_dir in base_path.glob("*/"):
        if split_dir.name in ["train", "dev", "test", "validation"]:
            split_files = discover_audio_files(str(split_dir))
            datasets[split_dir.name] = split_files
    
    return datasets
```

:::

::::

## Manifest Creation

::::{tab-set}

:::{tab-item} From Directory with Transcriptions

```python
def create_manifest_with_transcripts(audio_dir: str, transcript_dir: str, 
                                   output_manifest: str) -> None:
    """Create manifest from separate audio and transcript directories."""
    
    audio_path = Path(audio_dir)
    transcript_path = Path(transcript_dir)
    
    manifest_entries = []
    
    for audio_file in audio_path.rglob("*.wav"):
        # Find corresponding transcript file
        transcript_file = transcript_path / (audio_file.stem + ".txt")
        
        if transcript_file.exists():
            # Read transcription
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
            
            # Get audio info
            try:
                info = sf.info(str(audio_file))
                
                entry = {
                    "audio_filepath": str(audio_file.absolute()),
                    "text": transcription,
                    "duration": info.duration,
                    "sample_rate": info.samplerate,
                    "channels": info.channels
                }
                manifest_entries.append(entry)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
    
    # Write JSONL manifest
    import json
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created manifest with {len(manifest_entries)} entries")

# Usage
create_manifest_with_transcripts(
    audio_dir="/data/speech/audio",
    transcript_dir="/data/speech/transcripts", 
    output_manifest="/data/speech/manifest.jsonl"
)
```

:::

:::{tab-item} From Filename Patterns

Extract metadata from structured filenames:

```python
import re

def create_manifest_from_filenames(audio_dir: str, filename_pattern: str) -> list[dict]:
    """Create manifest using filename patterns for metadata extraction."""
    
    # Example pattern: "speaker_{speaker}_session_{session}_{utterance}.wav"
    pattern = re.compile(filename_pattern)
    
    manifest_entries = []
    
    for audio_file in Path(audio_dir).rglob("*.wav"):
        match = pattern.match(audio_file.name)
        
        if match:
            metadata = match.groupdict()
            
            # Get audio duration
            try:
                info = sf.info(str(audio_file))
                
                entry = {
                    "audio_filepath": str(audio_file.absolute()),
                    "duration": info.duration,
                    **metadata  # Add extracted metadata
                }
                
                # Look for transcription in filename or separate file
                transcript_file = audio_file.with_suffix(".txt")
                if transcript_file.exists():
                    with open(transcript_file, 'r') as f:
                        entry["text"] = f.read().strip()
                
                manifest_entries.append(entry)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
    
    return manifest_entries

# Usage for pattern: "speaker_001_session_01_utterance_005.wav"
pattern = r"speaker_(?P<speaker_id>\d+)_session_(?P<session_id>\d+)_utterance_(?P<utterance_id>\d+)\.wav"
entries = create_manifest_from_filenames("/data/structured_audio", pattern)
```

:::

::::

## Batch Creation Strategies

::::{tab-set}

:::{tab-item} Size-Based Batching

```python
def create_size_based_batches(audio_files: list, batch_size: int = 32) -> list[AudioBatch]:
    """Create batches with fixed number of files."""
    
    batches = []
    
    for i in range(0, len(audio_files), batch_size):
        batch_data = audio_files[i:i + batch_size]
        
        batch = AudioBatch(
            data=batch_data,
            filepath_key="audio_filepath",
            task_id=f"local_batch_{i // batch_size:04d}",
            dataset_name="local_audio_dataset"
        )
        
        batches.append(batch)
    
    return batches
```

:::

:::{tab-item} Duration-Based Batching

```python
def create_duration_based_batches(audio_files: list, target_duration: float = 300.0) -> list[AudioBatch]:
    """Create batches with target total duration."""
    
    batches = []
    current_batch = []
    current_duration = 0.0
    
    for audio_file in audio_files:
        duration = audio_file.get("duration", 0)
        
        # Start new batch if target duration exceeded
        if current_duration + duration > target_duration and current_batch:
            batch = AudioBatch(
                data=current_batch,
                filepath_key="audio_filepath",
                task_id=f"duration_batch_{len(batches):04d}"
            )
            batches.append(batch)
            
            current_batch = []
            current_duration = 0.0
        
        current_batch.append(audio_file)
        current_duration += duration
    
    # Handle remaining files
    if current_batch:
        batch = AudioBatch(
            data=current_batch,
            filepath_key="audio_filepath", 
            task_id=f"duration_batch_{len(batches):04d}"
        )
        batches.append(batch)
    
    return batches
```

:::
::::

## Validation and Quality Control

::::{tab-set}

:::{tab-item} Pre-Processing Validation

```python
def validate_local_audio_dataset(audio_dir: str) -> dict:
    """Validate local audio dataset before processing."""
    
    validation_results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": [],
        "format_distribution": {},
        "duration_stats": [],
        "size_stats": []
    }
    
    for audio_file in Path(audio_dir).rglob("*"):
        if audio_file.suffix.lower() in ['.wav', '.flac', '.mp3', '.ogg', '.m4a']:
            validation_results["total_files"] += 1
            
            try:
                # Validate audio file
                info = sf.info(str(audio_file))
                
                validation_results["valid_files"] += 1
                validation_results["duration_stats"].append(info.duration)
                validation_results["size_stats"].append(audio_file.stat().st_size)
                
                # Track format distribution
                format_key = f"{info.format}_{info.samplerate}Hz"
                validation_results["format_distribution"][format_key] = \
                    validation_results["format_distribution"].get(format_key, 0) + 1
                
            except Exception as e:
                validation_results["invalid_files"].append({
                    "file": str(audio_file),
                    "error": str(e)
                })
    
    return validation_results

# Usage
results = validate_local_audio_dataset("/data/my_audio")
print(f"Validation: {results['valid_files']}/{results['total_files']} files valid")
```

:::

:::{tab-item} Transcription Matching

```python
def match_audio_transcriptions(audio_dir: str, transcript_dir: str) -> tuple[list, list]:
    """Match audio files with their transcriptions."""
    
    audio_files = list(Path(audio_dir).rglob("*.wav"))
    transcript_files = list(Path(transcript_dir).rglob("*.txt"))
    
    matched_pairs = []
    unmatched_audio = []
    
    for audio_file in audio_files:
        # Look for transcription with same stem name
        transcript_candidates = [
            t for t in transcript_files 
            if t.stem == audio_file.stem
        ]
        
        if transcript_candidates:
            transcript_file = transcript_candidates[0]
            
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
            
            matched_pairs.append({
                "audio_filepath": str(audio_file.absolute()),
                "text": transcription,
                "transcript_file": str(transcript_file)
            })
        else:
            unmatched_audio.append(str(audio_file))
    
    return matched_pairs, unmatched_audio
```

:::

::::

## Integration with NeMo Curator Pipeline

::::{tab-set}

:::{tab-item} Complete Local File Pipeline

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, AudioBatch
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter

def create_local_audio_pipeline(audio_directory: str) -> Pipeline:
    """Create pipeline for processing local audio files."""
    
    pipeline = Pipeline(name="local_audio_processing")
    
    # 1. Custom stage to load local files
    @processing_stage(name="local_file_loader")
    def load_local_files(_: EmptyTask) -> list[AudioBatch]:
        # Discover and validate audio files
        audio_files = discover_audio_files(audio_directory)
        
        # Create batches
        batches = create_size_based_batches(audio_files, batch_size=16)
        
        return batches
    
    pipeline.add_stage(load_local_files)
    
    # 2. Continue with standard audio processing
    pipeline.add_stage(
        InferenceAsrNemoStage(
            model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
        ).with_(resources=Resources(gpus=1.0))
    )
    
    # 3. Quality assessment
    pipeline.add_stage(GetPairwiseWerStage())
    pipeline.add_stage(GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    ))
    
    # 4. Export results
    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(
        path=f"{audio_directory}/processed_results"
    ))
    
    return pipeline

# Usage
pipeline = create_local_audio_pipeline("/data/my_speech_collection")
executor = XennaExecutor()
pipeline.run(executor)
```

:::

:::{tab-item} Incremental Processing

```python
import os
import json
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.tasks import EmptyTask, AudioBatch
from nemo_curator.pipeline import Pipeline

def create_incremental_pipeline(audio_dir: str, processed_manifest: str = None) -> Pipeline:
    """Process only new files since last run."""
    
    # Track already processed files
    processed_files = set()
    if processed_manifest and os.path.exists(processed_manifest):
        with open(processed_manifest, 'r') as f:
            for line in f:
                entry = json.loads(line)
                processed_files.add(entry["audio_filepath"])
    
    @processing_stage(name="incremental_loader")
    def load_new_files(_: EmptyTask) -> list[AudioBatch]:
        # Discover all files
        all_files = discover_audio_files(audio_dir)
        
        # Filter to new files only
        new_files = [
            f for f in all_files 
            if f["audio_filepath"] not in processed_files
        ]
        
        print(f"Processing {len(new_files)} new files out of {len(all_files)} total")
        
        return create_size_based_batches(new_files)
    
    pipeline = Pipeline(name="incremental_audio_processing")
    pipeline.add_stage(load_new_files)
    # ... continue with processing stages
    
    return pipeline
```

:::

::::

## File Organization Patterns

::::{tab-set}

:::{tab-item} Speaker-Organized Datasets

```
/data/speaker_dataset/
├── speaker_001/
│   ├── session_01/
│   │   ├── utterance_001.wav
│   │   ├── utterance_001.txt
│   │   └── ...
│   └── session_02/
├── speaker_002/
└── ...
```

```python
def load_speaker_organized_data(base_dir: str) -> dict[str, AudioBatch]:
    """Load speaker-organized audio dataset."""
    
    speaker_batches = {}
    
    for speaker_dir in Path(base_dir).glob("speaker_*"):
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        speaker_files = []
        
        # Load all audio files for this speaker
        for audio_file in speaker_dir.rglob("*.wav"):
            transcript_file = audio_file.with_suffix(".txt")
            
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    transcription = f.read().strip()
                
                try:
                    info = sf.info(str(audio_file))
                    
                    speaker_files.append({
                        "audio_filepath": str(audio_file.absolute()),
                        "text": transcription,
                        "speaker_id": speaker_id,
                        "duration": info.duration,
                        "session": audio_file.parent.name
                    })
                except Exception as e:
                    print(f"Error with {audio_file}: {e}")
        
        if speaker_files:
            speaker_batches[speaker_id] = AudioBatch(
                data=speaker_files,
                filepath_key="audio_filepath",
                dataset_name=f"speaker_dataset_{speaker_id}"
            )
    
    return speaker_batches
```

:::

:::{tab-item} Language-Organized Datasets

```python
def load_multilingual_dataset(base_dir: str) -> dict[str, AudioBatch]:
    """Load language-organized audio dataset."""
    
    language_batches = {}
    
    for lang_dir in Path(base_dir).glob("*"):
        if not lang_dir.is_dir():
            continue
            
        lang_code = lang_dir.name
        lang_files = []
        
        # Process all audio files for this language
        audio_files = discover_audio_files(str(lang_dir))
        
        for audio_info in audio_files:
            # Add language metadata
            audio_info["language"] = lang_code
            lang_files.append(audio_info)
        
        if lang_files:
            language_batches[lang_code] = AudioBatch(
                data=lang_files,
                filepath_key="audio_filepath",
                dataset_name=f"multilingual_dataset_{lang_code}"
            )
    
    return language_batches
```

:::

::::

## Performance Optimization

::::{tab-set}

:::{tab-item} Lazy Loading

```python
class LazyAudioLoader:
    """Lazy loader for large audio collections."""
    
    def __init__(self, audio_directory: str, batch_size: int = 32):
        self.audio_directory = Path(audio_directory)
        self.batch_size = batch_size
        self._audio_files = None
    
    @property
    def audio_files(self) -> list[dict]:
        """Lazy discovery of audio files."""
        if self._audio_files is None:
            self._audio_files = discover_audio_files(str(self.audio_directory))
        return self._audio_files
    
    def __iter__(self):
        """Iterate over audio batches."""
        for i in range(0, len(self.audio_files), self.batch_size):
            batch_data = self.audio_files[i:i + self.batch_size]
            yield AudioBatch(
                data=batch_data,
                filepath_key="audio_filepath",
                task_id=f"lazy_batch_{i // self.batch_size:04d}"
            )
    
    def __len__(self) -> int:
        return len(self.audio_files)

# Usage
loader = LazyAudioLoader("/large/audio/dataset", batch_size=64)
for batch in loader:
    # Process batch
    pass
```

:::

:::{tab-item} Parallel File Discovery

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import concurrent.futures

def parallel_file_discovery(directories: list[str]) -> list[dict]:
    """Discover audio files in parallel across multiple directories."""
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submit discovery tasks for each directory
        futures = {
            executor.submit(discover_audio_files, dir_path): dir_path 
            for dir_path in directories
        }
        
        all_files = []
        for future in concurrent.futures.as_completed(futures):
            directory = futures[future]
            try:
                files = future.result()
                # Add directory metadata
                for file_info in files:
                    file_info["source_directory"] = directory
                all_files.extend(files)
            except Exception as e:
                print(f"Error processing {directory}: {e}")
    
    return all_files
```

:::

::::

## Error Handling

### Robust File Processing

```python
def robust_audio_loading(audio_dir: str, skip_errors: bool = True) -> tuple[list[dict], list[dict]]:
    """Load audio files with comprehensive error handling."""
    
    valid_files = []
    error_files = []
    
    for audio_file in Path(audio_dir).rglob("*"):
        if audio_file.suffix.lower() not in ['.wav', '.flac', '.mp3', '.ogg']:
            continue
            
        try:
            # Validate audio file
            info = sf.info(str(audio_file))
            
            # Check for reasonable duration
            if not (0.1 <= info.duration <= 3600):  # 0.1s to 1 hour
                raise ValueError(f"Invalid duration: {info.duration}")
            
            # Check sample rate
            if info.samplerate < 8000:  # Minimum quality threshold
                raise ValueError(f"Low sample rate: {info.samplerate}")
            
            file_info = {
                "audio_filepath": str(audio_file.absolute()),
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format
            }
            
            valid_files.append(file_info)
            
        except Exception as e:
            error_info = {
                "audio_filepath": str(audio_file),
                "error": str(e),
                "error_type": type(e).__name__
            }
            error_files.append(error_info)
            
            if not skip_errors:
                raise
    
    return valid_files, error_files
```

## Integration Examples

### Complete Local Processing Workflow

```python
def process_local_audio_collection(audio_dir: str, output_dir: str) -> None:
    """Complete workflow for processing local audio collection."""
    
    # Step 1: Discover and validate files
    print("Discovering audio files...")
    valid_files, error_files = robust_audio_loading(audio_dir)
    
    print(f"Found {len(valid_files)} valid files, {len(error_files)} errors")
    
    # Step 2: Create manifest
    manifest_path = os.path.join(output_dir, "input_manifest.jsonl")
    with open(manifest_path, 'w') as f:
        for file_info in valid_files:
            f.write(json.dumps(file_info) + '\n')
    
    # Step 3: Create processing pipeline
    pipeline = Pipeline(name="local_audio_complete")
    
    # Load from manifest
    pipeline.add_stage(JsonlReader(file_paths=manifest_path))
    
    # ASR processing
    pipeline.add_stage(
        InferenceAsrNemoStage(
            model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
        ).with_(resources=Resources(gpus=1.0))
    )
    
    # Quality assessment (no ground truth WER for transcription-only)
    pipeline.add_stage(GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    ))
    
    # Export results
    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(
        path=os.path.join(output_dir, "transcribed_results")
    ))
    
    # Step 4: Execute pipeline
    executor = XennaExecutor()
    pipeline.run(executor)
    
    print(f"Processing complete. Results in {output_dir}")

# Usage
process_local_audio_collection(
    audio_dir="/data/podcast_collection",
    output_dir="/data/processed_podcasts"
)
```
