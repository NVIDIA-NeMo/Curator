---
description: "Load audio files from local directories by creating custom manifests"
categories: ["data-loading"]
tags: ["local-files", "custom-manifests", "audio-discovery"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---

(audio-load-data-local)=

# Load Local Audio Files

Load audio files from local directories by creating custom manifests that reference your audio files. This guide covers supported formats and basic approaches for organizing local audio data for NeMo Curator processing.

## Overview

To process local audio files with NeMo Curator, you need to create a manifest file that lists your audio files and their metadata. NeMo Curator doesn't provide automatic audio file discovery - you must create a JSONL manifest first.

## Supported Audio Formats

NeMo Curator supports audio formats compatible with the `soundfile` library:

| Format | Extension | Description | Recommended Use |
|--------|-----------|-------------|-----------------|
| WAV | `.wav` | Uncompressed, high quality | ASR training, high-quality datasets |
| FLAC | `.flac` | Lossless compression | Archival, high-quality with compression |
| MP3 | `.mp3` | Compressed format | Web content, podcasts |
| OGG | `.ogg` | Open-source compression | General purpose |

:::{note}
MP3 (`.mp3`) support depends on your system's `libsndfile` build. For the most reliable behavior across environments, prefer WAV (`.wav`) or FLAC (`.flac`) formats.
:::

## Creating Manifests for Local Files

### Basic Manifest Creation

Create a JSONL manifest file that lists your local audio files:

```python
import os
import json

def create_audio_manifest(audio_dir: str, manifest_path: str):
    """Create a basic manifest for local audio files."""
    
    manifest_entries = []
    
    # Find all audio files in directory
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.flac', '.mp3', '.ogg')):
            audio_path = os.path.abspath(os.path.join(audio_dir, filename))
            
            # Basic entry - ASR will generate transcriptions
            entry = {
                "audio_filepath": audio_path,
                "text": ""  # Empty - will be filled by ASR inference
            }
            manifest_entries.append(entry)
    
    # Write manifest file
    with open(manifest_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created manifest with {len(manifest_entries)} entries: {manifest_path}")

# Usage
create_audio_manifest(
    audio_dir="/path/to/your/audio/files",
    manifest_path="local_audio_manifest.jsonl"
)
```

### Directory Organization Examples

**Paired Audio-Text Files**:

```text
/data/my_speech/
├── sample_001.wav
├── sample_001.txt
├── sample_002.wav
├── sample_002.txt
└── ...
```

**Separated Directories**:

```text
/data/my_speech/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcripts/
    ├── sample_001.txt
    ├── sample_002.txt
    └── ...
```

### Processing Local Audio with Manifest

After creating your manifest, process it with NeMo Curator:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter

def process_local_audio_manifest(manifest_path: str, output_dir: str):
    """Process local audio files using a manifest."""
    
    pipeline = Pipeline(name="local_audio_processing")
    
    # Load manifest
    pipeline.add_stage(JsonlReader(file_paths=manifest_path))
    
    # ASR processing
    pipeline.add_stage(InferenceAsrNemoStage(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
    ))
    
    # Calculate duration and filter
    pipeline.add_stage(GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    ))
    
    # Keep files between 1-30 seconds
    pipeline.add_stage(PreserveByValueStage(
        input_value_key="duration",
        target_value=1.0,
        operator="ge"
    ))
    pipeline.add_stage(PreserveByValueStage(
        input_value_key="duration",
        target_value=30.0,
        operator="le"
    ))
    
    # Export results
    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(path=output_dir))
    
    # Execute pipeline
    pipeline.run()

# Usage: First create manifest, then process
create_audio_manifest("/path/to/audio/files", "local_manifest.jsonl")
process_local_audio_manifest("local_manifest.jsonl", "/output/processed_audio")
```

### Manifest with Existing Transcriptions

If you have existing transcription files, include them in your manifest:

```python
import os
import json

def create_manifest_with_transcripts(audio_dir: str, transcript_dir: str, manifest_path: str):
    """Create manifest pairing audio files with existing transcriptions."""
    
    manifest_entries = []
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.flac', '.mp3', '.ogg')):
            # Find matching transcript file
            base_name = os.path.splitext(filename)[0]
            transcript_file = os.path.join(transcript_dir, f"{base_name}.txt")
            
            audio_path = os.path.abspath(os.path.join(audio_dir, filename))
            
            # Read transcript if it exists
            text = ""
            if os.path.exists(transcript_file):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            
            entry = {
                "audio_filepath": audio_path,
                "text": text
            }
            manifest_entries.append(entry)
    
    # Write manifest
    with open(manifest_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created manifest with {len(manifest_entries)} entries: {manifest_path}")

# Usage
create_manifest_with_transcripts(
    audio_dir="/path/to/audio/files",
    transcript_dir="/path/to/transcripts", 
    manifest_path="paired_manifest.jsonl"
)
```

## Best Practices

### Organize Your Files

Structure your audio files for easy manifest creation:

```text
/data/my_speech_project/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
├── transcripts/ (optional)
│   ├── sample_001.txt
│   ├── sample_002.txt
│   └── ...
└── manifest.jsonl (generated)
```

### Validate File Paths

Ensure all audio files exist before processing:

```python
import os

def validate_manifest(manifest_path: str):
    """Check that all audio files in manifest exist."""
    
    missing_files = []
    valid_count = 0
    
    with open(manifest_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line.strip())
            audio_path = entry.get("audio_filepath", "")
            
            if not os.path.exists(audio_path):
                missing_files.append(f"Line {line_num}: {audio_path}")
            else:
                valid_count += 1
    
    if missing_files:
        print(f"Warning: {len(missing_files)} missing files:")
        for missing in missing_files[:5]:  # Show first 5
            print(f"  {missing}")
    
    print(f"Validation complete: {valid_count} valid files")
    return len(missing_files) == 0

# Validate before processing
if validate_manifest("local_manifest.jsonl"):
    process_local_audio_manifest("local_manifest.jsonl", "/output")
```
