---
description: "Load and process the multilingual FLEURS speech dataset with automated download and manifest creation"
categories: ["data-loading"]
tags: ["fleurs", "multilingual", "speech-corpus", "automated-download", "huggingface"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---

(audio-load-data-fleurs)=

# Load FLEURS Dataset

The [FLEURS dataset](https://huggingface.co/datasets/google/fleurs) is a multilingual speech dataset covering 102 languages, built on top of the FLoRes machine translation benchmark. NeMo Curator provides automated tools to download, extract, and prepare FLEURS data for audio curation pipelines.

## How It Works

The `CreateInitialManifestFleursStage` handles the complete FLEURS data preparation workflow:

1. **Download**: Retrieves audio files and transcription files from Hugging Face
2. **Extract**: Unpacks compressed audio archives
3. **Manifest Creation**: Generates structured manifests with audio file paths and transcriptions
4. **Manifest References**: Produces entries that point to extracted audio files; this stage does not decode or check audio content

---

## Usage

### Basic FLEURS Loading

```python
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.pipeline import Pipeline

# Create FLEURS loading stage
fleurs_stage = CreateInitialManifestFleursStage(
    lang="hy_am",              # Armenian language
    split="dev",               # Development split
    raw_data_dir="/path/to/audio/data"
)

# Add to pipeline
pipeline = Pipeline(name="fleurs_loading")
pipeline.add_stage(fleurs_stage.with_(batch_size=4))

# Execute
pipeline.run()
```

Note: You can omit the explicit executor and call `pipeline.run()` without arguments. By default, it uses `XennaExecutor`.

### Language Options

FLEURS supports 102 languages identified by ISO 639-1 and ISO 3166-1 alpha-2 codes:

```python
# Common language examples
languages = [
    "en_us",    # English (US)
    "es_419",   # Spanish (Latin America) 
    "fr_fr",    # French (France)
    "de_de",    # German (Germany)
    "zh_cn",    # Chinese (Simplified)
    "ja_jp",    # Japanese
    "ko_kr",    # Korean
    "hy_am",    # Armenian
    "ar_eg",    # Arabic (Egypt)
]
```

### Data Splits

Choose from three available data splits:

```python
splits = ["train", "dev", "test"]

# Load training data for multiple languages
for lang in ["en_us", "es_419", "fr_fr"]:
    stage = CreateInitialManifestFleursStage(
        lang=lang,
        split="train", 
        raw_data_dir=f"/data/fleurs/{lang}"
    )
```

## Output Format

The FLEURS loading stage generates `AudioBatch` objects with the following structure:

```json
{
    "audio_filepath": "/absolute/path/to/audio.wav",
    "text": "ground truth transcription text"
}
```

## Configuration Options

### Stage Parameters

```python
CreateInitialManifestFleursStage(
    lang="en_us",                    # Language code
    split="train",                   # Data split
    raw_data_dir="/data/fleurs",     # Storage directory
    filepath_key="audio_filepath",   # Key for audio file paths
    text_key="text",                 # Key for transcription text
)
```

### Batch Processing

```python
# Configure batch size for processing
fleurs_stage.with_(batch_size=8)  # Process 8 files per batch
```

To persist results to disk, add a writer stage (for example, `JsonlWriter`). In a full pipeline, a writer stage creates the `result/` directory shown below.

## File Organization

After processing, your directory structure will look like:

```text
/data/fleurs/en_us/
├── dev.tsv              # Transcription metadata (used internally by the stage)
├── dev.tar.gz          # Compressed audio files
├── dev/                # Extracted audio files
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── result/             # Processed JSONL manifests (if using full pipeline)
    └── *.jsonl
```
