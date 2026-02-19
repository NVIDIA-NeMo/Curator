---
name: aud-asr
description: Transcribe audio using NeMo ASR models (Parakeet, FastConformer). GPU recommended.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: audio
  gpu-required: "recommended"
  parent-skill: audio
---

# Audio ASR (Transcription) Skill

Transcribe audio files using NeMo ASR models. GPU recommended for speed.

## When This Skill Applies

- User wants to transcribe audio to text
- User mentions: "transcribe", "ASR", "speech-to-text", "convert audio"
- User has WAV/MP3/FLAC audio files

## ASR Models (Verified)

| Model | Languages | Notes |
|-------|-----------|-------|
| `nvidia/parakeet-tdt-0.6b-v2` | English | Test-verified, recommended |

For additional models, see [NeMo ASR Model Catalog](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html).

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. What language is the audio?
2. How many hours of audio?
3. What audio format? (WAV, MP3, FLAC)

### Step 2: Create Manifest

NeMo ASR uses JSON manifest files:

```json
{"audio_filepath": "/path/to/audio.wav", "duration": 5.2}
{"audio_filepath": "/path/to/audio2.wav", "duration": 3.1}
```

Create manifest from directory:

```python
import os
import json
import librosa

def create_manifest(audio_dir: str, output_path: str):
    with open(output_path, "w") as f:
        for filename in os.listdir(audio_dir):
            if filename.endswith((".wav", ".mp3", ".flac")):
                filepath = os.path.join(audio_dir, filename)
                duration = librosa.get_duration(path=filepath)
                f.write(json.dumps({
                    "audio_filepath": filepath,
                    "duration": duration,
                }) + "\n")
```

### Step 3: Generate Pipeline Code

```python
# Audio ASR Pipeline
# GPU Recommended for speed

import torch
if not torch.cuda.is_available():
    print("Warning: GPU not available. ASR will run on CPU (slower).")

from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.pipeline import Pipeline

def transcribe_audio(input_manifest: str, output_manifest: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
    print(f"Input: {input_manifest}")
    print(f"Model: {model_name}")
    
    asr_stage = InferenceAsrNemoStage(
        model_name=model_name,
        batch_size=32,
        pred_text_key="transcript",  # Output field for transcriptions
    )
    
    pipeline = Pipeline(name="asr", stages=[asr_stage])
    results = pipeline.run()
    
    print(f"Transcribed {len(results)} audio files")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input manifest (JSONL)")
    parser.add_argument("--output", required=True, help="Output manifest")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    args = parser.parse_args()
    transcribe_audio(args.input, args.output, args.model)
```

## Stage Parameters

### InferenceAsrNemoStage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | NeMo ASR model name |
| `filepath_key` | str | `"audio_filepath"` | Key for audio file path in manifest |
| `pred_text_key` | str | `"pred_text"` | Output key for transcription |
| `batch_size` | int | `16` | Batch size for inference |

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd):/data \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/asr_pipeline.py \
    --input /data/manifest.json \
    --output /data/transcribed.json
```

## Resource Requirements

| Model | GPU Memory | Speed (approx) |
|-------|------------|----------------|
| Parakeet 0.6B | ~4GB | GPU: fast, CPU: slower |

Note: ASR can run on CPU but inference will be significantly slower.
