---
description: "Beginner tutorial for audio processing using the FLEURS dataset"
categories: ["tutorials"]
tags: ["beginner", "fleurs-dataset", "asr-inference", "quality-filtering"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "audio-only"
---

(audio-tutorials-beginner)=
# Beginner Audio Processing Tutorial

Learn the basics of audio processing with NeMo Curator using the FLEURS multilingual speech dataset. This tutorial walks you through a complete audio processing pipeline from data loading to quality assessment and filtering.

## Overview

This tutorial demonstrates the core audio processing workflow:

1. **Load Dataset**: Download and prepare the FLEURS dataset
2. **ASR Inference**: Transcribe audio using NeMo ASR models  
3. **Quality Assessment**: Calculate Word Error Rate (WER)
4. **Duration Analysis**: Extract audio file durations
5. **Filtering**: Keep only high-quality samples
6. **Export**: Save processed results

## Working Example Location

The complete working code for this tutorial is located at:
```
tutorials/audio/fleurs/
```

## Prerequisites

- NeMo Curator installed
- NVIDIA GPU (recommended for ASR inference)
- Internet connection for dataset download
- Basic Python knowledge

## Step-by-Step Walkthrough

### Step 1: Import Required Modules

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter
```

### Step 2: Create the Pipeline

```python
def create_audio_pipeline(args):
    """Create audio processing pipeline."""
    
    pipeline = Pipeline(name="audio_inference", description="Process FLEURS dataset with ASR")
    
    # Stage 1: Load FLEURS dataset
    pipeline.add_stage(
        CreateInitialManifestFleursStage(
            lang=args.lang,           # e.g., "hy_am" for Armenian
            split=args.split,         # "dev", "train", or "test"
            raw_data_dir=args.raw_data_dir
        )
    )
    
    # Stage 2: ASR inference
    pipeline.add_stage(
        InferenceAsrNemoStage(
            model_name=args.model_name  # e.g., "nvidia/stt_hy_fastconformer_hybrid_large_pc"
        )
    )
    
    # Stage 3: Calculate WER
    pipeline.add_stage(
        GetPairwiseWerStage(
            text_key="text",           # Ground truth field
            pred_text_key="pred_text", # ASR prediction field
            wer_key="wer"             # Output WER field
        )
    )
    
    # Stage 4: Extract duration
    pipeline.add_stage(
        GetAudioDurationStage(
            audio_filepath_key="audio_filepath",
            duration_key="duration"
        )
    )
    
    # Stage 5: Filter by WER threshold
    pipeline.add_stage(
        PreserveByValueStage(
            input_value_key="wer",
            target_value=args.wer_threshold,  # e.g., 75.0
            operator="le"  # less than or equal
        )
    )
    
    # Stage 6: Convert to DocumentBatch for export
    pipeline.add_stage(AudioToDocumentStage())
    
    # Stage 7: Export results
    result_dir = f"{args.raw_data_dir}/result"
    pipeline.add_stage(
        JsonlWriter(
            path=result_dir,
            write_kwargs={"force_ascii": False}
        )
    )
    
    return pipeline
```

### Step 3: Run the Pipeline

```python
def main():
    # Configuration
    class Args:
        lang = "hy_am"  # Armenian language
        split = "dev"   # Development split
        raw_data_dir = "/data/fleurs_output"
        model_name = "nvidia/stt_hy_fastconformer_hybrid_large_pc"
        wer_threshold = 75.0
    
    args = Args()
    
    # Create and run pipeline
    pipeline = create_audio_pipeline(args)
    pipeline.run()
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()
```

## Running the Complete Example

To run the working tutorial:

```bash
cd tutorials/audio/fleurs/

# Basic run with default settings
python run.py --raw_data_dir /data/fleurs_output

# Customize parameters
python run.py \
    --raw_data_dir /data/fleurs_output \
    --lang ko_kr \
    --split train \
    --model_name nvidia/stt_ko_fastconformer_hybrid_large_pc \
    --wer_threshold 50.0
```

## Understanding the Results

After running the pipeline, you'll find:

- **Downloaded data**: FLEURS audio files and transcriptions
- **Processed manifest**: JSONL file with ASR predictions and quality metrics
- **Filtered results**: Only samples meeting the WER threshold

Example output entry:
```json
{
    "audio_filepath": "/data/fleurs_output/dev/sample.wav",
    "text": "բարև աշխարհ",
    "pred_text": "բարև աշխարհ", 
    "wer": 0.0,
    "duration": 2.3
}
```
