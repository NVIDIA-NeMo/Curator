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

This tutorial demonstrates the core audio curation workflow:

1. **Load Dataset**: Download and prepare the FLEURS dataset
2. **ASR Inference**: Transcribe audio using NeMo ASR models
3. **Quality Assessment**: Calculate Word Error Rate (WER)
4. **Duration Analysis**: Extract audio file durations
5. **Filtering**: Keep only high-quality samples
6. **Export**: Save processed results

**What you'll learn:**
- How to build an end-to-end audio curation pipeline
- Loading multilingual speech datasets (FLEURS)
- Running ASR inference with NeMo models
- Calculating quality metrics (WER, duration)
- Filtering audio by quality thresholds
- Exporting curated results in JSONL format

**Time to complete:** Approximately 15-30 minutes (depending on dataset size and GPU availability)

## Working Example Location

The complete working code for this tutorial is located at:

```
<nemo_curator_repository>/tutorials/audio/fleurs/
├── run.py              # Main tutorial script
├── README.md           # Tutorial documentation
└── requirements.txt    # Python dependencies
```

**Accessing the code:**
```bash
# Clone NeMo Curator repository
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator/tutorials/audio/fleurs/

# Install dependencies
pip install -r requirements.txt
```

## Prerequisites

* NeMo Curator installed (see [Installation Guide](docs/admin/installation.md))
* NVIDIA GPU (required for ASR inference, minimum 16GB VRAM recommended)
* Internet connection for dataset download
* Basic Python knowledge
* CUDA-compatible PyTorch installation
* Sufficient disk space (FLEURS dataset requires ~10-50GB depending on language and split)

:::{tip}
If you don't have a GPU available, you can skip the ASR inference stage and work with pre-existing transcriptions. See the [Custom Manifests](../load-data/custom-manifests.md) guide for details.
:::

## Step-by-Step Walkthrough

### Step 1: Import Required Modules

Import all necessary stages and components for the audio curation pipeline:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources
```

**Key components:**
- `Pipeline`: Container for organizing and executing processing stages
- `XennaExecutor`: Backend executor for running the pipeline
- `CreateInitialManifestFleursStage`: Downloads and prepares FLEURS dataset
- `InferenceAsrNemoStage`: Runs ASR inference with NeMo models
- `GetPairwiseWerStage`: Calculates Word Error Rate
- `PreserveByValueStage`: Filters data based on threshold values
- `JsonlWriter`: Exports results in JSONL format

### Step 2: Create the Pipeline

Build the audio curation pipeline by adding stages in sequence:

```python
def create_audio_pipeline(args):
    """Create audio curation pipeline."""
    
    pipeline = Pipeline(name="audio_inference", description="Process FLEURS dataset with ASR")
    
    # Stage 1: Load FLEURS dataset
    pipeline.add_stage(
        CreateInitialManifestFleursStage(
            lang=args.lang,           # e.g., "hy_am" for Armenian
            split=args.split,         # "dev", "train", or "test"
            raw_data_dir=args.raw_data_dir
        ).with_(batch_size=4)  # Process 4 samples per batch
    )
    
    # Stage 2: ASR inference
    pipeline.add_stage(
        InferenceAsrNemoStage(
            model_name=args.model_name,  # e.g., "nvidia/stt_hy_fastconformer_hybrid_large_pc"
            pred_text_key="pred_text"  # Field name for ASR predictions
        ).with_(resources=Resources(gpus=1.0))  # Allocate 1 GPU
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

**Stage explanations:**
1. **CreateInitialManifestFleursStage**: Downloads FLEURS dataset from Hugging Face and creates audio manifest
2. **InferenceAsrNemoStage**: Loads NeMo ASR model and generates transcriptions (requires GPU)
3. **GetPairwiseWerStage**: Compares ground truth and predictions to calculate WER
4. **GetAudioDurationStage**: Reads audio files to extract duration metadata
5. **PreserveByValueStage**: Filters samples, keeping only those with WER ≤ threshold
6. **AudioToDocumentStage**: Converts AudioBatch to DocumentBatch format for export
7. **JsonlWriter**: Saves filtered results as JSONL manifest

### Step 3: Run the Pipeline

Configure pipeline parameters and execute:

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
    
    # Create pipeline
    pipeline = create_audio_pipeline(args)
    
    # Create executor
    executor = XennaExecutor()
    
    # Run pipeline
    pipeline.run(executor)
    
    print("Pipeline completed!")
    print(f"Results saved to: {args.raw_data_dir}/result/")

if __name__ == "__main__":
    main()
```

**Configuration parameters:**
- `lang`: Language code from FLEURS dataset (e.g., "en_us", "ko_kr", "es_419")
- `split`: Dataset split to process ("dev", "train", or "test")
- `raw_data_dir`: Directory for downloading and storing FLEURS data
- `model_name`: NeMo ASR model identifier from NGC or Hugging Face
- `wer_threshold`: Maximum acceptable WER percentage (samples above this are filtered out)

:::{note}
The first run will download the FLEURS dataset for your selected language, which may take several minutes depending on network speed.
:::

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

**Command-line options:**
- `--raw_data_dir`: Output directory for dataset and results (required)
- `--lang`: FLEURS language code (default: "hy_am")
- `--split`: Dataset split to process (default: "dev")
- `--model_name`: NeMo ASR model name (default: matches language)
- `--wer_threshold`: Maximum WER for filtering (default: 75.0)

**Expected execution time:**
- Dataset download (first run): 5-15 minutes
- ASR inference: 1-5 minutes for dev split (~100 samples) with GPU
- Quality assessment and export: < 1 minute

**System requirements during execution:**
- GPU memory: 8-16GB (depending on model size)
- Disk space: 10-50GB (dataset + results)
- RAM: 8GB minimum

## Understanding the Results

After running the pipeline, you'll find:

* **Downloaded data**: FLEURS audio files and transcriptions in `<raw_data_dir>/downloads/`
* **Processed manifest**: JSONL file(s) with ASR predictions and quality metrics in `<raw_data_dir>/result/`
* **Filtered results**: Only samples meeting the WER threshold

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

**Field descriptions:**
- `audio_filepath`: Absolute path to audio file
- `text`: Ground truth transcription from FLEURS
- `pred_text`: ASR model prediction
- `wer`: Word Error Rate percentage (0.0 = perfect match)
- `duration`: Audio duration in seconds

**Analyzing results:**
```bash
# Count filtered samples
cat /data/fleurs_output/result/*.jsonl | wc -l

# View first 5 samples
head -n 5 /data/fleurs_output/result/*.jsonl | jq .

# Calculate average WER
cat /data/fleurs_output/result/*.jsonl | jq -r '.wer' | awk '{sum+=$1; count+=1} END {print "Average WER:", sum/count "%"}'
```

**Using Python:**
```python
import json
import pandas as pd
from pathlib import Path

# Load results
result_files = list(Path("/data/fleurs_output/result").glob("*.jsonl"))
data = []
for file in result_files:
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

df = pd.DataFrame(data)

# Summary statistics
print(f"Total samples: {len(df)}")
print(f"Average WER: {df['wer'].mean():.2f}%")
print(f"Average duration: {df['duration'].mean():.2f}s")
print(f"WER range: {df['wer'].min():.2f}% - {df['wer'].max():.2f}%")
```

## Troubleshooting

### Common Issues

**GPU out of memory:**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use a smaller ASR model:
```python
pipeline.add_stage(
    CreateInitialManifestFleursStage(...).with_(batch_size=2)  # Reduce from 4 to 2
)
```

**Dataset download fails:**
```
ConnectionError: Failed to download FLEURS dataset
```
**Solution:** Check internet connection and retry. The stage will resume from where it left off.

**No GPU available:**
```
RuntimeError: No CUDA GPUs are available
```
**Solution:** Ensure CUDA is installed and GPU is accessible:
```bash
nvidia-smi  # Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Model download fails:**
```
OSError: Model 'nvidia/stt_...' not found
```
**Solution:** Verify model name is correct and you have internet access. Check available models at [NGC Catalog](https://catalog.ngc.nvidia.com/).

### Performance Optimization

- **Increase batch size** for faster processing (if GPU memory allows)
- **Use multiple GPUs** by setting `Resources(gpus=2.0)` or higher
- **Process subset of data** by using `split="dev"` (smaller than "train")
- **Skip ASR inference** if you already have both predicted and target transcriptions. (remove InferenceAsrNemoStage)

## Next Steps

After completing this tutorial, explore:

- **[Custom Manifests](../load-data/custom-manifests.md)**: Process your own audio datasets
- **[WER Filtering](../process-data/quality-assessment/wer-filtering.md)**: Advanced quality filtering techniques
- **[Duration Filtering](../process-data/quality-assessment/duration-filtering.md)**: Filter by audio length and speech rate
- **[NeMo ASR Models](../process-data/asr-inference/nemo-models.md)**: Explore available ASR models for different languages

## Best Practices

- **Start with dev split**: Test your pipeline on the smaller development split before processing the full training set
- **Adjust WER thresholds by language**: Some languages may require more lenient thresholds (e.g., 75-80% for low-resource languages)
- **Monitor GPU usage**: Use `nvidia-smi` to track GPU memory and utilization during processing
- **Validate results**: Always inspect a sample of output records to verify quality
- **Document parameters**: Keep track of configuration values (thresholds, models) for reproducibility

## Related Topics

- **[Audio Curation Quickstart](docs/get-started/audio.md)**: Quick introduction to audio curation
- **[FLEURS Dataset](../load-data/fleurs-dataset.md)**: Detailed FLEURS dataset documentation
- **[Quality Assessment](../process-data/quality-assessment/index.md)**: Comprehensive quality metrics guide
- **[Save & Export](../save-export.md)**: Advanced export options and formats
