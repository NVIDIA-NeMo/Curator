---
description: "Step-by-step beginner tutorial for audio curation using the FLEURS dataset with ASR inference and quality filtering"
categories: ["tutorials"]
tags: ["beginner", "fleurs-dataset", "asr-inference", "quality-filtering", "hands-on"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "audio-only"
---

(audio-tutorials-beginner)=
# Create an Audio Pipeline

Learn the basics of creating an audio pipeline in Curator by following an ASR inference and quality filtering example.

```{contents} Tutorial Steps:
:local:
:depth: 2
```

## Before You Start

- Follow the [Get Started guide](gs-audio) to install the package, prepare your environment, and set up your data paths.

### Concepts and Mental Model

Use this overview to understand how stages pass data through the pipeline.

```{mermaid}
flowchart LR
  A[Audio Files] --> R[AudioReader]
  R --> ASR[ASR Inference]
  ASR --> M[Quality Metrics]
  M --> F[Filter by Quality]
  F --> W[Write Results]
  classDef dim fill:#f6f8fa,stroke:#d0d7de,color:#24292f;
  class R,ASR,M,F,W dim;
```

- **Pipeline**: An ordered list of stages that process data.
- **Stage**: A modular operation (for example, read, transcribe, filter, write).
- **Executor**: Runs the pipeline (Ray/Xenna backend).
- **Data units**: Input audio → transcriptions → quality metrics → filtered samples.
- **Common choices**:
  - **ASR Models**: NeMo FastConformer, Whisper, or custom models
  - **Quality Metrics**: Word Error Rate (WER), confidence scores
  - **Filtering**: WER thresholds, duration limits, confidence filtering
- **Outputs**: Filtered manifests (JSONL), quality reports, and processed audio metadata for downstream tasks (such as ASR training).

For more information, refer to the [Audio Concepts](about-concepts-audio) section.

---

## 1. Define Imports and Paths

Import required classes and define paths used throughout the example.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor

from nemo_curator.stages.audio.datasets.fleurs import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources

DATA_DIR = "/path/to/audio_data"
OUT_DIR = "/path/to/output"
LANGUAGE = "hy_am"  # Armenian
```

## 2. Create the Pipeline

Instantiate a named pipeline to orchestrate the stages.

```python
pipeline = Pipeline(
    name="audio_curation", 
    description="ASR inference and quality filtering"
)
```

## 3. Define Stages

Add modular stages to read, transcribe, assess quality, filter, and write outputs.

### Load Audio Dataset

Load the FLEURS dataset and create initial audio manifests.

```python
pipeline.add_stage(
    CreateInitialManifestFleursStage(
        lang=LANGUAGE,
        split="dev",
        raw_data_dir=DATA_DIR
    ).with_(batch_size=4)
)
```

### Perform ASR Inference

[Transcribe audio](audio-process-asr-nemo) using NeMo Framework models.

```python
pipeline.add_stage(
    InferenceAsrNemoStage(
        model_name="nvidia/stt_hy_fastconformer_hybrid_large_pc",
        filepath_key="audio_filepath",
        pred_text_key="pred_text"
    ).with_(resources=Resources(gpus=1.0))
)
```

### Calculate Quality Metrics

Compute Word Error Rate between ground truth and predicted transcriptions.

```python
pipeline.add_stage(
    GetPairwiseWerStage(
        text_key="text",
        pred_text_key="pred_text", 
        wer_key="wer"
    )
)
```

### Add Audio Duration

Extract audio duration metadata for filtering and analysis.

```python
pipeline.add_stage(
    GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    )
)
```

### Filter by Quality

Keep only high-quality samples based on WER threshold.

```python
pipeline.add_stage(
    PreserveByValueStage(
        input_value_key="wer",
        target_value=50.0,  # WER <= 50%
        operator="le"
    )
)
```

### Export Results

Convert to document format and write filtered results. Refer to [Save & Export](audio-save-export) for output format details.

```python
pipeline.add_stage(AudioToDocumentStage())

pipeline.add_stage(
    JsonlWriter(
        path=OUT_DIR,
        write_kwargs={"force_ascii": False}
    )
)
```

## 4. Run the Pipeline

Run the configured pipeline using the executor.

```python
pipeline.run()
```

