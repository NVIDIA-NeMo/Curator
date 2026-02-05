---
name: aud-wer
description: Calculate Word Error Rate (WER) to assess transcription quality. Filter low-quality audio-transcript pairs.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: audio
  gpu-required: "false"
  parent-skill: audio
---

# Audio WER (Quality) Skill

Calculate Word Error Rate to assess audio/transcript quality. WER calculation runs on CPU.

## When This Skill Applies

- User wants to filter audio by transcription quality
- User mentions: "WER", "word error rate", "quality", "accuracy"
- User has audio with reference transcripts and wants to filter low-quality pairs

## What is WER?

Word Error Rate measures transcript accuracy:
- **WER = (Insertions + Deletions + Substitutions) / Reference Words × 100**
- 0% = Perfect match
- 10% = High quality
- 20% = Acceptable
- >30% = Low quality (noisy audio or bad transcript)

**Important**: WER values are returned as percentages (e.g., 15.0 = 15%), NOT decimals.

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. Do you have reference transcripts?
2. What WER threshold is acceptable?
3. Are you filtering or just analyzing?

### Step 2: Explain the Process

1. Run ASR on audio to get predicted transcripts
2. Compare predictions to reference transcripts
3. Calculate WER for each pair
4. Filter out high-WER pairs

### Step 3: Generate Pipeline Code

```python
# Audio WER Pipeline
# GPU Recommended for ASR, WER calculation is CPU-only

import torch
if not torch.cuda.is_available():
    print("Warning: GPU not available. ASR will run on CPU (slower).")

from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import PreserveByValueStage
from nemo_curator.pipeline import Pipeline

def filter_by_wer(input_manifest: str, output_manifest: str, max_wer: float = 15.0, reference_field: str = "text"):
    """Filter audio by WER threshold.
    
    Args:
        max_wer: Maximum WER as percentage (e.g., 15.0 = 15%). NOT decimal.
    """
    print(f"Input: {input_manifest}")
    print(f"Max WER: {max_wer}%")
    
    # Stage 1: Run ASR
    asr_stage = InferenceAsrNemoStage(
        model_name="nvidia/parakeet-tdt-0.6b-v2",
        batch_size=32,
        pred_text_key="asr_transcript",
    )
    
    # Stage 2: Calculate WER
    wer_stage = GetPairwiseWerStage(
        text_key=reference_field,        # Reference transcript field
        pred_text_key="asr_transcript",  # ASR output field
        wer_key="wer",                   # Output field for WER score
    )
    
    # Stage 3: Filter by WER (keep entries where WER < threshold)
    filter_stage = PreserveByValueStage(
        input_value_key="wer",
        target_value=max_wer,
        operator="lt",  # Less than
    )
    
    pipeline = Pipeline(
        name="wer_filtering",
        stages=[asr_stage, wer_stage, filter_stage],
    )
    
    results = pipeline.run()
    print(f"Kept {len(results)} utterances with WER < {max_wer}%")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-wer", type=float, default=15.0, help="Max WER as percentage (e.g., 15.0 = 15%%)")
    parser.add_argument("--reference-field", default="text")
    args = parser.parse_args()
    filter_by_wer(args.input, args.output, args.max_wer, args.reference_field)
```

## Stage Parameters

### GetPairwiseWerStage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_key` | str | `"text"` | Field containing reference transcript |
| `pred_text_key` | str | `"pred_text"` | Field containing ASR prediction |
| `wer_key` | str | `"wer"` | Output field for WER score |

### PreserveByValueStage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_value_key` | str | required | Field to compare |
| `target_value` | int/str | required | Value to compare against |
| `operator` | str | `"eq"` | Comparison operator: "lt", "le", "eq", "ne", "ge", "gt" |

## WER Thresholds

| Threshold | Description | Use Case |
|-----------|-------------|----------|
| 5.0 (5%) | Very strict | Near-perfect transcripts |
| 10.0 (10%) | Strict | Professional quality |
| 15.0 (15%) | Standard | Good quality |
| 20.0 (20%) | Permissive | Acceptable quality |
| 30.0 (30%) | Lenient | Include noisy data |

## Manifest Format

Input manifest must have reference transcripts:

```json
{"audio_filepath": "/path/to/audio.wav", "duration": 5.2, "text": "reference transcript here"}
```

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd):/data \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/wer_pipeline.py \
    --input /data/manifest.json \
    --output /data/filtered.json \
    --max-wer 15.0
```

## Interpreting Results

- High WER but good audio → Bad reference transcript
- High WER with noisy audio → Low-quality recording
- Low WER → High-quality pair, keep it
