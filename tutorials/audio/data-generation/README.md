# Multi-Speaker Conversation Data Generation

End-to-end pipeline for generating synthetic multi-speaker conversation audio
using NeMo Curator. Chains LLM conversation generation, TTS synthesis, forced
alignment, and SDP-style conversation merging.

## Pipeline Overview

```
Topics JSONL
    │
    ▼
┌──────────────────────┐
│  vLLMInference       │  GPU — generates multi-turn conversations
│  (Qwen 2.5 7B)      │       from topic prompts
└──────────────────────┘
    │  AudioTask per turn
    ▼
┌──────────────────────┐
│  ChatterboxTTSStage  │  GPU — synthesizes per-turn WAV files
│  (voice cloning)     │       with reference speaker voices
└──────────────────────┘
    │  AudioTask with audio_filepath
    ▼
┌──────────────────────┐
│  MFAAlignmentStage   │  CPU — Montreal Forced Aligner produces
│  (batch alignment)   │       TextGrid → RTTM + CTM per turn
└──────────────────────┘
    │  AudioTask with RTTM/CTM
    ▼
┌──────────────────────┐
│  MergeConversation   │  CPU — merges turns into full conversations
│  SDPStage            │       with multi-channel audio + combined
│                      │       RTTM/CTM/seglst
└──────────────────────┘
    │
    ▼
Merged Manifest JSONL
```

## Prerequisites

### Software

- Python 3.10+
- NeMo Curator with audio stages installed
- Chatterbox TTS (`pip install chatterbox-tts`)
- Montreal Forced Aligner (`conda install -c conda-forge montreal-forced-aligner`)
- vLLM (`pip install vllm`)

### MFA Models

Download MFA models before running:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

### Reference Voices

Prepare a directory of reference voice WAV files. The stage supports two layouts:

1. **Dialog layout**: `wavs/<dialog>/<speaker>.wav` with optional `rttms/` siblings
2. **MLS layout**: `<speaker_id>/<book_id>/<segment>.flac`

## Quick Start

From the Curator repo root:

```bash
# 2-speaker dialogs
python tutorials/audio/data-generation/main.py \
    --config-path . \
    --config-name pipeline \
    input_manifest=tutorials/audio/data-generation/topics/topics_dialog.jsonl \
    output_dir=/data/conversations_2spk \
    reference_voices_dataset=/data/reference_voices \
    prompt_file=tutorials/audio/data-generation/prompts/dialog_prompt.yaml

# 3-speaker conversations
python tutorials/audio/data-generation/main.py \
    --config-path . \
    --config-name pipeline \
    input_manifest=tutorials/audio/data-generation/topics/topics_group.jsonl \
    output_dir=/data/conversations_3spk \
    reference_voices_dataset=/data/reference_voices \
    prompt_file=tutorials/audio/data-generation/prompts/triperson_prompt.yaml

# 4-speaker conversations
python tutorials/audio/data-generation/main.py \
    --config-path . \
    --config-name pipeline \
    input_manifest=tutorials/audio/data-generation/topics/topics_group.jsonl \
    output_dir=/data/conversations_4spk \
    reference_voices_dataset=/data/reference_voices \
    prompt_file=tutorials/audio/data-generation/prompts/fourperson_prompt.yaml
```

## Configuration

All parameters are set in `pipeline.yaml` and can be overridden from the CLI
using Hydra syntax.

### Required Parameters

| Parameter | Description |
|---|---|
| `input_manifest` | Path to JSONL file with `{"topic": "..."}` entries |
| `output_dir` | Root directory for all generated outputs |
| `reference_voices_dataset` | Path to reference voice audio files |
| `prompt_file` | Path to conversation prompt YAML |

### LLM Settings

| Parameter | Default | Description |
|---|---|---|
| `llm_model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `llm_max_model_len` | `4096` | Maximum context length |
| `llm_max_tokens` | `4096` | Maximum generation tokens |
| `llm_temperature` | `0.7` | Sampling temperature |

### TTS Settings

| Parameter | Default | Description |
|---|---|---|
| `language` | `null` | `null` for English, ISO 639-1 code for multilingual |
| `tts_sample_rate` | `24000` | Output sample rate (Hz) |
| `tts_cfg_weight` | `0.5` | Classifier-free guidance weight |
| `tts_exaggeration` | `0.5` | Emotion exaggeration (float or `[min, max]`) |
| `tts_temperature` | `0.8` | Sampling temperature |

### MFA Settings

| Parameter | Default | Description |
|---|---|---|
| `mfa_acoustic_model` | `english_us_arpa` | MFA acoustic model |
| `mfa_dictionary` | `english_us_arpa` | MFA pronunciation dictionary |
| `mfa_g2p_model` | `english_us_arpa` | MFA grapheme-to-phoneme model |

### Merge Settings

| Parameter | Default | Description |
|---|---|---|
| `max_pause_duration` | `2.0` | Maximum silence between speakers (seconds) |
| `max_intra_turn_pause` | `1.0` | Maximum silence within a turn (seconds) |

## Hydra Override Examples

```bash
# Use a larger LLM
python tutorials/audio/data-generation/main.py \
    --config-path . --config-name pipeline \
    input_manifest=topics/topics_dialog.jsonl \
    output_dir=/data/output \
    reference_voices_dataset=/data/voices \
    prompt_file=prompts/dialog_prompt.yaml \
    llm_model=Qwen/Qwen2.5-7B-Instruct-1M

# Vary TTS expressiveness per conversation
python tutorials/audio/data-generation/main.py \
    --config-path . --config-name pipeline \
    input_manifest=topics/topics_dialog.jsonl \
    output_dir=/data/output \
    reference_voices_dataset=/data/voices \
    prompt_file=prompts/dialog_prompt.yaml \
    "stages.2.exaggeration=[0.25, 0.85]"

# Use Ray Data backend
python tutorials/audio/data-generation/main.py \
    --config-path . --config-name pipeline \
    input_manifest=topics/topics_dialog.jsonl \
    output_dir=/data/output \
    reference_voices_dataset=/data/voices \
    prompt_file=prompts/dialog_prompt.yaml \
    backend=ray_data

# Multilingual (French)
python tutorials/audio/data-generation/main.py \
    --config-path . --config-name pipeline \
    input_manifest=topics/topics_dialog.jsonl \
    output_dir=/data/output_fr \
    reference_voices_dataset=/data/voices \
    prompt_file=prompts/dialog_prompt_french.yaml \
    language=fr \
    mfa_acoustic_model=french_mfa \
    mfa_dictionary=french_mfa \
    mfa_g2p_model=french_mfa
```

## Output Structure

```
output_dir/
├── audio/                          # Per-turn WAV files (from TTS)
│   └── <conversation_id>/
│       ├── turn_000_speaker_0.wav
│       ├── turn_001_speaker_1.wav
│       └── ...
├── alignment/                      # Per-turn alignment (from MFA)
│   ├── textgrids/
│   ├── rttms/
│   └── ctms/
├── conversations/                  # Merged conversations
│   └── <conversation_id>/
│       ├── speaker_0.wav           # Per-speaker audio
│       ├── speaker_1.wav
│       ├── multichannel.wav        # Multi-channel (one channel per speaker)
│       ├── mixed.wav               # Mixed mono
│       ├── speaker_0.rttm          # Per-speaker RTTM
│       ├── combined.rttm           # Combined RTTM
│       ├── combined.ctm            # Combined CTM
│       └── segments.seglst.json    # Segment list
└── merged_manifest.jsonl           # Final output manifest
```

## Pipeline Stages (Source PRs)

| Stage | Source |
|---|---|
| `vLLMInference` | Curator main branch (`nemo_curator.models.VLLMModel`) |
| `ChatterboxTTSStage` | [PR #1976](https://github.com/NVIDIA-NeMo/Curator/pull/1976) |
| `MFAAlignmentStage` | [PR #1977](https://github.com/NVIDIA-NeMo/Curator/pull/1977) |
| `MergeConversationSDPStage` | [PR #2031](https://github.com/NVIDIA-NeMo/Curator/pull/2031) |

## File Structure

```
tutorials/audio/data-generation/
├── pipeline.yaml                   # Hydra pipeline configuration
├── main.py                         # Two-phase pipeline runner
├── README.md                       # This file
├── prompts/
│   ├── dialog_prompt.yaml          # 2-speaker conversation prompt
│   ├── triperson_prompt.yaml       # 3-speaker conversation prompt
│   └── fourperson_prompt.yaml      # 4-speaker conversation prompt
└── topics/
    ├── topics_dialog.jsonl         # 25 topics for 2-speaker dialogs
    └── topics_group.jsonl          # 25 topics for 3-4 speaker discussions
```
