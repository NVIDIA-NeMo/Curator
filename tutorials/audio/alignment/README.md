# MFA Forced Alignment Pipeline

Forced alignment of audio with transcripts using the [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/).

## Overview

This tutorial takes a JSONL audio manifest (audio files + transcripts), runs MFA batch alignment, and produces word-level TextGrid files with optional RTTM (speech activity) and CTM (word timing) outputs.

Montreal Forced Aligner is a tool that aligns orthographic transcriptions to audio recordings, producing **word-level** and **phone-level** time boundaries stored in Praat TextGrid files. The `MFAAlignmentStage` wraps MFA as a NeMo Curator processing stage, enabling:

- **Batch alignment** -- groups of audio files are aligned in a single `mfa align` call for efficiency
- **TextGrid output** -- the native MFA alignment format
- **RTTM output** -- speech activity segments derived from word boundaries (useful for diarization pipelines)
- **CTM output** -- word-level timing in NIST CTM format (useful for ASR evaluation)

The tutorial uses the generic YAML runner in `nemo_curator/config/run.py`; no
tutorial-specific Python runner is needed. The YAML selects the executor
backend and, for Xenna, its execution mode; the generic runner constructs that
executor and passes it to `Pipeline.run()`.

### Pipeline flow

```
┌──────────────┐    ┌───────────────────┐    ┌────────────────┐
│ManifestReader│───▶│ MFAAlignmentStage │───▶│ ManifestWriter │
│ (turn JSONL) │    │ (align + convert) │    │ (result.jsonl) │
└──────────────┘    └───────────────────┘    └────────────────┘
  manifest input      TextGrid/RTTM/CTM        enriched output
```

## Prerequisites

### 1. Install NeMo Curator with alignment dependencies

```bash
uv sync --extra audio_cuda12
```

This installs `praatio` (for TextGrid parsing) and other audio dependencies via `audio_common`.

### 2. Install Montreal Forced Aligner

MFA is distributed via conda/micromamba (not pip). Install it in a separate environment:

```bash
# Using micromamba
micromamba create -n mfa -c conda-forge montreal-forced-aligner
micromamba activate mfa

# Or using conda
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
```

If MFA is in a separate conda environment, provide the full path to the binary via `mfa_command`:

```bash
mfa_command=/path/to/micromamba/envs/mfa/bin/mfa
```

### 3. Download MFA models

```bash
# Acoustic model + pronunciation dictionary (English example)
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Optional: G2P model for out-of-vocabulary words
mfa model download g2p english_us_arpa
```

Models are stored under `~/.mfa/pretrained_models/` by default. Override with `mfa_root_dir` or the `MFA_ROOT_DIR` environment variable.

## Input Format

The pipeline expects a JSONL manifest where each line is a JSON object with at least:

```json
{"audio_filepath": "/data/audio/utt001.wav", "text": "hello world", "speaker": "speaker_a"}
```

| Key | Required | Description |
|-----|----------|-------------|
| `audio_filepath` | Yes | Path to the WAV audio file |
| `text` | Yes | Transcript text for alignment |
| `speaker` | No | Speaker label (used in RTTM output; defaults to `"unknown"`) |
| `duration` | No | Audio duration in seconds (computed automatically if missing) |

The key names are configurable via `text_key`, `audio_filepath_key`, and `speaker_key`.

## Quick Start

From the Curator repository root:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/alignment \
  --config-name pipeline \
  input_manifest=/data/manifest.jsonl \
  output_dir=/data/aligned
```

`--config-path` is relative to `nemo_curator/config/run.py`, while manifest
and output paths are resolved from the current working directory. Run the
command from the repository root as shown.

See `pipeline.yaml` for all configurable parameters. Override any top-level field from the command line:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/alignment \
  --config-name pipeline \
  input_manifest=/data/manifest.jsonl \
  output_dir=/data/aligned \
  acoustic_model=english_mfa \
  dictionary=english_mfa \
  batch_size=512
```

Results are written to `/data/aligned/result.jsonl`.

## All configurable settings (`pipeline.yaml`)

| Setting | Default | Description |
|---|---|---|
| `input_manifest` | *(required)* | Path to input JSONL manifest |
| `output_dir` | *(required)* | Root output directory; result manifest is `${output_dir}/result.jsonl` |
| `backend` | `ray_data` | Execution backend: `xenna` or `ray_data` |
| `batch_size` | `256` | Files aligned per `mfa align` invocation (`MFAAlignmentStage.batch_size`) |
| `acoustic_model` | `english_us_arpa` | MFA acoustic model name or path |
| `dictionary` | `english_us_arpa` | MFA dictionary name or path |
| `g2p_model` | `english_us_arpa` | MFA G2P model (empty string to disable) |
| `num_jobs` | `1` | Parallel MFA jobs passed to MFA `-j`; also sizes the stage's CPU reservation |
| `beam` | `100` | MFA beam size |
| `retry_beam` | `400` | MFA retry beam for failed alignments |
| `align_timeout_seconds` | `3600.0` | Hard timeout for each `mfa align` subprocess |
| `create_rttm` | `true` | Whether to generate RTTM speech-activity files |
| `create_ctm` | `true` | Whether to generate CTM word-timing files |

Override any setting on the command line, e.g. `num_jobs=8`.

## Choosing a backend

| Backend | Description | When to use |
|---|---|---|
| `ray_data` | Default for this tutorial. `MFAAlignmentStage` forces a single actor cluster-wide (`num_workers()` returns `1`) so MFA/Kaldi never runs concurrently against a shared model directory. | **Recommended.** |
| `xenna` | Cosmos-Xenna streaming engine. `xenna_stage_spec()` requests exactly one MFA worker per node instead. | Multi-node runs where Xenna's per-node scheduling is preferred. |

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/alignment \
  --config-name pipeline \
  input_manifest=/data/manifest.jsonl \
  output_dir=/data/aligned \
  backend=xenna \
  execution_mode=streaming
```

`execution_mode` applies only to Xenna and is ignored when `backend` is `ray_data`.

## Pipeline stages

### Stage 1: `ManifestReader`

Reads the JSONL manifest line-by-line and emits one `AudioTask` per entry (no Pandas; ~1x file-size memory).

### Stage 2: `MFAAlignmentStage`

- Prepares a temporary MFA corpus (symlinked WAVs + `.txt` transcript files) per batch
- Runs a single `mfa align` subprocess per batch, bounded by `align_timeout_seconds`
- Parses resulting TextGrid files
- Converts to RTTM (if `create_rttm=true`) and CTM (if `create_ctm=true`)
- Adds output paths to `task.data`

A per-task pre-flight failure (invalid input, empty text, missing file) marks that entry with `mfa_skipped=true` and empty fallback outputs instead of aborting the whole batch, so one malformed manifest row never discards the rest of a batch.

### Stage 3: `ManifestWriterStage`

Appends each enriched entry to `${output_dir}/result.jsonl`. The output file is truncated once when the pipeline starts, so re-running with an unchanged manifest overwrites the previous result instead of accumulating shards.

## Output Format

The output manifest JSONL contains all original fields plus:

```json
{
  "audio_filepath": "/data/audio/utt001.wav",
  "text": "hello world",
  "speaker": "speaker_a",
  "duration": 1.23,
  "textgrid_filepath": "/data/aligned/textgrids/abc123/utt001.TextGrid",
  "rttm_filepath": "/data/aligned/rttms/utt001.rttm",
  "ctm_filepath": "/data/aligned/ctms/utt001.ctm",
  "mfa_skipped": false
}
```

| Field | Type | Description |
|---|---|---|
| `textgrid_filepath` | string | Path to the TextGrid alignment file (empty if MFA skipped this file) |
| `rttm_filepath` | string | Path to the RTTM speech-activity file (if `create_rttm=true`) |
| `ctm_filepath` | string | Path to the CTM word-timing file (if `create_ctm=true`) |
| `mfa_skipped` | bool | `true` if MFA silently dropped this file (see below) |
| `duration` | float | Audio duration in seconds (computed if missing from input) |

### Output directory structure

```
output_dir/
├── textgrids/          # MFA TextGrid alignments (subdirs per batch)
│   └── <batch_uuid>/
│       └── utt001.TextGrid
├── rttms/              # RTTM speech activity files (if enabled)
│   └── utt001.rttm
├── ctms/                # CTM word timing files (if enabled)
│   └── utt001.ctm
└── result.jsonl         # Output JSONL manifest
```

### RTTM format

```
SPEAKER utt001 1 0.120 0.890 <NA> <NA> speaker_a <NA> <NA>
```

Fields: `SPEAKER <file-id> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>`

Nearby speech intervals are merged when separated by less than `max_gap_for_merge` seconds (default 0.3s).

### CTM format

```
utt001 1 0.120 0.380 hello
utt001 1 0.510 0.390 world
```

Fields: `<file-id> <channel> <start> <duration> <word>`

## Multi-Node / Distributed Execution

When running on multiple nodes (e.g., via Xenna or Ray cluster), `MFAAlignmentStage` handles distributed MFA gracefully:

- **`setup_on_node()`** copies MFA pretrained models from shared storage (NFS/Lustre) to each node's local storage (e.g., `/tmp`), namespaced by a digest of the resolved source root plus the requested acoustic/dictionary/G2P model identity. The cache is populated atomically under a lock and validated against a completeness marker before reuse, so a stale, wrong-source, or interrupted copy is never silently served. This avoids file-locking issues that Kaldi (used internally by MFA) has with network filesystems.
- **`xenna_stage_spec()`** requests exactly 1 MFA worker per node under Xenna; the Ray Data backend instead forces a single MFA actor cluster-wide via `num_workers()`. Either way, MFA's internal parallelism is controlled via `num_jobs` (MFA `-j`).
- Set `copy_models_to_local=False` if MFA models are already on local storage.

## Non-English Languages

MFA supports [many languages](https://mfa-models.readthedocs.io/en/latest/). To align a different language:

1. Download the appropriate models:

```bash
mfa model download acoustic german_mfa
mfa model download dictionary german_mfa
mfa model download g2p german_mfa
```

2. Pass them to the pipeline:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/alignment \
  --config-name pipeline \
  input_manifest=/data/german_manifest.jsonl \
  output_dir=/data/aligned_de \
  acoustic_model=german_mfa \
  dictionary=german_mfa \
  g2p_model=german_mfa
```

## MFA-Skipped Files

MFA may silently skip files it cannot align (out-of-vocabulary words, acoustic mismatch, very short audio, etc.). When this happens:

- The stage creates **fallback** RTTM/CTM files (duration-based: one segment spanning the full audio)
- The entry is marked with `"mfa_skipped": true` in the output manifest
- `"textgrid_filepath"` is set to an empty string

You can filter these entries downstream or audit them separately.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `mfa: command not found` | Provide the full path via `mfa_command=/path/to/mfa` |
| `praatio` import error | Run `uv sync --extra audio_cuda12` (or `audio_cpu`) from the Curator repo root |
| `Kaldi error: cannot lock file` | Enable `copy_models_to_local=true` (default) or use local storage for `mfa_root_dir` |
| Many files silently skipped | Check for OOV words; provide a G2P model or expand the dictionary |
| `mfa align` OOM | Reduce `batch_size` to process fewer files per invocation |
| Slow alignment | Increase `num_jobs` or ensure MFA has access to all CPU cores |
| A single `mfa align` subprocess hangs | Reduce `align_timeout_seconds` to fail fast; the whole process group is killed on expiry |

## License

This tutorial and the `MFAAlignmentStage` are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

MFA itself is licensed under the [MIT License](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/LICENSE).
