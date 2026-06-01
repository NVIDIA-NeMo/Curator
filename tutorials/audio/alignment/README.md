# MFA Forced Alignment Pipeline

Forced alignment of audio with transcripts using the [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/).

This pipeline takes a JSONL audio manifest (audio files + transcripts), runs MFA batch alignment, and produces word-level TextGrid files with optional RTTM (speech activity) and CTM (word timing) outputs.

## What is MFA?

Montreal Forced Aligner is a tool that aligns orthographic transcriptions to audio recordings, producing **word-level** and **phone-level** time boundaries stored in Praat TextGrid files.

The `MFAAlignmentStage` wraps MFA as a NeMo Curator processing stage, enabling:

- **Batch alignment** -- groups of audio files are aligned in a single `mfa align` call for efficiency
- **TextGrid output** -- the native MFA alignment format
- **RTTM output** -- speech activity segments derived from word boundaries (useful for diarization pipelines)
- **CTM output** -- word-level timing in NIST CTM format (useful for ASR evaluation)

## Prerequisites

### 1. Install NeMo Curator with alignment dependencies

```bash
# Using uv (recommended)
uv sync --extra audio_alignment

# Using pip
pip install "nemo-curator[audio_alignment]"
```

This installs `praatio` (for TextGrid parsing) and `soundfile`.

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

If MFA is in a separate conda environment, provide the full path to the binary via `--mfa-command`:

```bash
--mfa-command /path/to/micromamba/envs/mfa/bin/mfa
```

### 3. Download MFA models

```bash
# Acoustic model + pronunciation dictionary (English example)
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Optional: G2P model for out-of-vocabulary words
mfa model download g2p english_us_arpa
```

Models are stored under `~/.mfa/pretrained_models/` by default. Override with `--mfa-root-dir` or the `MFA_ROOT_DIR` environment variable.

## Quick Start

```bash
# Basic alignment with RTTM + CTM output
python tutorials/audio/alignment/pipeline.py \
    --input-manifest /data/manifest.jsonl \
    --output-dir /data/aligned

# TextGrid-only output (no RTTM/CTM conversion)
python tutorials/audio/alignment/pipeline.py \
    --input-manifest /data/manifest.jsonl \
    --output-dir /data/aligned \
    --no-rttm --no-ctm

# Custom MFA binary and models
python tutorials/audio/alignment/pipeline.py \
    --input-manifest /data/manifest.jsonl \
    --output-dir /data/aligned \
    --mfa-command /opt/micromamba/envs/mfa/bin/mfa \
    --mfa-root-dir /shared/mfa_models \
    --acoustic-model english_us_arpa \
    --dictionary english_us_arpa
```

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

The key names are configurable via `--text-key`, `--audio-filepath-key`, and `--speaker-key`.

## Pipeline Architecture

```
Input JSONL Manifest
    |
    v
MFAAlignmentStage (process_batch)
    |-- Prepares temporary corpus (symlinked WAVs + .txt files)
    |-- Runs single `mfa align` subprocess
    |-- Parses resulting TextGrid files
    |-- Converts to RTTM (if create_rttm=True)
    |-- Converts to CTM (if create_ctm=True)
    |-- Adds output paths to task.data
    |
    v
AudioToDocumentStage
    |
    v
JsonlWriter -> Output JSONL Manifest
```

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-manifest` | *required* | Path to input JSONL manifest |
| `--output-dir` | *required* | Root output directory |
| `--mfa-command` | `mfa` | Path to the MFA binary |
| `--mfa-root-dir` | `~/.mfa` | MFA root directory with pretrained models |
| `--acoustic-model` | `english_us_arpa` | MFA acoustic model name or path |
| `--dictionary` | `english_us_arpa` | MFA dictionary name or path |
| `--g2p-model` | `english_us_arpa` | MFA G2P model (empty string to disable) |
| `--text-key` | `text` | Manifest key for transcript text |
| `--audio-filepath-key` | `audio_filepath` | Manifest key for audio file path |
| `--speaker-key` | `speaker` | Manifest key for speaker label |
| `--beam` | `100` | MFA beam size |
| `--retry-beam` | `400` | MFA retry beam for failed alignments |
| `--num-jobs` | `0` (auto) | Parallel MFA jobs (0 = CPU count) |
| `--batch-size` | `256` | Files per `mfa align` invocation |
| `--no-rttm` | `false` | Skip RTTM generation |
| `--no-ctm` | `false` | Skip CTM generation |
| `--backend` | `ray_data` | Execution backend (`ray_data` or `xenna`) |
| `--clean` | `false` | Overwrite existing result directory |
| `--verbose` | `false` | Enable DEBUG logging |

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
  "ctm_filepath": "/data/aligned/ctms/utt001.ctm"
}
```

### Output directory structure

```
output_dir/
├── textgrids/          # MFA TextGrid alignments (subdirs per batch)
│   └── <batch_uuid>/
│       └── utt001.TextGrid
├── rttms/              # RTTM speech activity files (if enabled)
│   └── utt001.rttm
├── ctms/               # CTM word timing files (if enabled)
│   └── utt001.ctm
└── result/             # Output JSONL manifest
    └── *.jsonl
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

## Using with Hydra

```bash
python tutorials/audio/alignment/run.py \
    --config-path=. --config-name=pipeline \
    output_dir=/data/aligned
```

See `pipeline.yaml` for all configurable parameters. Override any field from the command line:

```bash
python tutorials/audio/alignment/run.py \
    --config-path=. --config-name=pipeline \
    output_dir=/data/aligned \
    processors.0.acoustic_model=english_mfa \
    processors.0.dictionary=english_mfa \
    batch_size=512
```

## Multi-Node / Distributed Execution

When running on multiple nodes (e.g., via Xenna or Ray cluster), `MFAAlignmentStage` handles distributed MFA gracefully:

- **`setup_on_node()`** copies MFA pretrained models from shared storage (NFS/Lustre) to each node's local storage (e.g., `/tmp`). This avoids file-locking issues that Kaldi (used internally by MFA) has with network filesystems.
- **`xenna_stage_spec()`** requests exactly 1 MFA worker per node, since MFA itself uses internal parallelism via `--num-jobs`.
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
python tutorials/audio/alignment/pipeline.py \
    --input-manifest /data/german_manifest.jsonl \
    --output-dir /data/aligned_de \
    --acoustic-model german_mfa \
    --dictionary german_mfa \
    --g2p-model german_mfa
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
| `mfa: command not found` | Provide the full path via `--mfa-command /path/to/mfa` |
| `praatio` import error | Install with `pip install 'praatio>=6.0'` or `uv sync --extra audio_alignment` |
| `Kaldi error: cannot lock file` | Enable `copy_models_to_local=True` (default) or use local storage for `--mfa-root-dir` |
| Many files silently skipped | Check for OOV words; provide a G2P model or expand the dictionary |
| `mfa align` OOM | Reduce `--batch-size` to process fewer files per invocation |
| Slow alignment | Increase `--num-jobs` or ensure MFA has access to all CPU cores |

## License

This tutorial and the `MFAAlignmentStage` are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

MFA itself is licensed under the [MIT License](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/LICENSE).
