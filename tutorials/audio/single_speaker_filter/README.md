# Single-Speaker Filtering with Streaming Sortformer

Filter an ASR manifest to keep only audio files containing exactly one speaker, using [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) for diarization via NeMo Curator's `InferenceSortformerStage`.

The pipeline includes **per-task hash-based checkpointing** — if a run is interrupted, re-running with the same `--output-dir` resumes from where it left off.

## Prerequisites

- Python 3.11+
- NeMo Curator installed (see [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- NVIDIA GPU(s) with at least 8 GB VRAM (e.g. V100, A100, H100, or RTX 3090+)

## Input format

NeMo-style JSONL manifest — one JSON object per line with at least `audio_filepath`:

```json
{"text": "the cat sat on a mat", "audio_filepath": "/data/file1.wav"}
{"text": "hello world", "audio_filepath": "/data/file2.wav", "duration": 3.2}
```

All fields are preserved in the output; extra fields (e.g. `duration`) pass through unchanged.

## Usage

```bash
python tutorials/audio/single_speaker_filter/run.py \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output
```

### Resume after failure

Simply re-run with the same `--output-dir`. Tasks whose inference checkpoint already exists are skipped:

```bash
python tutorials/audio/single_speaker_filter/run.py \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output
```

### Clean run

```bash
python tutorials/audio/single_speaker_filter/run.py \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output --clean
```

### All options

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | *(required)* | Input JSONL manifest |
| `--output-dir` | `output` | Root for checkpoints and filtered manifest |
| `--model` | `nvidia/diar_streaming_sortformer_4spk-v2` | HF Sortformer model id |
| `--clean` | off | Remove output directory before running |
| `--chunk-len` | `340` | Streaming chunk size (80ms frames) |
| `--chunk-right-context` | `40` | Right context frames |
| `--fifo-len` | `40` | FIFO queue size in frames |
| `--spkcache-update-period` | `300` | Speaker cache update period in frames |
| `--spkcache-len` | `188` | Speaker cache size in frames |

## Pipeline stages

1. **ALMManifestReader** — Reads the JSONL manifest and emits one `AudioTask` per entry.
2. **InferenceSortformerStage** — Runs Streaming Sortformer on each audio file (GPU). Adds `diar_segments` to each task.
3. **SingleSpeakerFilterStage** — Counts unique speakers from `diar_segments`. Keeps only entries with exactly 1 speaker; multi-speaker or zero-speaker entries produce an empty task (no output rows).
4. **ALMManifestWriterStage** — Writes the surviving entries to the output JSONL manifest.

## Output

`<output-dir>/filtered_manifest.jsonl` — same JSONL format as input, with an added `num_speakers` field:

```json
{"text": "the cat sat on a mat", "audio_filepath": "/data/file1.wav", "num_speakers": 1}
```

`<output-dir>/checkpoints/` — per-stage checkpoint directories for resume support.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SIGSEGV / actor crash during model load | See [Known Issues](../README.md#known-issues) — set `OTEL_SDK_DISABLED=true` |

## Streaming configuration

All frame values are in 80ms units. See the [callhome_diar tutorial](../callhome_diar/README.md) for latency trade-off configurations.

## Model limitations

- Maximum 4 speakers per recording
- Trained primarily on English speech
- Performance may degrade on noisy or very long recordings
