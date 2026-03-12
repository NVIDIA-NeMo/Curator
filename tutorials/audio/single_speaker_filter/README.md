# Single-Speaker Filtering with Streaming Sortformer

Filter an ASR manifest to keep only audio files containing exactly one speaker, using [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) for diarization via NeMo Curator's `InferenceSortformerStage`.

The pipeline includes **per-task hash-based checkpointing** — if a run is interrupted, re-running with the same `--output-dir` resumes from where it left off.

## Prerequisites

- Python 3.10+
- NeMo Curator installed (`pip install -e .` from the Curator repo)
- NeMo ASR toolkit (`pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'`)
- GPU(s) for Sortformer inference
- `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0` set before the Ray cluster starts (see below)

## Input format

NeMo-style JSONL manifest — one JSON object per line with at least `audio_filepath`:

```json
{"text": "the cat sat on a mat", "audio_filepath": "/data/file1.wav"}
{"text": "hello world", "audio_filepath": "/data/file2.wav", "duration": 3.2}
```

All fields are preserved in the output; extra fields (e.g. `duration`) pass through unchanged.

## Usage

```bash
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0

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

1. **ManifestReaderStage** — Reads the JSONL manifest and emits one `AudioBatch` per entry.
2. **InferenceSortformerStage** — Runs Streaming Sortformer on each audio file (GPU). Adds `diar_segments` to each task.
3. **SingleSpeakerFilterStage** — Counts unique speakers from `diar_segments`. Keeps only entries with exactly 1 speaker; multi-speaker or zero-speaker entries produce an empty task (no output rows).

## Output

`<output-dir>/filtered_manifest.jsonl` — same JSONL format as input, with an added `num_speakers` field:

```json
{"text": "the cat sat on a mat", "audio_filepath": "/data/file1.wav", "num_speakers": 1}
```

`<output-dir>/checkpoints/` — per-stage checkpoint directories for resume support.

## Streaming configuration

All frame values are in 80ms units. See the [callhome_diar tutorial](../callhome_diar/README.md) for latency trade-off configurations.

## Model limitations

- Maximum 4 speakers per recording
- Trained primarily on English speech
- Performance may degrade on noisy or very long recordings
