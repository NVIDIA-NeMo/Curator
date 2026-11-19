# Single-Speaker Filtering with Streaming Sortformer

Filter an ASR manifest to keep only audio files containing exactly one speaker, using [Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) for diarization with NeMo Curator's adapter-backed `InferenceSortformerStage`.

## Prerequisites

- Python 3.11+
- NeMo Curator installed (see the [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- NVIDIA GPU(s) with at least 8 GB VRAM (e.g., V100, A100, H100, or RTX 3090+)

## Input Format

NeMo-style JSONL manifest — one JSON object per line with at least `audio_filepath`:

```json
{"text": "the cat sat on a mat", "audio_filepath": "/data/file1.wav"}
{"text": "hello world", "audio_filepath": "/data/file2.wav", "duration": 3.2}
```

All fields are preserved in the output; extra fields (e.g. `duration`) pass through unchanged.

## Usage

From the repository root:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/single_speaker_filter \
  --config-name pipeline \
  manifest_path=/path/to/manifest.jsonl \
  output_path=/path/to/output/filtered_manifest.jsonl
```

`--config-path` is relative to `nemo_curator/config/run.py`. An interrupted run
starts again from the beginning.

### Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `manifest_path` | *(required)* | Input JSONL manifest |
| `output_path` | `./single_speaker_manifest.jsonl` | Filtered output manifest |
| `model_id` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Hugging Face Sortformer model ID |
| `chunk_len` | `340` | Streaming chunk size (80ms frames) |
| `chunk_right_context` | `40` | Right context frames |
| `fifo_len` | `40` | FIFO queue size in frames |
| `spkcache_update_period` | `300` | Speaker cache update period in frames |
| `spkcache_len` | `188` | Speaker cache size in frames |
| `backend` | `xenna` | Execution backend: `xenna` or `ray_data` |

## Pipeline Stages

1. **ManifestReader** — Reads the JSONL manifest and emits one `AudioTask` per entry.
2. **InferenceSortformerStage** — The task-facing stage loads and normalizes each whole recording, then calls `NeMoSortformerAdapter` on the GPU. It adds `diar_segments` and `num_speakers` to each task.
3. **PreserveByValueStage** — Keeps only entries whose `num_speakers` equals 1; multi-speaker and zero-speaker entries are dropped.
4. **ManifestWriterStage** — Writes the surviving entries to the output JSONL manifest.

## Output

`output_path` — same JSONL format as input, with the added `diar_segments` and `num_speakers` fields:

```json
{"text": "the cat sat on a mat", "audio_filepath": "/data/file1.wav", "diar_segments": [{"start": 0.0, "end": 2.4, "speaker": "speaker_0"}], "num_speakers": 1}
```

The stage keeps Curator fields, errors, and output assembly separate from the
model implementation. Streaming model settings go in the stage's
`adapter_kwargs`; GPU allocation and batching go in its `resources` and
`batch_size`. To cap the number of Sortformer workers, add
`+stages.1.num_workers=N`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SIGSEGV / actor crash during model load | See [Known Issues](../README.md#known-issues) — set `OTEL_SDK_DISABLED=true` |

## Streaming Configuration

All frame values are in 80ms units. See the [callhome_diar tutorial](../callhome_diar/README.md) for latency trade-off configurations.

Each manifest row remains one whole-recording diarization item. Do not split a
recording into arbitrary chunks before Sortformer: speaker clustering and
speaker labels must remain coherent across the recording. Independent rows
may still be batched and reordered internally; outputs are scattered back to
their original row order.

## Model Limitations

- Maximum 4 speakers per recording
- Trained primarily on English speech
- Performance may degrade on noisy or very long recordings
