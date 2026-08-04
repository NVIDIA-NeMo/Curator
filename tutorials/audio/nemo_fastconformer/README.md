# NeMo FastConformer ASR through the shared adapter

This is the smallest runnable NeMo FastConformer pipeline in Curator. It reads
a NeMo-style JSONL manifest, resamples every file to 16 kHz mono, transcribes
with the generic `ASRStage` configured with `NeMoASRAdapter`, and writes the
rows back to JSONL.

There is no FastConformer-specific Curator stage. `ASRStage` owns task I/O,
audio preparation, batching, and output assembly; `NeMoASRAdapter` owns NeMo
checkpoint download, model lifecycle, and inference. Use this split for all
ordinary NeMo ASR transcription pipelines. The specialized forced-alignment
stage remains separate because it emits word timestamps rather than only a
transcript.

## Requirements

- x86_64 Linux
- `ffmpeg`
- NeMo Curator's audio dependencies
- One CUDA GPU is recommended; CPU inference is supported for a small smoke run

From the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

For CPU-only use, install `audio_cpu` instead.

## Run the bundled smoke input

The repository already contains a two-row manifest and two short OPUS files:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/nemo_fastconformer \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl \
  output_path=/tmp/nemo_fastconformer_output.jsonl \
  workspace_dir=/tmp/nemo_fastconformer_workspace
```

`--config-path` is relative to `nemo_curator/config/run.py`; manifest and
output paths are relative to the current working directory. Run the command
from the repository root as shown. The first run downloads
`nvidia/stt_en_fastconformer_ctc_large`.

For a CPU smoke run, append `gpus_per_actor=0`. CPU execution is much slower
than GPU execution.

## Input and output

Each input row must contain `audio_filepath`:

```json
{"audio_filepath": "/data/sample.opus"}
```

`ResampleAudioStage` preserves the source path and adds
`resampled_audio_filepath`, `audio_item_id`, and `duration`. It caches
16-bit, 16 kHz, mono WAV files under `workspace_dir`.

`ASRStage` opens only the resampled files in its current batch and supplies
contiguous mono waveforms to `NeMoASRAdapter`. It adds `pred_text` by
default. The writer produces one JSON object per input row at `output_path`.
Inspect `_skipme` and `additional_notes` before consuming output; Curator
uses those fields to retain and explain rows that could not be transcribed.

## Useful overrides

| Setting | Default | Purpose |
|---|---|---|
| `model_id` | `nvidia/stt_en_fastconformer_ctc_large` | Any compatible pretrained NeMo ASR checkpoint |
| `pred_text_key` | `pred_text` | Output transcript column |
| `gpus_per_actor` | `1` | GPUs scheduled for each ASR worker; set `0` for CPU |
| `stages.2.batch_size` | `16` | Number of waveforms per NeMo transcription call |
| `stages.2.adapter_kwargs.num_workers` | `0` | NeMo transcription data-loader workers |
| `stages.2.adapter_kwargs.enable_local_attention` | `false` | Convert a compatible FastConformer checkpoint to local attention |

When local attention is enabled, configure its left/right context with:

```bash
stages.2.adapter_kwargs.enable_local_attention=true \
'stages.2.adapter_kwargs.local_attention_context_size=[128,128]'
```

## Use the adapter in Python

```python
from nemo_curator.stages.audio.inference.asr.stage import ASRStage

asr = ASRStage(
    adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
    model_id="nvidia/stt_en_fastconformer_ctc_large",
    audio_filepath_key="audio_filepath",
    batch_size=16,
)
```

Normally place a `ResampleAudioStage` before this stage and keep the default
`audio_filepath_key="resampled_audio_filepath"`. The direct-file form above
is useful when upstream data is already readable; `ASRStage` still normalizes
audio to the configured `target_sample_rate` before calling the adapter.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ffmpeg` is not found | Install `ffmpeg` and ensure it is on `PATH` |
| CUDA out of memory | Reduce `stages.2.batch_size` or select a smaller checkpoint |
| Model import fails | Install `audio_cuda12` or `audio_cpu` for your platform |
| First run appears idle | Wait for the NeMo checkpoint download and inspect the Ray logs |
| Local-attention conversion fails | Disable it or use a FastConformer checkpoint exposing the required conversion APIs |
