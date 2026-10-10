# Speaker Diarization on CallHome English with NeMo Curator

This tutorial runs [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) speaker diarization on the [CallHome English](https://catalog.ldc.upenn.edu/LDC97S42) dataset using NeMo Curator's adapter-backed `InferenceSortformerStage`, then evaluates Diarization Error Rate (DER).

Inference runs in parallel via `Pipeline` + `XennaExecutor` for high throughput.

## Prerequisites

- Python 3.11+
- NeMo Curator installed (see the [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- [`ffmpeg`](https://ffmpeg.org/) command-line tool (for stereo-to-mono conversion; pre-installed in the NeMo Curator container)
- CallHome English dataset with `.wav` files and `eng/*.cha` ground-truth annotations

### Dataset Layout

```
/path/to/callhome_eng0/
├── 0638.wav
├── 4065.wav
├── ...              # 176 WAV files total
└── eng/
    ├── 0638.cha
    ├── 4065.cha
    └── ...          # CHAT-format ground-truth annotations
```

## Usage

### Quickstart

```bash
python tutorials/audio/callhome_diar/run.py \
  --data-dir /path/to/callhome_eng0
```

### Full Options

```bash
python tutorials/audio/callhome_diar/run.py \
  --data-dir /path/to/callhome_eng0 \
  --output-dir ./output \
  --collar 0.25 \
  --clean
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | *(required)* | Path to CallHome dataset root |
| `--output-dir` | `output` | Root for RTTM files, results JSON, and checkpoints |
| `--collar` | `0.25` | Collar tolerance (seconds) for DER scoring |
| `--clean` | off | Remove entire output directory before re-running |
| `--model` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Hugging Face model id |

### Streaming Configuration

All values are in **80 ms frames**. Override using `--chunk-len`,
`--chunk-right-context`, etc. The tutorial places these model-specific values
in the NeMo adapter's `adapter_kwargs`; the stage continues to own task I/O,
whole-recording preparation, result fields, RTTM files, and resume behavior.

| Configuration | Latency | chunk_len | chunk_right_context | fifo_len | spkcache_update_period | spkcache_len |
|---------------|---------|-----------|---------------------|----------|------------------------|--------------|
| Very high (default) | 30.4 s | 340 | 40 | 40 | 300 | 188 |
| High | 10.0 s | 124 | 1 | 124 | 124 | 188 |
| Low | 1.04 s | 6 | 7 | 188 | 144 | 188 |
| Ultra low | 0.32 s | 3 | 1 | 188 | 144 | 188 |

## What the Script Does

1. **File discovery (`CallHomeReaderStage`)** — Scans the dataset directory for WAV files with matching `.cha` annotations, skipping already-processed files. Emits one `AudioTask` per file.
2. **Mono conversion (`EnsureMonoStage`)** — CallHome WAVs are stereo (one channel per speaker). This stage downmixes to mono 16 kHz via `ffmpeg` so the model sees both speakers.
3. **Diarization inference (`InferenceSortformerStage`)** — Loads each whole recording and sends it to `NeMoSortformerAdapter`. It adds `diar_segments` and `num_speakers`, and writes RTTM files to `<output-dir>/rttm/`.
4. **DER evaluation (`DERComputationStage`)** — Compares predicted segments against CHA ground truth. Scoring is restricted to the UEM region (min/max annotated timestamps from CHA) with a configurable collar tolerance (default 0.25 s).

`XennaExecutor` distributes tasks across workers for parallel processing. After the pipeline completes, the script prints macro-average, weighted-average, speaker count accuracy, and best/worst files.

## Example Output

```
============================================================
COMPLETED: 139 files evaluated (collar=0.25s)
============================================================

  Macro-avg  DER=6.2%  Miss=1.5%  FA=3.4%  Conf=1.3%
  Weighted   DER=6.0%  Miss=1.4%  FA=3.3%  Conf=1.3%
  Speaker count match: 109/139 (78%)

  Best 5: 4588=0.0%, 4601=0.0%, 4637=0.2%, 4660=0.3%, 4822=0.5%
  Worst 5: 4247=28.1%, 4325=22.4%, 4556=19.7%, 4870=18.3%, 4902=17.6%
```

## Pipeline Integration

`InferenceSortformerStage` can be composed with any reader stage in a NeMo
Curator pipeline. The stage uses the maintained v2.1 checkpoint by default;
the explicit values below show the stage-adapter boundary:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage
from nemo_curator.stages.resources import Resources

pipeline = Pipeline(
    name="diarization",
    stages=[
        MyAudioReaderStage(data_dir="/path/to/audio"),  # your reader stage
        InferenceSortformerStage(
            adapter_target=(
                "nemo_curator.models.audio.speaker_diarization.sortformer."
                "NeMoSortformerAdapter"
            ),
            model_id="nvidia/diar_streaming_sortformer_4spk-v2.1",
            rttm_out_dir="./rttm",
            adapter_kwargs={
                "chunk_len": 340,
                "chunk_right_context": 40,
                "precision": "fp32",
            },
        ).with_(
            resources=Resources(gpus=1),
            batch_size=4,
            num_workers=1,
        ),
    ],
)

results = pipeline.run(executor=XennaExecutor())
```

`batch_size` is the finite set of independent recordings offered to one stage
call. The stage may sort those recordings by duration before inference and
then restores their original order. Do not split one recording into arbitrary
chunks before this stage: speaker clustering and speaker identity require the
whole recording. The checkpoint's streaming parameters control its internal
inference behavior; they do not turn this Curator stage into a live-stream
source.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SIGSEGV / actor crash during model load | See [Known Issues](../README.md#known-issues) — set `OTEL_SDK_DISABLED=true` |

## Model Limitations

- Maximum 4 speakers per recording
- Trained primarily on English speech
- Performance may degrade on noisy or very long recordings
- The stage downmixes and resamples inputs to mono 16 kHz. This tutorial also
  materializes normalized files in `EnsureMonoStage` so the benchmark inputs
  are explicit and reusable.
