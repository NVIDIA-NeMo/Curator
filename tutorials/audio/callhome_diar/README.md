# Speaker Diarization on CallHome English with NeMo Curator

This tutorial runs [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) speaker diarization on the [CallHome English](https://catalog.ldc.upenn.edu/LDC97S42) dataset using NeMo Curator's `InferenceSortformerStage`, then evaluates Diarization Error Rate (DER).

Inference runs in parallel via `Pipeline` + `XennaExecutor` for high throughput.

## Prerequisites

- Python 3.10+
- NeMo Curator installed (`pip install -e .` from the Curator repo)
- For XennaExecutor GPU stages: `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0` must be set before any Ray cluster starts. The included `run.py` sets this automatically; if you launch the pipeline from another script or Slurm, set this env var in the process that starts the job (e.g. `export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0` before running Python).
- NeMo ASR toolkit (`pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'`)
- `sox` command-line tool (for stereo-to-mono conversion)
- CallHome English dataset with `.wav` files and `eng/*.cha` ground-truth annotations

### Dataset layout

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

### Quick start

```bash
# run.py sets RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0 automatically; if you see "no GPUs are available" on GPU workers, set it before Python:
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0
python tutorials/audio/callhome_diar/run.py \
  --data-dir /path/to/callhome_eng0
```

### Full options

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
| `--model` | `nvidia/diar_streaming_sortformer_4spk-v2` | Hugging Face model id |

### Streaming configuration

All values are in **80 ms frames**. Override via `--chunk-len`, `--chunk-right-context`, etc.

| Configuration | Latency | chunk_len | chunk_right_context | fifo_len | spkcache_update_period | spkcache_len |
|---------------|---------|-----------|---------------------|----------|------------------------|--------------|
| Very high (default) | 30.4 s | 340 | 40 | 40 | 300 | 188 |
| High | 10.0 s | 124 | 1 | 124 | 124 | 188 |
| Low | 1.04 s | 6 | 7 | 188 | 144 | 188 |
| Ultra low | 0.32 s | 3 | 1 | 188 | 144 | 188 |

## What the script does

1. **File discovery (`CallHomeReaderStage`)** — Scans the dataset directory for WAV files with matching `.cha` annotations, skipping already-processed files. Emits one `AudioBatch` per file.
2. **Mono conversion (`EnsureMonoStage`)** — CallHome WAVs are stereo (one channel per speaker). This stage downmixes to mono 16 kHz via `sox` so the model sees both speakers.
3. **Diarization inference (`InferenceSortformerStage`)** — Runs Streaming Sortformer on each mono file. Also writes RTTM files to `--rttm-out-dir`.
4. **DER evaluation (`DERComputationStage`)** — Compares predicted segments against CHA ground truth. Scoring is restricted to the UEM region (min/max annotated timestamps from CHA) with a configurable collar tolerance (default 0.25 s).

`XennaExecutor` distributes tasks across workers for parallel processing. After the pipeline completes, the script prints macro-average, weighted-average, speaker count accuracy, and best/worst files.

## Example output

```
COMPLETED: 139 files evaluated (collar=0.25s)

--- Macro-Average (equal weight per file) ---
  DER:     6.2%
  Miss:    1.5%
  FA:      3.4%
  Confuse: 1.3%
  Correct: 97.2%

--- Speaker Count ---
  Exact match: 109/139 (78%)
```

## Pipeline integration

`InferenceSortformerStage` can be composed with any reader stage in a NeMo Curator pipeline:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage

pipeline = Pipeline(
    name="diarization",
    stages=[
        MyAudioReaderStage(data_dir="/path/to/audio"),  # your reader stage
        InferenceSortformerStage(
            model_name="nvidia/diar_streaming_sortformer_4spk-v2",
            rttm_out_dir="./rttm",
        ),
    ],
)

results = pipeline.run(executor=XennaExecutor())
```

## Model limitations

- Maximum 4 speakers per recording
- Trained primarily on English speech
- Performance may degrade on noisy or very long recordings
