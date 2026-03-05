# Speaker Diarization on CallHome English with NeMo Curator

This tutorial runs [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) speaker diarization on the [CallHome English](https://catalog.ldc.upenn.edu/LDC97S42) dataset using NeMo Curator's `InferenceSortformerStage`, then evaluates Diarization Error Rate (DER).

Inference runs in parallel via `Pipeline` + `RayActorPoolExecutor` — each GPU is shared across multiple actor replicas for high throughput.

## Prerequisites

- Python 3.10+
- NeMo Curator installed (`pip install -e .` from the Curator repo)
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
python tutorials/audio/callhome_diar/run.py \
  --data-dir /path/to/callhome_eng0
```

### Full options

```bash
python tutorials/audio/callhome_diar/run.py \
  --data-dir /path/to/callhome_eng0 \
  --rttm-out-dir ./rttm_callhome_sortformer \
  --results-json callhome_results.json \
  --collar 0.25 \
  --clean
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | *(required)* | Path to CallHome dataset root |
| `--rttm-out-dir` | `rttm_callhome_sortformer` | Directory for RTTM output files |
| `--results-json` | `callhome_sortformer_results.json` | Per-file DER results JSON |
| `--collar` | `0.25` | Collar tolerance (seconds) for DER scoring |
| `--clean` | off | Remove existing RTTM directory before re-running |
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

1. **Mono conversion** — CallHome WAVs are stereo (one channel per speaker). The script downmixes to mono 16 kHz via `sox` so the model sees both speakers.
2. **Parallel GPU inference** — Creates one `AudioBatch` task per file. `RayActorPoolExecutor` launches actor replicas (based on `gpu_memory_gb=8.0`), each loading its own model copy. Tasks are distributed across actors for parallel diarization.
3. **RTTM output** — Each file produces an RTTM file in `--rttm-out-dir`. Already-processed files are skipped on re-runs.
4. **DER evaluation** — Compares predicted RTTM against CHA ground truth. Scoring is restricted to the UEM region (min/max annotated timestamps from CHA) with a configurable collar tolerance (default 0.25 s). Reports macro-average, weighted-average, speaker count accuracy, and best/worst files.

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

The same stage can be used in any NeMo Curator pipeline:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.tasks import AudioBatch

pipeline = Pipeline(
    name="diarization",
    stages=[
        InferenceSortformerStage(
            model_name="nvidia/diar_streaming_sortformer_4spk-v2",
            rttm_out_dir="./rttm",
        ),
    ],
)

tasks = [
    AudioBatch(
        data=[{"audio_filepath": "audio.wav", "session_name": "session1"}],
        task_id="task_1",
        dataset_name="my_dataset",
    ),
]

executor = RayActorPoolExecutor()
results = pipeline.run(executor=executor, initial_tasks=tasks)
```

## Model limitations

- Maximum 4 speakers per recording
- Trained primarily on English speech
- Performance may degrade on noisy or very long recordings
