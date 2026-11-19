# Speaker Diarization on CallHome English with NeMo Curator

This tutorial runs [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) speaker diarization on the [CallHome English](https://catalog.ldc.upenn.edu/LDC97S42) dataset using NeMo Curator's adapter-backed `InferenceSortformerStage`, then evaluates Diarization Error Rate (DER).

The tutorial is one YAML file, `pipeline.yaml`, run by `nemo_curator/config/run.py` on the Xenna executor.

## Prerequisites

- Python 3.11+
- NeMo Curator installed (see the [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- Licensed CallHome English audio and CHAT annotations, staged as shown below

### Stage the Dataset

The audio and CHAT annotations may be distributed separately and the source
audio may require conversion. Convert the recordings to WAV, then stage each
recording and its matching annotation under one directory before running the
tutorial:

```
/path/to/callhome_eng0/
├── 0638.wav
├── 4065.wav
├── ...
└── eng/
    ├── 0638.cha
    ├── 4065.cha
    └── ...          # matching CHAT-format annotations
```

## Usage

From the repository root:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/callhome_diar \
  --config-name pipeline \
  data_dir=/path/to/callhome_eng0
```

`--config-path` is relative to `nemo_curator/config/run.py`. Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | *(required)* | CallHome dataset root (WAVs, with `.cha` files under `eng/`) |
| `output_dir` | `./callhome_output` | Root for RTTM files and `results.jsonl` |
| `collar` | `0.25` | Collar tolerance (seconds) for DER scoring |
| `model_id` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Hugging Face model id |

### Streaming Configuration

All values are in **80 ms frames**. Override them with `chunk_len=...`,
`chunk_right_context=...`, `fifo_len=...`, `spkcache_update_period=...`, and
`spkcache_len=...`. The YAML places these model-specific values in the NeMo
adapter's `adapter_kwargs`; the stage continues to own task I/O,
whole-recording preparation, result fields, and RTTM files.

| Configuration | Latency | chunk_len | chunk_right_context | fifo_len | spkcache_update_period | spkcache_len |
|---------------|---------|-----------|---------------------|----------|------------------------|--------------|
| Very high (default) | 30.4 s | 340 | 40 | 40 | 300 | 188 |
| High | 10.0 s | 124 | 1 | 124 | 124 | 188 |
| Low | 1.04 s | 6 | 7 | 188 | 144 | 188 |
| Ultra low | 0.32 s | 3 | 1 | 188 | 144 | 188 |

## What the Pipeline Does

1. **File discovery (`CreateInitialManifestAudioFolderStage`)** — Emits one `AudioTask` per WAV file in `data_dir`.
2. **Diarization inference (`InferenceSortformerStage`)** — Loads each whole recording, downmixes the stereo CallHome audio to mono 16 kHz in memory, and sends it to `NeMoSortformerAdapter`. It adds `diar_segments` and `num_speakers`, and writes RTTM files to `<output_dir>/rttm/`.
3. **DER evaluation (`DiarizationErrorRateStage`)** — Compares predicted segments against the matching `eng/<stem>.cha` reference and writes `der_metrics`. Scoring is restricted to the UEM region (min/max annotated timestamps from the CHA file) with a configurable collar tolerance (default 0.25 s). Recordings without a `.cha` file get `der_metrics: null`.
4. **`ManifestWriterStage`** — Writes one row per recording to `<output_dir>/results.jsonl`.

An interrupted run starts again from the beginning.

### Summarize the results

Each `der_metrics` entry holds `der`, `miss`, `fa`, `conf`, and `correct` (percent of
reference speech), `gt_speech_s`, `pred_speech_s`, `gt_speakers`, and `pred_speakers`.
Macro- and speech-weighted averages:

```bash
python - <<'PY'
import json

rows = [json.loads(line) for line in open("callhome_output/results.jsonl")]
scored = [r["der_metrics"] for r in rows if r["der_metrics"]]
total = sum(m["gt_speech_s"] for m in scored)
for key in ("der", "miss", "fa", "conf"):
    macro = sum(m[key] for m in scored) / len(scored)
    weighted = sum(m[key] * m["gt_speech_s"] for m in scored) / total
    print(f"{key}: macro={macro:.1f}% weighted={weighted:.1f}%")
match = sum(m["gt_speakers"] == m["pred_speakers"] for m in scored)
print(f"speaker count match: {match}/{len(scored)}")
PY
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
- The stage downmixes and resamples inputs to mono 16 kHz in memory.
