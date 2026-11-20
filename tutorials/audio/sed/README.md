# Sound Event Detection with PANNs CNN14

Run framewise sound event detection (SED) over a JSONL audio manifest with the
PANNs `Cnn14_DecisionLevelMax` model and the AudioSet class vocabulary.

## Pipeline

`pipeline.yaml` runs two stages:

1. `ManifestReader` reads the input JSONL manifest.
2. `SEDInferenceStage` with `PANNsSEDAdapter` loads each file, resamples it to
   32 kHz mono, runs inference, and writes a compressed NPZ sidecar per file.

## Prerequisites

- GPU recommended (the stage requests 4 GB of GPU memory per actor).

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

## Quick start

Each input row needs an `audio_filepath`:

```json
{"audio_filepath": "/absolute/path/to/audio.wav"}
```

From the repository root:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/sed \
  --config-name pipeline \
  manifest_path=/absolute/path/to/input.jsonl \
  output_dir=/absolute/path/to/sed_output
```

`--config-path` is relative to `nemo_curator/config/run.py`.

## Checkpoint

With `checkpoint_path=null` (the default), the adapter loads
`Cnn14_DecisionLevelMax_mAP=0.385.pth` from the PyTorch Hub cache and downloads
it when it is missing. To use a fixed location, append:

```bash
'checkpoint_path=/absolute/path/to/Cnn14_DecisionLevelMax_mAP\=0.385.pth'
```

An existing file is loaded without network access; a missing one is
downloaded to that path. `Cnn14_mAP=0.431.pth` is a clip-level tagging
checkpoint and is not suitable for framewise SED.

## Output

Each task gains:

| Field | Description |
|---|---|
| `_sed_framewise` | `(frames, 527)` AudioSet probability matrix, about 100 frames per second |
| `sed_valid_frames` | Number of frames that cover the original audio (before batch padding) |
| `sed_fps` | Frame rate of the matrix |
| `npz_filepath` | Path of the NPZ sidecar under `<output_dir>/framewise` |

Each NPZ contains `framewise`, `fps`, `valid_frames`, `audio_filepath`, and
`original_num_samples`. Scores are not thresholded into labelled events.

## Common overrides

| Override | Effect |
|---|---|
| `batch_size=16` | Files per inference call (default `32`) |
| `stages.1.framewise_dtype=float32` | Store full-precision probabilities |
| `backend=xenna` | Run on Xenna instead of Ray Data |

See the [Sound Event Detection guide](../../../fern/versions/main/pages/curate-audio/process-data/sound-event-detection.mdx)
for more detail.
