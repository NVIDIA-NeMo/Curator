# NeMo speech YAML reader and per-shard Opus writer

This tutorial reads ordinary or tarred NeMo speech manifests from a NeMo
`input_cfg` YAML file. Each physical manifest is one restart boundary. The
pipeline decodes its rows into `AudioTask` objects, writes collision-safe Opus
clips and durable row receipts, then publishes one output manifest and one
completion marker per input shard.

A physical manifest/tar pair is both the resume unit and the reader memory
unit. The current reader materializes that shard's rows and decoded outputs
before emission, so size physical shards to fit one worker's RAM. Peak reader
memory scales with concurrently active reader workers; prefer more bounded
physical shards over one very large manifest.

## Install

From the repository root:

```bash
uv sync --extra audio_cpu
source .venv/bin/activate
```

Opus output requires either libsndfile with OGG/Opus support (used through
SoundFile) or `ffmpeg` built with the `libopus` encoder. Check the available
encoders before a large run:

```bash
python - <<'PY'
import soundfile as sf
print("SoundFile OGG/Opus:", sf.check_format("OGG", "OPUS"))
PY

ffmpeg -hide_banner -encoders 2>/dev/null | grep libopus || true
```

## Create the input YAML

The top level can be a mapping containing `input_cfg`, as below, or a list of
entries directly:

```yaml
input_cfg:
  - corpus: librispeech
    language: en
    type: nemo
    manifest_filepath: /datasets/librispeech/manifests/train-clean-100.jsonl
    shard_key_prefix: librispeech/en/train-clean-100

  - corpus: granary
    language: hi
    type: nemo_tarred
    manifest_filepath: /datasets/granary/hi/manifest__OP_0..127_CL_.json
    tarred_audio_filepaths: /datasets/granary/hi/audio__OP_0..127_CL_.tar
    shard_key_prefix: granary/hi/train
```

Each entry supports:

| Field | Required | Meaning |
|---|---:|---|
| `manifest_filepath` | Yes | One manifest, a list of manifests, or NeMo shard-range syntax. |
| `type` | No | `nemo` or `nemo_tarred`. When omitted, the tar field selects `nemo_tarred`. |
| `tarred_audio_filepaths` | For `nemo_tarred` | One tar, a list, or the same shard-range shape as the manifests. Expanded manifest and tar counts must match. |
| `corpus` | No | Dataset name and shard-key anchor; defaults to `unknown`. |
| `language` | No | Copied to `source_lang` when a row does not already set it. |
| `shard_key_prefix` | Recommended | Stable, relative output namespace. It must not be absolute or contain `..`. |

For a non-tarred manifest, each JSONL row normally contains
`audio_filepath`, `duration`, and `text`. For a tarred manifest,
`audio_filepath` names the member inside the matching tar; the usual NeMo
tarred-manifest metadata includes an integer `shard_id` as well as `duration`.
For example, an offset row can be:

```json
{"audio_filepath":"session-sub1.wav","duration":12.5,"offset":30.0,"text":"...","shard_id":0}
```

Input YAML and manifests can be local paths or
fsspec-compatible URIs. Use absolute local paths because executor workers can
have a different working directory from the driver. Tar files are opened by
NeMo/Lhotse rather than fsspec directly, so use a local path, a URI supported
by its active I/O backend, or a NeMo `pipe:` source; generic fsspec support for
a scheme is not sufficient.

Without `shard_key_prefix`, `corpus` must occur exactly once as a component of
the manifest path. Supplying an explicit prefix avoids ambiguous keys and
makes the output layout stable if an input root changes.

## Run with done-marker resume

Done-marker mode is the default and is the simplest choice when output lives
on a filesystem shared by every worker:

```bash
python tutorials/audio/nemo_speech_io/run.py \
  --input-config /absolute/path/to/input.yaml \
  --output-dir /shared/curator/nemo-speech-output \
  --resume-mode done_markers \
  --backend xenna
```

Do **not** pass `--checkpoint-path` in this mode. A valid
`<shard>.jsonl.done` marker plus its manifest is the scheduling authority. A
rerun skips complete shards. For an incomplete shard, discovery removes its
partial manifest and row-receipt directory, but retains already committed
Opus files so the writer can validate and reuse them while replaying the
shard. Pass `--no-cleanup-partial` only for investigation; stale receipts can
otherwise make recovery harder to reason about.

Audio ownership claims are immutable restart records. If transformation logic
or output identity changes, use a fresh output directory instead of mixing a
new run with prior claims.

## Run with pipeline checkpoints

Checkpoint mode uses `Pipeline.run(checkpoint_path=...)` as the only
scheduling authority:

```bash
python tutorials/audio/nemo_speech_io/run.py \
  --input-config /absolute/path/to/input.yaml \
  --output-dir /shared/curator/nemo-speech-output \
  --resume-mode checkpoint \
  --checkpoint-path /shared/curator/checkpoints/nemo-speech \
  --backend xenna
```

Do **not** combine checkpoint scheduling with `done_markers`. In checkpoint
mode, discovery deliberately ignores `.done` files and does not clean partial
receipts. Curator records completed source partitions in the checkpoint and
replays incomplete partitions. Checkpoint-backed runs require a Ray cluster
that was started before `Pipeline.run`; see the resumable-processing guide for
cluster setup and lifecycle details.

The finalizer still writes `.done` files in checkpoint mode, but they are
output-integrity records, not the scheduling authority.
The implementation marks done-marker discovery as non-resumable, so Curator
rejects an accidental `checkpoint_path` combination before it can clean
partial receipts.

Use the same absolute `output_dir` for the reader, writer, and finalizer. The
reader registers every physical shard there before downstream fan-out; using
a different writer directory would lose the evidence needed to detect a row
whose every child was dropped.

## Sample-rate contract

`NeMoSpeechAudioReader` preserves the source sample rate. The writer's
`target_sample_rate` is a validation target, not a resampling request. Every
waveform-bearing task must already match it; the default tutorial value is
16 kHz. If the source differs, insert an upstream resampling stage before the
writer or produce a resampled input dataset. The writer raises rather than
silently changing sample rate.

You can choose a different already-matching rate:

```bash
python tutorials/audio/nemo_speech_io/run.py \
  --input-config /absolute/path/to/48khz.yaml \
  --output-dir /shared/curator/nemo-speech-48khz \
  --target-sample-rate 48000
```

## Finalization is required

`NeMoSpeechWriterStage` atomically writes each Opus clip and a private JSON
receipt. It intentionally does **not** publish manifests or completion markers
from executor workers. After `pipeline.run()` succeeds, the driver must call:

```python
from nemo_curator.stages.audio.io import finalize_nemo_speech_output

manifests = finalize_nemo_speech_output("/shared/curator/nemo-speech-output")
```

The tutorial runner does this automatically. The finalizer first validates
all pending shards—receipt count, expected inputs, path safety, and referenced
Opus files—before writing any marker. It then atomically publishes each JSONL
manifest followed by its structured `.done` marker, which binds the marker to
the manifest's SHA-256.

Run exactly one finalizer after all writer workers are quiescent. Concurrent
pipeline writers or finalizers targeting the same output directory are not a
supported coordination mechanism.

The writer is a terminal sink and returns no decoded waveforms to the driver.
Its receipts are the durable per-row acknowledgements.

If the process stops after the pipeline finished but before finalization, do
not rerun completed work merely to publish it. Durable receipts can be
finalized separately:

```bash
python tutorials/audio/nemo_speech_io/run.py \
  --output-dir /shared/curator/nemo-speech-output \
  --finalize-only
```

Never call the finalizer in a `finally` block: a failed pipeline may have an
incomplete receipt set, which the finalizer correctly rejects.

Fan-out is supported: every child from one reader row keeps the same private
`_shard_input_id`, and all of its output rows are included. If an intermediate
stage intentionally turns one input into no outputs, it must emit a terminal
placeholder such as `vad_empty` or `read_error`. Silently dropping every child
leaves no durable evidence that the input was handled, so finalization rejects
the shard instead of publishing a misleading completion marker. If a custom
batch fan-out produces non-deterministic framework task IDs, assign a unique,
stable `_shard_output_id` to every child before it reaches the writer.

## Output layout

The output directory must be a local/shared POSIX-style filesystem—not an S3,
GCS, or other object-store URI—and must resolve to the same path on every
worker. Atomic rename and filesystem durability are part of the writer's
contract.

For a shard key such as `granary/hi/train/manifest_0`, output looks like:

```text
output/
├── granary/hi/train/manifest_0.jsonl
├── granary/hi/train/manifest_0.jsonl.done
├── granary/hi/train/manifest_0/audio/.../*.opus
└── .nemo_curator/
    ├── nemo_speech_shards/
    │   └── granary/hi/train/manifest_0.json
    ├── nemo_speech_rows/
    │   └── granary/hi/train/manifest_0/*.json
    └── nemo_speech_audio_owners/
        └── <hash[0:2]>/<hash[2:4]>/<sha256>.json
```

The public manifest uses Opus paths relative to the shard manifest's parent,
so NeMo resolves them correctly when the manifest is nested under `output`.
The internal `output_audio_filepath` field and ownership records remain
relative to the output root. The writer preserves JSON-safe input metadata,
adds both `sample_rate` and `sampling_rate`, and records
`original_audio_filepath`. Decode failures and over-duration rows are retained
as `read_error` placeholders with an empty `audio_filepath`, so output row
accounting still matches the input shard. The `.nemo_curator` row receipts and
hash-sharded audio-ownership records are restart state and should remain beside
the published output. The shard registration is written before downstream
fan-out or filtering and records the physical shard's expected input count, so
finalization detects even an all-dropped shard with no row receipts. Row
receipts still record durable task completion, while ownership claims still
prevent two task outputs from claiming the same Opus path and are required for
finalization checks.

Empty manifests are rejected explicitly. For Opus backfill, a preset
`output_audio_filepath` must be a relative `.opus` path below the matching
shard directory. `save_audio=False` preserves each resolved source reference;
for a non-tar manifest's relative input path, the reader keeps that row value
in task metadata while the output manifest uses NeMo's resolved path so it can
be read from its new location. It never emits a synthetic path to an unwritten
Opus file.

## Use the API directly

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io import (
    NeMoSpeechAudioReader,
    NeMoSpeechWriterStage,
    finalize_nemo_speech_output,
)

output_dir = "/shared/curator/nemo-speech-output"

pipeline = Pipeline(name="nemo_speech_io")
pipeline.add_stage(
    NeMoSpeechAudioReader(
        yaml_path="/absolute/path/to/input.yaml",
        output_dir=output_dir,
        resume_mode="done_markers",
        reader_workers=8,
    )
)
pipeline.add_stage(
    NeMoSpeechWriterStage(
        output_dir=output_dir,
        target_sample_rate=16000,
        writer_concurrency=4,
    )
)

pipeline.run(executor=XennaExecutor())
finalize_nemo_speech_output(output_dir)
```

Filters can be set with repeated `--corpus` and `--language` CLI flags, or
with `corpus_filter=[...]` and `language_filter=[...]` on the reader. Use
`--manifest-only`/`save_audio=False` only when you intentionally want manifest
and receipt output without encoded Opus files; the original input audio
references remain in those rows.
