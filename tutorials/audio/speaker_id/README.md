# Speaker ID + MOS pipeline

End-to-end example wiring the speaker-ID and UTMOSv2 stages into a single
[`Pipeline`](../../../nemo_curator/pipeline/pipeline.py) over **NeMo tarred shards**.

## Stages

| # | Stage | Task flow | Output |
|---|-------|-----------|--------|
| 1 | `SpeakerEmbeddingLhotseStage` | `_EmptyTask → _EmptyTask` | per-shard `embeddings/embeddings_<id>.npz` |
| 2 | `SpeakerClusteringStage` | `_EmptyTask → _EmptyTask` | `annotated_manifests/` with `speaker_label`, `confidence_score` |
| 3 | `JsonlReader` | `_EmptyTask → DocumentBatch` | reads the annotated manifests back |
| 4 | `GetUtmosv2ScoreStage` *(optional)* | `DocumentBatch → DocumentBatch` | adds `utmosv2_score` |
| 5 | `JsonlWriter` *(optional)* | `DocumentBatch → ∅` | `scored_manifests/` |

Stages 1–2 communicate via disk. Each emits its output task only *after*
`process()` finishes writing, so a downstream stage never observes a partial
directory — the chain is correct even under streaming execution.

## Usage

Speaker ID only (no audio files needed beyond the tars):

```bash
python pipeline.py \
    --input-manifest "/data/manifest__OP_0..9_CL_.json" \
    --input-tar      "/data/audio__OP_0..9_CL_.tar" \
    --output-dir     /data/speaker_id_out \
    --threshold      0.292 \
    --gpus           1.0
```

Add UTMOSv2 MOS scoring. UTMOSv2 reads the actual audio, so for tarred data
point `--audio-root` at a directory where each manifest `audio_filepath`
resolves to a readable file:

```bash
python pipeline.py \
    --input-manifest "/data/manifest__OP_0..9_CL_.json" \
    --input-tar      "/data/audio__OP_0..9_CL_.tar" \
    --output-dir     /data/speaker_id_out \
    --utmos --audio-root /data/extracted_wavs
```

`_OP_..._CL_` is NeMo's brace-expand syntax for shard ranges (`{0..9}`).
Run `python pipeline.py --help` for all options.
