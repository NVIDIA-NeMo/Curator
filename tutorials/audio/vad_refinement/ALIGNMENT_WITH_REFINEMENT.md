# Alignment with refinement

"Разметка с уточнением" uses the public Curator `NeMoASRAlignerStage`:

```text
audio -> Curator NeMo ASR words/timestamps -> physical tar shards
      -> Silero + Sortformer -> refined word annotations
```

This stage recognizes speech and timestamps its recognized words. It does not
force-align a supplied transcript and does not run Montreal Forced Aligner.
Optional input text is retained in `source_text`; `text` contains the recognized
transcript. The protected lower bound is the word alignment produced by Curator.
Refinement preserves its words, labels and original extents. No transcription
accuracy or boundary-accuracy gain is implied.

## Environment and input

Use Python 3.12+ and the Curator audio environment. The reader and
`NeMoASRAlignerStage` are imported directly from the public library; no local
copy or private MFA stage is bundled. For Silero/Sortformer, use the compatible
NeMo Speech build and pinned Nemotron checkpoint in [README.md](README.md).
No MFA executable, dictionary, acoustic model or Praatio installation is needed.

Input JSONL uses NeMo reader conventions:

```json
{"audio_filepath":"/data/clip.wav","text":"optional supplied transcript","speaker":"reader-1","offset":0.0,"duration":4.2}
```

Audio and positive duration are required. Optional `offset` selects a portion of
the recording. The selected clip is materialized once as PCM-16 WAV and used by
both ASR and refinement. Its time starts at zero; original path, offset, row and
shard identifiers are saved separately. Use speaker labels without whitespace.

Alternatively, `--input-config /data/input.yaml` accepts Curator's NeMo
`input_cfg`, including ordinary and tarred inputs:

```yaml
input_cfg:
  - type: nemo_tarred
    corpus: read_speech
    language: en
    shard_key_prefix: read_speech
    manifest_filepath: /data/manifest_0.jsonl
    tarred_audio_filepaths: /data/audio_0.tar
```

Tarred manifests need matching `shard_id` fields. Plain JSONL is decoded in
bounded batches. Existing tarred input materializes one physical shard's audio
per reader process; size those input shards to fit available memory.

## Run

From a Curator source checkout:

```bash
python -m tutorials.audio.vad_refinement.alignment_with_refinement \
  --input-manifest /data/audio.jsonl \
  --output-root /data/alignment_run \
  --aligner-model nvidia/parakeet-tdt_ctc-1.1b \
  --aligner-decoder rnnt --aligner-device cuda \
  --alignment-batch-size 32 --records-per-shard 64 \
  --sortformer-model /models/Nemotron-3-Diarization.nemo
```

The aligner model and RNNT decoder above match the defaults of Curator's stage.
Choose a model appropriate for the input language. `--aligner-model-path`
restores an existing local checkpoint instead of downloading by model name.
`--aligner-device cpu` is available for a small smoke run; normal model loading
can require substantial memory. Models load once per alignment/refinement
process. The command does not submit cluster jobs or launch workers itself.

Run phases separately with `--stage align`, `--stage shard` and `--stage refine`.
Alignment does not need Sortformer arguments. The shard phase needs only
`--output-root` and optional `--records-per-shard`. Refinement needs the output
root and Sortformer arguments but does not load the ASR model.

For parallel refinement, launch one command per GPU, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tutorials.audio.vad_refinement.alignment_with_refinement \
  --stage refine --output-root /data/alignment_run \
  --sortformer-model /models/Nemotron-3-Diarization.nemo \
  --sortformer-device cuda --worker-count 2 --worker-index 0

CUDA_VISIBLE_DEVICES=1 python -m tutorials.audio.vad_refinement.alignment_with_refinement \
  --stage refine --output-root /data/alignment_run \
  --sortformer-model /models/Nemotron-3-Diarization.nemo \
  --sortformer-device cuda --worker-count 2 --worker-index 1
```

The same worker arguments partition alignment batches. Each alignment worker
receives the same input and batch size; this repeats decoding/I/O across workers.
Wait for all alignment workers to finish before running the single-writer shard
phase. The `all` and `shard` modes require one worker. Workers do not use
Ray/Xenna/Slurm automatically. Existing outputs are rejected; use a fresh output
root for retries. There is no automatic checkpoint/restart support.

## Outputs

```text
alignment_run/
  aligned/<batch-id>.jsonl
  clips/<clip-id>.wav
  alignments/<clip-id>.TextGrid
  shards/
    input.json
    shard-000000/
      manifest_0.jsonl
      audio_0.tar
      alignments/<clip-id>.TextGrid
      shard.json
  refined/shard-000000/
    manifest_refined.jsonl
    complete.json
    clips/<row>/...
```

Copy the complete `shards/` directory to `<new-output-root>/shards/` to run
refinement elsewhere. Its manifests, tar files and alignment sidecars suffice;
original working clips and TextGrids are not required. `input.json` contains
relative NeMo paths and can be used from inside the `shards/` directory.

Rows record `alignment_source="curator_nemo_asr"`, the source transcript,
recognized transcript and `alignment_textgrid`. The legacy refinement input name
`mfa_textgrid` is kept as a compatibility alias to that same TextGrid; it does
not indicate that MFA was run. Audits identify the actual anchor source.
Refinement outputs retain their own audio and baseline alignment copies, with
no references to temporary extraction files. Missing/invalid/overlapping word
timestamps fail explicitly; no word boundaries are synthesized or repaired.
Filtering remains a separate opt-in operation.

## Tests

In an existing Curator/NeMo audio environment:

```bash
python -m pytest -o addopts= \
  --confcutdir=tests/tutorials/audio/vad_refinement \
  tests/tutorials/audio/vad_refinement/integration -q
```

Tests call the actual public `NeMoASRAlignerStage` batch and timestamp conversion
methods with fixture model hypotheses, then exercise TextGrid generation, real
NeMo tar reading and refinement. They cover supplied/recognized text separation,
nonzero input offsets, filename collisions, bounded reads, portable shards,
disjoint workers, missing results and invalid word timestamps. No neural model
inference is performed by these tests. Prior MFA validation applies to the
removed implementation, not this ASR entry point. A new end-to-end neural smoke
run and an independent accuracy evaluation have not been performed.
