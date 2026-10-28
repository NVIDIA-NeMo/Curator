# Alignment with refinement

"Разметка с уточнением" starts with audio and its supplied transcript:

```text
audio + transcript -> Curator MFA batch alignment -> physical tar shards
                   -> Silero + Sortformer -> refined word annotations
```

The original single-session refinement command remains available. This entry
point handles independent, single-speaker clips such as read-speech datasets.
It does not transcribe audio, separate speakers, select training data or submit
cluster jobs. Original MFA words remain the hard lower bound throughout.

## Environment and input

Use Python 3.12+ and a Curator audio environment with its NeMo/Lhotse reader,
SoundFile and TextGrid. For refinement, use the compatible NeMo Speech source
build and pinned Nemotron checkpoint documented in [README.md](README.md).
MFA is a separate application: supply the executable from an existing MFA
environment using `--mfa-command`. The acoustic model and pronunciation
dictionary must match the transcript language. Prepare those models before
running; this recipe does not install environments or select models by language.

The batch MFA implementation is adapted from an existing Curator
`MFAAlignmentStage`, with its Apache-2.0 notice retained in `mfa_alignment.py`.
It is bundled inside this source-checkout recipe because that stage is not yet
in public Curator. The recipe requests TextGrid output only; RTTM/CTM conversion
and its optional Praatio dependency are not needed. Filename collisions are
rejected, and missing MFA word alignments fail the batch instead of substituting
uniform word durations.

An input JSONL contains one audio/text pair per row:

```json
{"audio_filepath":"/data/clip.wav","text":"the supplied transcript","speaker":"reader-1","offset":0.0,"duration":4.2}
```

`audio_filepath`, `text` and a positive `duration` follow NeMo reader conventions.
The optional `offset` selects a portion of the input recording. Use a
whitespace-free `speaker` or `speaker_id` if supplied. The reader decodes the
selected portion once; the recipe stores a canonical PCM-16 WAV consumed by
both MFA and refinement. Clip-local time starts at zero. The source file,
offset, shard and row identifiers are retained separately as provenance.

Alternatively, `--input-config /data/input.yaml` accepts the existing Curator
NeMo `input_cfg` format, including `nemo` and `nemo_tarred` entries, corpus and
language metadata, and physical manifest/tar pairs. For example:

```yaml
input_cfg:
  - type: nemo_tarred
    corpus: read_speech
    language: en
    shard_key_prefix: read_speech
    manifest_filepath: /data/manifest_0.jsonl
    tarred_audio_filepaths: /data/audio_0.tar
```

NeMo tarred manifests must have matching `shard_id` fields. Plain JSONL input is
decoded in batches bounded by `--mfa-batch-size`. Existing tarred input uses
Curator's physical-shard reader, which materializes one shard's decoded clips
in memory; size those input shards to fit one reader process.

## Run all three steps

Run from the Curator checkout:

```bash
python -m tutorials.audio.vad_refinement.alignment_with_refinement \
  --input-manifest /data/audio_and_text.jsonl \
  --output-root /data/alignment_run \
  --mfa-command /path/to/mfa-environment/bin/mfa \
  --mfa-dictionary english_us_arpa \
  --mfa-acoustic-model english_us_arpa \
  --mfa-root-dir /path/to/prepared-mfa-models \
  --mfa-jobs 4 --mfa-batch-size 64 \
  --records-per-shard 64 \
  --sortformer-model /models/Nemotron-3-Diarization.nemo
```

The example MFA models are for English. Choose appropriate installed models for
other languages. Each output root belongs to one run. Existing stage outputs
are rejected rather than overwritten; use a fresh root for retries. Completion
markers describe successful shards, not automatic checkpoint/restart support.

Use one language/model pair per alignment run. MFA groups clips by their
provided speaker labels; clips without speaker metadata receive distinct
speaker identifiers rather than being treated as one shared speaker.

## Separate CPU alignment from GPU refinement

The stages can use separate existing environments. First run the command above
with `--stage align`; the Sortformer arguments are unnecessary for this step.
MFA uses `--mfa-jobs` CPU workers within each batch.

After every alignment worker has finished successfully, assemble physical shards
with one writer:

```bash
python -m tutorials.audio.vad_refinement.alignment_with_refinement \
  --stage shard --output-root /data/alignment_run --records-per-shard 64
```

Then run refinement, with one process per GPU. For two GPUs, launch these two
commands independently:

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

Each process loads its detectors once and owns disjoint whole tar shards. Silero
uses the existing CPU path; Sortformer uses the selected device. The commands
do not create processes or invoke Ray/Xenna/Slurm automatically.

`--worker-count`/`--worker-index` also partition alignment batches deterministically.
Each worker must receive the same inputs and batch size. Plain input is read by
each alignment worker, so this trades extra decoding/I/O for parallel MFA.
MFA model roots are copied into separate worker/node-local directories to avoid
shared extraction/history files. Wait for all alignment workers before the
single-writer shard step. `--stage all` and `--stage shard` require one worker.

## Outputs and portability

```text
alignment_run/
  aligned/<batch-id>.jsonl
  clips/<clip-id>.wav
  mfa/textgrids/<batch>/...
  shards/
    input.json
    shard-000000/
      manifest_0.jsonl
      audio_0.tar
      alignments/<clip-id>.TextGrid
      shard.json
  refined/
    shard-000000/
      manifest_refined.jsonl
      complete.json
      clips/<row>/...
```

The `shards/` directory is independently portable: copy the manifests, tar files
and alignment sidecars together. Audio is stored in NeMo-compatible tar files;
TextGrids remain sidecars so ordinary NeMo audio readers can consume the tar.
`input.json` contains relative NeMo configuration paths: use it from within
`shards/` when connecting another Curator pipeline. Refinement can run after
copying this directory to `<new-output-root>/shards/`; the old `aligned/`,
`clips/` and `mfa/` directories are not needed.

The refined manifest preserves one row per clip, source provenance, paths to
the original and refined TextGrids, and the refinement audit. Per-clip RTTMs
and debug outputs are also retained. Different clips from the same speaker
are independent sessions; they are not combined into a fictitious recording.
Refinement outputs include their own audio and MFA copies, so their paths do
not refer to temporary extraction directories.

A missing/corrupt audio file, missing MFA words, overlapping word intervals or
invalid timestamps fail the affected operation. No successful completion marker
is written for a failed refinement shard. No input is silently filtered, and
no synthetic word boundaries are substituted for failed MFA alignment.

## Tests

The integration suite requires the existing Curator/NeMo audio environment but
does not run MFA or neural inference by default:

```bash
python -m pytest -o addopts= \
  --confcutdir=tests/tutorials/audio/vad_refinement \
  tests/tutorials/audio/vad_refinement/integration -q
```

It exercises the real NeMo tar reader, nonzero source offsets, repeated input
filenames, portable shards after deleting original working files, disjoint
worker outputs, missing audio and missing MFA results. Alignment and detector
predictions are fixtures; these tests establish the file/timing integration,
not real-model alignment quality. Existing real-model validation of the
refinement algorithm does not establish end-to-end MFA accuracy for this new
entry point.

A real MFA 3.4.1 smoke check aligned one public LibriSpeech clip (14.085 seconds,
34 MFA words), packed it into a physical tar shard and decoded it through
Curator's NeMo reader. Refinement with empty detector activity preserved all
34 original word intervals exactly. That smoke check used no neural inference
and submitted no cluster jobs.
