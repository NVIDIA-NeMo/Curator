# Alignment with refinement

"Разметка с уточнением" uses the standard Montreal Forced Aligner (MFA):

```text
audio + supplied text -> mfa align -> physical NeMo tar shards
                      -> Silero + Sortformer -> refined word annotations
```

MFA aligns the supplied transcript to the audio. The transcript is preserved;
no ASR transcription replaces it. MFA word intervals are the protected lower
bound: refinement may widen them but cannot contract, remove or relabel words.
The recipe invokes the official executable directly. It does not bundle the
user's custom MFA stage or invoke NeMoASRAlignerStage.

## Environment and inputs

Use Python 3.12+ in an existing Curator audio environment. Curator's reader
handles ordinary and tarred audio. Silero/Sortformer require the compatible
NeMo Speech build documented in [README.md](README.md).

MFA is an external application with its own environment. This adapter targets
MFA 3.x and is tested with 3.4.1. Its standard command syntax is documented in
[the official MFA guide](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html).
Prepare a pronunciation dictionary and acoustic model for the transcript's
language. Pass local model files for offline runs; named models may be
downloaded by MFA. This recipe does not install or modify environments.

Input JSONL contains one audio/transcript pair per row:

```json
{"audio_filepath":"/data/clip.wav","text":"the supplied transcript","speaker":"reader-1","offset":0.0,"duration":4.2}
```

Audio, a nonempty `text`, and positive duration are required. Optional `offset`
selects a portion of the recording. The reader applies this offset once and
materializes a PCM-16 WAV used by MFA and refinement. Its time origin is zero;
the original file, offset, row and shard are retained as provenance. Speaker
labels group MFA's corpus directories; missing speaker labels receive distinct
clip-specific identifiers. Use one language/model pair per run.

Alternatively, `--input-config /data/input.yaml` accepts the existing Curator
NeMo `input_cfg` format, including `nemo` and `nemo_tarred` entries:

```yaml
input_cfg:
  - type: nemo_tarred
    corpus: read_speech
    language: en
    shard_key_prefix: read_speech
    manifest_filepath: /data/manifest_0.jsonl
    tarred_audio_filepaths: /data/audio_0.tar
```

Tarred manifests require matching `shard_id` fields. Plain JSONL is decoded in
bounded batches. Existing tarred input materializes one physical shard per
reader process; size those input shards to fit available memory.

## Run

From the Curator source checkout:

```bash
python -m tutorials.audio.vad_refinement.alignment_with_refinement \
  --input-manifest /data/audio_and_text.jsonl \
  --output-root /data/alignment_run \
  --mfa-command /path/to/mfa-environment/bin/mfa \
  --mfa-dictionary /models/english_us_arpa.dict \
  --mfa-acoustic-model /models/english_us_arpa.zip \
  --mfa-jobs 4 --alignment-batch-size 32 --records-per-shard 64 \
  --sortformer-model /models/Nemotron-3-Diarization.nemo
```

The example models are for English. `--mfa-model-root` can point to an existing
MFA model directory; installed names are resolved from its `pretrained_models/`
subdirectories. Each worker gets isolated MFA state and each batch gets its own
corpus, work directory, output TextGrids and log. Shared model files are read
without changing the user's original MFA directory.

Use `--stage align`, `--stage shard` and `--stage refine` to run phases in
separate environments. Alignment requires the MFA arguments but not Sortformer.
The shard phase requires only the output root and optional records-per-shard.
Refinement requires Sortformer but does not invoke MFA again.

For parallel refinement, launch one process per GPU:

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

The worker arguments also partition MFA batches. Alignment workers receive the
same input and batch size; this repeats decoding/I/O across workers. MFA uses
`--mfa-jobs` CPU processes within a batch. Wait for all alignment workers before
the single-writer shard phase. Modes `all` and `shard` require one worker.
The commands do not launch workers or submit Ray/Xenna/Slurm jobs automatically.
Existing outputs are rejected; use a fresh root for retries. Completion markers
are integrity signals, not automatic checkpoint/restart support.

## Outputs

```text
alignment_run/
  aligned/<batch-id>.jsonl
  clips/<clip-id>.wav
  mfa/batch-*/
    corpus/<speaker>/<clip-id>.wav
    corpus/<speaker>/<clip-id>.lab
    textgrids/...
    mfa.log
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

Copy the complete `shards/` directory to `<new-output-root>/shards/` to refine
elsewhere. Manifests, tar files and TextGrid sidecars suffice; original working
clips/MFA outputs are unnecessary. `input.json` has relative NeMo paths and can
be used from inside the shards directory. Refinement outputs retain their own
audio and baseline TextGrid copies, without temporary extraction paths.

Rows identify `alignment_source="montreal_forced_aligner"`. Their
`alignment_textgrid` and `mfa_textgrid` fields point to the same unchanged MFA
alignment, while `text` and `source_text` retain the supplied transcript.
Missing TextGrids, missing words, invalid timestamps or unreadable audio fail
explicitly. No evenly distributed fallback word timings are substituted.
Filtering remains separate and opt-in. Logs/audits contain local paths and
transcripts; review them before publication.

## Tests

In an existing Curator/NeMo audio environment:

```bash
python -m pytest -o addopts= \
  --confcutdir=tests/tutorials/audio/vad_refinement \
  tests/tutorials/audio/vad_refinement/integration -q
```

The automated integration tests run a fixture executable through the real
subprocess adapter, then exercise TextGrid preservation, the real NeMo tar
reader and refinement. They cover required transcripts, failing commands,
missing MFA output, nonzero offsets, filename collisions, bounded reads,
portable shards and disjoint workers. These fixtures do not measure MFA or
neural-model accuracy. Real smoke checks must be reported separately.

A real MFA 3.4.1 smoke check aligned one public LibriSpeech clip (14.085
seconds, 34 words) through this direct CLI adapter. The supplied transcript
and copied TextGrid were unchanged. Physical tar reading and refinement with
empty detector fixtures preserved every MFA word interval exactly. This
check performed no neural inference and submitted no cluster jobs.
