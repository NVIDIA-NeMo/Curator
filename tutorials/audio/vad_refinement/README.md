# Refine aligned words with Silero and Sortformer

For audio with transcripts but no word alignment, use
[Alignment with refinement](ALIGNMENT_WITH_REFINEMENT.md): Curator's Montreal Forced Aligner stage
alignment, physical NeMo tar shards, then Silero/Sortformer refinement. It also
accepts existing NeMo tarred input and supports separate CPU/GPU worker commands.

This recipe refines forced-alignment word boundaries with two independent speech detectors:

- Silero VAD;
- NeMo Sortformer; and
- optionally, Sound Expert event masking before the vote.

The refinement is one deterministic, equal-weight vote over non-overlapping atomic time
regions. Original aligned words are hard speech anchors: they cannot contract, disappear,
change label, change order, or overlap.

The pipeline writes new TextGrid, RTTM, manifest, SegLST, and audit files. Source inputs are
never modified.

## Why atomic voting

MFA words, Silero intervals, and Sortformer intervals are nested and intersect at different
timestamps. Comparing only nearby endpoints requires tolerances, weights, pause thresholds,
and repeated postprocessing. Atomic voting instead partitions the timeline at every observed
boundary and makes one decision per resulting region.

The package intentionally has no:

- boundary-distance tolerance;
- maximum boundary shift;
- Silero/Sortformer edge weighting;
- short-pause merging threshold;
- transition penalty;
- minimum-silence objective; or
- second final-activity filtering pass.

## Algorithm

For each manifest row:

1. Read lexical words from the segment-local MFA TextGrid.
2. Run Silero and Sortformer once on the full speaker recording.
3. Clip both detector streams to the row and convert them to segment-local time.
4. Collect every MFA, Silero, and Sortformer start/end boundary.
5. Sort the boundaries into maximal non-overlapping atomic regions.
6. Vote speech/silence independently on each region.
7. Merge adjacent speech-voted regions into consensus components.
8. Project components once onto the original words.
9. Generate TextGrids and RTTMs directly from the projected words.

### Vote rule

All systems have weight `1.0`.

Inside an MFA word:

```text
output = speech
```

This hard constraint guarantees word preservation.

Outside an MFA word, MFA votes silence:

```text
speech_votes = silero_speech + sortformer_speech
output = speech_votes == 2
```

Therefore an MFA gap is widened only where both detectors agree that speech is present. A
region supported by only one detector remains silence.

### Word projection

For each word, find the consensus component containing that word. The component may widen the
word start or end, subject to:

```text
new_start <= original_start
new_end >= original_end
new_start >= previous_word.end
new_end <= next_word.start
```

Consensus components that touch no MFA word are discarded because the pipeline does not invent
transcript words.

When adjacent words belong to the same continuous consensus component, divide
their original MFA gap equally. The shared boundary is
`MFA_end(first) + (MFA_start(second) - MFA_end(first)) / 2`. This preserves both
MFA words in full and gives each half of the gap, without overlapping word labels.
Touching MFA words keep their shared boundary. If the consensus contains a pause
between the words, keep that pause instead of filling it.

### Objective

Outside hard MFA-word constraints, the vote maximizes duration-weighted agreement:

```text
agreement =
  sum(region_duration * matching_source_votes)
  / sum(region_duration * number_of_sources)
```

There is no preference for more speech or less silence.

## Repository contents

| File | Purpose |
| --- | --- |
| `rectify_vad_boundaries.py` | Runs inference, atomic voting, TextGrid projection, RTTM writing, and auditing. |
| `filter_segments_by_vad_agreement.py` | Scores or filters rows by post-refinement frame agreement. |
| `sound_expert_intervals.py` | Converts optional Sound Expert posteriors into event intervals. |
| `textgrid_rttm.py` | Reads and writes TextGrid, RTTM, and recording-global intervals. |
| `build_seglst_variants.py` | Builds baseline and refined SegLST files. |
| `run_vad_refinement.sh` | Shell launcher for one or more manifests. |
| `examples/manifest.example.jsonl` | Minimal input example. |
| `tests/` | CPU-only tests with synthetic TextGrids and fake detector outputs. |

No audio, transcripts, model weights, credentials, internal URLs, or evaluation records are
included.

## Requirements

- Linux
- Python 3.11 or 3.12
- CUDA-capable GPU recommended for Sortformer and Sound Expert

Install development and lightweight inference dependencies:

```bash
# From the Curator repository root on x86_64 Linux:
uv sync --extra audio_cuda12
source .venv/bin/activate
python -m pip install -r tutorials/audio/vad_refinement/requirements.txt
```

The Curator audio extra includes NeMo ASR and Silero. For a separately managed environment, follow the
[official NeMo instructions](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
for the installed CUDA and PyTorch versions.

## Selected Sortformer checkpoint

Use [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization),
the September 2026 successor to Sortformer. For reproducibility, download the
checkpoint at the reviewed revision instead of following a moving model branch:

```bash
hf download nvidia/Nemotron-3-Diarization Nemotron-3-Diarization.nemo \
  --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e \
  --local-dir /models/nemotron3
export SORTFORMER_MODEL=/models/nemotron3/Nemotron-3-Diarization.nemo
```

The model card requires Python 3.12+ and current NVIDIA NeMo Speech. The
published PyPI NeMo 3.0.0 package cannot restore this checkpoint: its transformer
encoder rejects `self_attention_model='rope'`. The `nemo_version: 3.0.0` checkpoint
metadata does not identify a compatible build. Curator's locked PyPI package
therefore needs a recipe-specific replacement in the inference environment.

The real-model CPU smoke test passes with NeMo Speech commit
`d32dd550ffb016cd7b4133bb2bc69f15ba866361`, Python 3.12, PyTorch/torchaudio
2.11.0 CPU, Silero VAD 6.2.3, Transformers 4.57.6 and huggingface-hub 0.36.0.
After preparing the audio environment above, resolve the source build's ASR
dependencies and explicitly replace NeMo:

```bash
python -m pip install \
  'nemo-toolkit[asr] @ https://github.com/NVIDIA-NeMo/Speech/archive/d32dd550ffb016cd7b4133bb2bc69f15ba866361.tar.gz'
python -m pip install --force-reinstall --no-deps \
  'nemo-toolkit @ https://github.com/NVIDIA-NeMo/Speech/archive/d32dd550ffb016cd7b4133bb2bc69f15ba866361.tar.gz'
```

The second command is necessary because this source build also reports version
3.0.0; pip can otherwise retain the incompatible installed release. A subsequent
`uv sync` restores the lockfile's PyPI build, so apply the source replacement
after syncing this inference environment. This override is local to the recipe;
Curator's shared dependency lock is unchanged. Record the source commit alongside
package versions. The weights use OpenMDW-1.1; the recipe code uses Apache-2.0.

The recipe reads frame spacing from the loaded model's output subsampling factor
and preprocessor stride; older models can use the encoder stride. It does not
assume a fixed 10 ms output spacing. The selected checkpoint emits 10 ms frames.
The normal regression suite loads no models. The optional smoke test below
checks real inference and file generation; it does not measure boundary accuracy.

TextGrid reads preserve the input floating-point timestamps instead of the
TextGrid library's default five-decimal rounding. RTTM output retains 17
significant digits so serialization does not shrink protected MFA intervals or
round short speech intervals to zero. SegLST's explicit `--precision` option
still controls the decimal precision of that transcript-segment export. Timestamp
strings keep exactly that many decimal places without stripping trailing zeros:
`10.000` at precision 3 and `10` at precision 0.

## Input manifest

The input is JSON Lines. Every file represents one logical session, and every row represents
one aligned transcript segment on a full speaker-channel recording.

Required fields:

- `session_id`: identifier shared by all rows in one manifest;
- `speaker_id`: RTTM speaker label;
- `audio_filepath`: full speaker-channel audio;
- `offset`: row start on the full recording, in seconds;
- `duration`: row duration, in seconds; and
- `mfa_textgrid`: segment-local TextGrid with a tier named `words`.

Recommended fields:

- `segment_id`: stable output identifier;
- `text`: transcript for SegLST output; and
- `fallback_textgrid`: optional alignment used only if the primary TextGrid has no lexical
  words.

Example:

```json
{"session_id":"meeting-001","segment_id":"seg-0001","speaker_id":"speaker-00","audio_filepath":"/data/speaker-00.wav","offset":12.4,"duration":3.2,"mfa_textgrid":"/data/seg-0001.TextGrid","text":"example transcript"}
```

TextGrid times are segment-local. Detector intervals are recording-global and are localized
internally using `offset`.

MFA duration differences produce a warning rather than truncation or failure.
The recipe checks the recording duration using audio frame count / sample rate
and retains these values as `audio_num_frames`, `audio_sample_rate` and
`audio_duration`. The sharded recipe captures the actual clip sample count and
sample rate before passing the clip to MFA and persists them in the manifest.
Subsequent timing checks use those values without reopening the audio header.
For older manifests without these fields, the single-session recipe reads the
header once per speaker recording and reuses its metadata for subsequent segments.
`mfa_duration` preserves the source TextGrid duration;
`annotation_duration` covers the manifest segment and all protected MFA words.
The original `duration` remains unchanged for audio reading and physical shards.
Refined TextGrid, RTTM and SegLST exports retain the annotation extent even when
it exceeds the manifest duration or the audio end. No audio samples are added.
Warnings include all available durations and the segment offset, so substantial
differences can be reviewed without losing annotations. For a segment within a
long recording, the check compares its global end to the recording duration;
it does not treat the rest of the recording as part of the segment.

SegLST export reports an input-data error if an aligned word overlaps none of
the output fragments when splitting a segment. It does not silently drop that
word. Both baseline and refined exports preserve short pauses by default;
the baseline no longer merges gaps up to 0.2 seconds. Python callers can still
request pause merging explicitly with `maximum_pause`.

Offsets, durations and segment ends must be finite; offsets must be
non-negative and durations positive. Boolean times are rejected. Segment windows
for the same speaker must not overlap; adjacent windows and overlapping windows
for different speakers are allowed. Invalid inputs fail before model loading or
output creation. Overlapping same-speaker windows are an input-data problem:
the error identifies the speaker, conflicting manifest rows and boundary times.
The dataset owner must resolve those windows before rerunning refinement. The
recipe does not merge, trim or deduplicate conflicting input segments. Other
manifests can still be processed; the CLI returns a failure status if any manifest
has an error.

## Run

Shell launcher:

```bash
SORTFORMER_MODEL=/path/to/sortformer.nemo \
MANIFEST=/path/to/session.jsonl \
OUTPUT_ROOT=/path/to/output \
bash tutorials/audio/vad_refinement/run_vad_refinement.sh
```

Direct command:

```bash
python -m tutorials.audio.vad_refinement.rectify_vad_boundaries \
  --manifest /path/to/session.jsonl \
  --output-root /path/to/output \
  --sortformer-model /path/to/sortformer.nemo
```

Process a directory recursively:

```bash
python -m tutorials.audio.vad_refinement.rectify_vad_boundaries \
  --manifest-root /path/to/manifests \
  --manifest-pattern "**/*.jsonl" \
  --output-root /path/to/output \
  --sortformer-model /path/to/sortformer.nemo
```

Split work deterministically:

```bash
python -m tutorials.audio.vad_refinement.rectify_vad_boundaries ... --shard-count 4 --shard-index 0
python -m tutorials.audio.vad_refinement.rectify_vad_boundaries ... --shard-count 4 --shard-index 1
python -m tutorials.audio.vad_refinement.rectify_vad_boundaries ... --shard-count 4 --shard-index 2
python -m tutorials.audio.vad_refinement.rectify_vad_boundaries ... --shard-count 4 --shard-index 3
```

Inference is sequential within each process so models are loaded only once. Run one process per
GPU.

SegLST export clips intervals to each manifest row by default
(`--boundary-window 0`), so continuous speaker activity cannot expand adjacent
rows into overlapping transcript segments. The direct SegLST command can still
use an explicit positive `--boundary-window` for contextual expansion; those
exports can overlap neighboring rows. This export setting does not change
atomic voting or TextGrid word refinement.

## Optional event masking

Provide Sound Expert when non-speech vocal events should be removed from detector support:

```bash
SORTFORMER_MODEL=/path/to/sortformer.nemo \
SOUND_EXPERT_MODEL=nvidia/ParallelExpertEncoder \
MANIFEST=/path/to/session.jsonl \
OUTPUT_ROOT=/path/to/output \
bash tutorials/audio/vad_refinement/run_vad_refinement.sh
```

Event spans are subtracted from Silero and Sortformer before atomic voting. Original MFA words
remain hard speech anchors even when an event overlaps them.

## Outputs

Speaker RTTMs retain recording-based filenames when unique. If recording
stems collide with another speaker or the session ID, speaker RTTMs use unique
names under `rttm/speakers/`. Read the exact paths from `audit.json`; do not infer
them from input basenames. The combined session RTTM remains separate.


For `session_id=meeting-001`:

```text
OUTPUT_ROOT/
└── meeting-001/
    ├── audit.json
    ├── manifest_vad_refined.jsonl
    ├── rttm/
    │   ├── meeting-001_vad_refined.rttm
    │   └── <recording>_vad_refined.rttm
    ├── seglst/
    │   ├── meeting-001_mfa.seglst.json
    │   ├── meeting-001_vad_refined.seglst.json
    │   └── meeting-001_seglst_summary.json
    ├── textgrids/
    │   ├── meeting-001_vad_refined.TextGrid
    │   └── segments/
    │       └── <segment>_vad_refined.TextGrid
    └── vad_debug/
        ├── meeting-001_silero.rttm
        └── meeting-001_sortformer.rttm
```

Each segment audit records:

- the algorithm name;
- fixed source weights;
- number of atomic regions;
- consensus and anchored component counts;
- discarded unanchored component count; and
- every changed word boundary.

Generated audits contain local filesystem paths. Review or redact them before publication.

## Optional annotation diagnostics and filtering

Refinement, diagnostics and filtering are separate steps. The main launcher
performs refinement and optional SegLST generation; it does not run the agreement
tool or exclude rows based on agreement scores.

| Mode | How to request it | Result |
| --- | --- | --- |
| Refinement only | Run the refinement launcher/module | Refined annotations and audit files; no agreement-based selection. |
| Diagnostics without filtering | Run the agreement module without `--max-bad-ratio` | Reports for inspection; no filtered manifest is written. |
| Optional filtering | Explicitly supply `--max-bad-ratio` | The same diagnostic reports plus a separate selected manifest. |

### Inspect annotation problems without excluding data

Start with a diagnostic-only run. This reads the existing TextGrids and detector
intervals from the audit; it does not rerun neural inference or require a cluster.

If a manifest has a missing or corrupt audit, the tool continues with the other
manifests and saves their reports. The summary lists failed manifests under
`errors` and sets `complete` to `false`; the command returns a nonzero exit status.
An incomplete report must not be interpreted as a successful check of the entire dataset.

```bash
python -m tutorials.audio.vad_refinement.filter_segments_by_vad_agreement \
  --manifest /data/refined/meeting-001/manifest_vad_refined.jsonl \
  --output-dir /data/annotation-diagnostics \
  --frame-duration 0.01 --reference-source refined --vad-variant raw
```

No threshold is set by default. This command leaves the source manifest, audio
and TextGrids intact and does not create `filtered_manifests/`. It writes:

- `segment_frame_agreement.jsonl`: per-segment scores, file paths, session/speaker
  IDs, segment IDs, offsets, durations and available transcript text;
- `frame_agreement_summary.json`: scored/skipped counts, pooled statistics,
  per-language statistics and the source-manifest/audit paths; and
- `bad_ratio_distribution.csv`: a histogram of the per-segment disagreement
  ratios, with quantiles also recorded in the summary.

The tool compares speech/silence states at frame centers. A frame is bad only
when **both** Silero and Sortformer disagree with the selected TextGrid.
`bad_ratio = bad_frames / total_frames`; agreement with either detector makes
the frame good under this definition.

### What the diagnostics can highlight

These signals help prioritize potentially incorrect annotations and problematic
files for review. They indicate disagreement, not a confirmed annotation error.

| Signal | Potential annotation or file problem | Other explanations to check |
| --- | --- | --- |
| High `both_miss_speech_frames` | TextGrid marks speech where the audio may contain silence; word spans may extend too far, or the alignment may be shifted. | Quiet/degraded speech or speech missed by both detectors. |
| High `both_false_alarm_frames` | Both detectors find speech outside the annotated speech spans; words or speech regions may be missing or truncated. | Background speakers, vocalizations, noise or shared detector false alarms. |
| Many high-`bad_ratio` segments in one `audio_filepath` | A file may have the wrong annotation/audio pairing, channel, segment offset or broadly misaligned boundaries. | Recording-specific acoustic difficulty, language/domain mismatch or detector calibration. |
| `skipped_reference_unavailable` with a reason | A TextGrid/audit may be missing, unreadable or incompatible with the row's speaker/reference fields. | An incomplete run or an incorrect file path/configuration. |

Use `--reference-source mfa` to inspect the original MFA annotation and
`--reference-source refined` to inspect the refinement output. Keep the detector
variant and frame duration fixed when comparing them. Original MFA words remain
protected during refinement, so diagnostics can still highlight problems inside
those original words even though refinement does not remove or contract them.

### Find problematic files and segments

1. Inspect skipped counts/reasons first. Rows with no usable text are counted as
   `skipped_no_text`; they do not receive normal scored report rows. An unscored
   row or file must not be interpreted as clean.
2. Sort report rows with `status="scored"` by decreasing `bad_ratio`. Inspect
   `bad_frames` and segment duration as well, so short outliers do not obscure
   long problematic spans.
3. Group those rows by `audio_filepath` to identify files with repeated problems.
   For a pooled file score over its evaluated segments, use
   `sum(bad_frames) / sum(total_frames)`, rather than an unweighted mean of ratios.
   This describes only the evaluated segments; overlapping rows can count the
   same audio time more than once. The recipe writes segment-level results;
   grouping those results is the file-level triage step.
4. Use the reported `offset` and `duration` to locate the suspect region in the
   source recording, and open its MFA/refined TextGrid and `audit.json` alongside
   the audio. Check the transcript, word boundaries, speaker channel and timing
   origin before deciding whether to correct or exclude the segment.

The score does not directly verify word spelling, transcript completeness or
speaker identity. A low ratio can coexist with incorrect text, and a high ratio
can come from detector mistakes. The same detectors also guide refinement, so
post-refinement agreement is not an independent accuracy or DER measurement.
For `raw_event_excluded`, also inspect `event_excluded_frames`: event regions are
forced to silence in all compared streams and affect score interpretation.

### Enable filtering only when a selected subset is wanted

After inspecting the reports, explicitly provide a threshold to write a selected
manifest under the new output directory:

```bash
python -m tutorials.audio.vad_refinement.filter_segments_by_vad_agreement \
  --manifest /data/refined/meeting-001/manifest_vad_refined.jsonl \
  --output-dir /data/selected-subset \
  --frame-duration 0.01 --reference-source refined --vad-variant raw \
  --max-bad-ratio 0.10
```

`0.10` is an illustrative threshold, not a recommended universal cutoff. Choose
it using reviewed examples from the relevant corpus, language and recording
conditions. Scored rows with `bad_ratio <= threshold` enter
`filtered_manifests/`; unscored rows are not selected. All scored diagnostic rows,
including those above the threshold, remain in the report for review. The
original inputs are never edited or deleted by this selection step.

## Agent execution guide

An automation agent should follow this sequence:

1. Validate that all manifest rows share one `session_id`.
2. Verify each audio and TextGrid path before loading neural models.
3. Run one inference process per GPU.
4. Use the default detector thresholds unless the user explicitly requests detector
   calibration.
5. Do not add boundary tolerances, weighted edge averaging, pause merging, or a second activity
   pass.
6. Treat the original word count, labels, order, and extents as hard invariants.
7. Check `audit.json` and fail the run if a word contracts, disappears, changes label, or
   overlaps its neighbor.
8. Optionally run agreement diagnostics after refinement to identify suspicious annotations/files. Omit `--max-bad-ratio` for reports only; set a threshold explicitly only when a selected subset is requested.
9. Preserve source files and write all results under a new output root.

For reproducible reruns, record:

- repository revision;
- model checkpoint identifiers;
- detector thresholds;
- manifest checksum;
- shard count/index; and
- output audit path.

## Validate

The tests do not download neural models:

```bash
ruff format --check tutorials/audio/vad_refinement tests/tutorials/audio/vad_refinement
ruff check tutorials/audio/vad_refinement tests/tutorials/audio/vad_refinement
python -m pytest -o addopts= \
  --confcutdir=tests/tutorials/audio/vad_refinement \
  --ignore=tests/tutorials/audio/vad_refinement/integration \
  tests/tutorials/audio/vad_refinement -q
```

This CPU-only suite requires `pytest` and `TextGrid`; it does not load Ray,
PyTorch, NeMo, Silero models or GPU resources. `--confcutdir` keeps Curator's
repository-wide autouse Ray fixture outside this test run, and `-o addopts=`
omits the global test options. Detector outputs are synthetic, so these checks
validate refinement and file generation rather than neural-inference accuracy.


Tests verify:

- equal-weight atomic voting;
- both-detector agreement in MFA gaps;
- the last shared detector boundary is selected;
- original words never contract or disappear;
- word labels/order remain unchanged;
- projected words never overlap;
- input files remain unchanged; and
- expected TextGrid, RTTM, manifest, SegLST, and audit outputs are created.

## Limitations

- The TextGrid tier must be named `words`.
- Phone tiers are not regenerated.
- All rows in one manifest must share one session ID.
- One `speaker_id` must map to one full audio file per session.
- Detached VAD consensus components are discarded when they contain no transcript word.
- Frame-agreement scores are not transcript-accuracy scores.


### Optional real-model CPU smoke test

Provide the pinned local checkpoint and a short speech recording (at least two
seconds). The test performs actual Silero/Sortformer inference and writes
TextGrid, RTTM, manifest and audit files in pytest's temporary directory:

```bash
CURATOR_VAD_MODEL=/models/nemotron3/Nemotron-3-Diarization.nemo \
CURATOR_VAD_AUDIO=/data/short-speech.wav \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m pytest -o addopts= \
  --confcutdir=tests/tutorials/audio/vad_refinement \
  tests/tutorials/audio/vad_refinement/test_rectify_vad_boundaries.py \
  -k real_silero_sortformer -v
```

The test is skipped unless both environment variables are set. It uses synthetic
word anchors and checks finite speech intervals, word preservation, parseable
outputs and unchanged source annotations. It does not run MFA, evaluate transcript
correctness or demonstrate a quality gain. The recorded CPU check used the public
[Whisper JFK test clip](https://github.com/openai/whisper/blob/main/tests/jfk.flac);
weights and audio are not included in this repository. Sound Expert is disabled.

## License

Apache License 2.0. See `LICENSE`.

## Source and integration

Ported from NVIDIA's Apache-2.0 `vad-refinement-pipeline` 0.2.0, revision
`53329a018a049ec7fcf1d189f649df74557d9612`. `UPSTREAM.json` records original
module checksums. Module imports and launch commands use the Curator recipe
namespace. Atomic voting is preserved; word projection divides shared MFA gaps equally. Local adaptations
validate timeline inputs before inference, support legacy NeMo frame metadata,
prevent speaker/session RTTM filename collisions, and keep SegLST export
inside row boundaries by default.
The upstream regression tests are under `tests/tutorials/audio/vad_refinement`.

Run these commands from a Curator source checkout. The recipe is not included in
the installed Curator wheel.
The single-session refinement command consumes existing forced alignments; it
does not run MFA or infer transcripts. The separate alignment-with-refinement
entry point runs Curator's MFAAlignmentStage on supplied transcripts first.
Each manifest row uses a full, isolated speaker-channel recording. Sortformer
posteriors are reduced by the maximum across speakers into binary speech
activity, so this recipe does not assign speaker identities to mixed recordings.

The launcher's SegLST step runs after refinement. Frame-agreement filtering is
separate and opt-in. Neither atomic voting nor the filter uses DOVER-Lap.
Model inference runs sequentially within a process, with one detector instance
reused across manifests and one inference per speaker recording per manifest.
Use the CLI's deterministic shard options to run separate processes per GPU;
these standalone commands do not submit work to Curator's Ray executors.

The invariant is that original words may widen but never contract or disappear.
This deliberately preserves alignment errors as well as correct words; lower DER
or better boundary accuracy must be established against independent references.
