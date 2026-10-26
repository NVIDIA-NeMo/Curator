# Refine aligned words with Silero and Sortformer

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

## License

Apache License 2.0. See `LICENSE`.

## Source and integration

Ported from NVIDIA's Apache-2.0 `vad-refinement-pipeline` 0.2.0, revision
`53329a018a049ec7fcf1d189f649df74557d9612`. `UPSTREAM.json` records original
module checksums. Module imports and launch commands use the Curator recipe
namespace; the detector, voting, projection and output algorithms are preserved.
The upstream regression tests are under `tests/tutorials/audio/vad_refinement`.

Run these commands from a Curator source checkout. The recipe is not included in
the installed Curator wheel and does not register a new `ProcessingStage`.
It consumes existing forced alignments; it does not run MFA or infer transcripts.
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
