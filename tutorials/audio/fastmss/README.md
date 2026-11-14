# English audio and text to simulated meetings

This example connects existing Curator components:

```text
audio + supplied text + speaker ID
    -> Curator MFAAlignmentStage
    -> successful word CTM + aligned JSONL
    -> Lhotse MonoCuts with word alignments
    -> one JSONL request per meeting
    -> native FastMSS in Curator
    -> GetAudioDurationStage -> duration filter -> ManifestWriterStage
```

The example uses the module-side FastMSS integration in this branch. It does not
extend the agent, introduce a second MFA implementation, or transcribe the input.
Use existing environments; the script does not install packages or fetch models.

## Input

Provide a local JSONL manifest, one whole single-speaker WAV clip per row:

```json
{"audio_filepath":"/data/alice/clip.wav","text":"The supplied English transcript.","speaker":"alice"}
{"audio_filepath":"/data/bob/clip.wav","text":"Another supplied English transcript.","speaker":"bob"}
```

Use absolute paths, mono 16-kHz WAVs no longer than 20 seconds, accurate English
text and stable speaker IDs without whitespace, such as `Speaker_0` and
`Speaker_1`. Include at least two distinct speakers. This
small example takes the first 32 rows by default; arrange that prefix to contain
both speakers. It accepts whole clips: materialize excerpts with nonzero offsets
before using this example. Existing Curator resampling/mono stages can prepare
other audio formats; match their output rate to 16000 explicitly.
FFmpeg must be on `PATH` for the existing resampling and audio-export components.

MFA and FastMSS can use different existing environments. Phase 1 needs Curator's
`audio_common` dependencies, including Praatio, and an MFA 3.x executable. Phases
2–3 need Curator's `audio_simulation` extra and the pinned FastMSS distribution.
FastMSS's uv git source pin applies when installing from this checkout with uv;
plain `pip install fastmss` is not an equivalent installation route. MFA 3.4.1 is
the version used for the integration check. Pass an existing matching English
dictionary and acoustic model, for example local `english_us_arpa` files.

See the official [MFA alignment guide](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html)
for MFA's model and corpus conventions. The [official English demo](https://github.com/MontrealCorpusTools/librispeech-demo/tree/v1.0.0)
provides real speech and transcripts for a small demonstration.

## 1. Align the supplied transcripts with MFA

Run from the Curator checkout in an environment that can import the existing MFA
stage. Every output directory below must be new.

```bash
python -m tutorials.audio.fastmss.mfa_to_fastmss align \
  --input-manifest /data/english_audio_text.jsonl \
  --output-dir /data/example/aligned \
  --mfa-command /path/to/mfa-env/bin/mfa \
  --dictionary /models/english_us_arpa.dict \
  --acoustic-model /models/english_us_arpa.zip \
  --num-jobs 1 --limit 32
```

This reuses `MFAAlignmentStage`, including its timeout and subprocess cleanup.
Its working state stays under the example's output directory; the model files
are read from their supplied paths. Successful results are saved through the
standard `ManifestWriterStage` as `aligned/aligned.jsonl`, with paths to word
CTM, TextGrid and single-speaker RTTM files. If MFA skips a clip, the example
stops and records `failed_clips.json`. The stage's duration-based fallback CTM
is never accepted as a word alignment.

## 2. Prepare native FastMSS inputs

Switch to the existing environment with the `audio_simulation` dependencies:

```bash
python -m tutorials.audio.fastmss.mfa_to_fastmss prepare \
  --aligned-manifest /data/example/aligned/aligned.jsonl \
  --output-dir /data/example/native_inputs \
  --meetings 4 --duration 10
```

This validates word timing, preserves supplied text and speaker IDs, copies WAVs
into an immutable input tree and writes `speech_train.jsonl.gz` as genuine
embedded Lhotse MonoCuts. Each supervision contains `alignment.word` entries in
clip-relative seconds. No waveform is synthesized during this phase.
The request adapter rejects lazy Lhotse transforms, including recordings with
embedded RIR recordings. Apply transforms beforehand and supply materialized
local WAVs with matching alignments; nested external dependencies are not supported.
The adapter validates every supplied cut, including clips not counted toward
speaker capacity. Its declared recording rate must match native `samplerate`;
the pinned mixer does not resample mismatched inputs. Each cut must contain
exactly one supervision covering the complete cut, with nonempty text and
ordered in-bounds word alignment. Split or materialize multi-annotation clips
before using this adapter. These are metadata checks; waveform samples and
transcript accuracy are not inspected during admission.
Cut IDs must be unique across all splits of a native job: FastMSS caches audio
by cut ID. When enabling native `save_spk`, speaker IDs also become filename
components and must contain no `/`, `\` or NUL and at most 200 UTF-8 bytes.
These export restrictions do not rename or restrict path-like RTTM-only labels.

`meeting.yaml` inherits the toolkit's packaged defaults. The small example
explicitly requests two speakers per meeting, allows one source clip per
speaker, and disables FA splitting, noise and RIR generation. These settings
are a tutorial preset; native defaults and native generation code are unchanged.

## 3. Run bounded smoke and generate meetings

```bash
python -m tutorials.audio.fastmss.mfa_to_fastmss simulate \
  --prepared-inputs /data/example/native_inputs \
  --output-dir /data/example/simulation \
  --min-duration 1 --execute
```

The example writes finite request metadata and a worker environment lock, checks
the recipe, performs one-request smoke, then executes the requested small batch
through the standard Ray Actor Pool backend. Omit `--execute` to stop after the
smoke. Use a new simulation output directory for a later execution. Shared paths
and the same installed FastMSS environment must be visible to every worker.
The bounded smoke explicitly uses the same Actor Pool executor. A separate
Xenna deployment needs its Ray dashboard/state API; that is not a requirement
of this Actor Pool example.
Workers receive the fully resolved planning configuration before the selected
meeting count and isolated execution paths are set. A setting such as
`seed: ${n_meetings}` therefore keeps its planned value during one-request smoke.
Frozen YAML and worker temporary files stay below each run's output directory,
including when the caller's `TMPDIR` points into a native input tree.

The final `/data/example/simulation/meetings.jsonl` contains standard AudioTask
rows with `audio_filepath`, `duration`, `text` and `rttm_filepath`, plus a meeting
request identity and preserved native annotations. WAVs and meeting RTTMs live
under `simulation/generated/`. Reference text includes overlapping turns in
supervision start-time/ID order; duration is the actual complete mixture length,
which may exceed the native duration target. `result.json` reports requested and
retained meetings after the configured duration filter. These counts do not
measure retention of the original speech corpus.

## Composition with other stages

- `GetAudioDurationStage`, scalar filters, `ManifestWriterStage` and metadata
  checkpoints accept the generated standard rows. Use matching key settings if
  you rename generated fields.
- `ASRStage` needs its adapter/model configuration and
  `audio_filepath_key="audio_filepath"` for these rows; its default is the
  resampling stage's `resampled_audio_filepath`. Put ASR predictions in
  `pred_text` to keep the reference `text` intact. FastMSS declares that field
  as the existing `reference_text` role. After ASR, configure
  `ComputeWERStage(hypothesis_text_key="pred_text", reference_text_key="text")`;
  remap both keys consistently when using other names. WER on an overlapping
  mixture checks the entire joined reference; it is not speaker-attributed WER.
- A tested resident-audio route is
  `MonoConversionStage(output_sample_rate=16000)` -> configured
  `ASRStage(waveform_key="waveform", sample_rate_key="sample_rate")` ->
  `ComputeWERStage`. The ASR model and adapter are still required. Set
  `fail_on_audio_error=True` and verify a nonempty prediction before treating
  recognition as successful. The file-based ASR route additionally needs the
  decoder required by the installed TorchAudio version (TorchCodec in the
  integration environment).
- `MonoConversionStage` defaults to a strict 48-kHz expectation. Configure
  `output_sample_rate=16000` for the 16-kHz example. Stages requiring resident
  waveform/sample-rate fields need an explicit hydration/conversion step.
- Word/segment, ALM and speaker-separation workflows require their own declared
  alignment/segment/model inputs. A joined meeting reference and an RTTM path
  do not automatically supply those structures.

The MFA stage on the target main is a native ProcessingStage and is not currently
registered as AgentReady. This tutorial deliberately invokes it through its
normal stage lifecycle. It does not claim that a fresh agent can automatically
plan the entire audio-to-MFA-to-FastMSS workflow. The FastMSS portion uses the
existing manifest contracts and module capability cards.

The adapter's documented first-version limits still apply: local prepared
inputs, no noise/RIR generation, and bounded input size and generation batches.
See the module cards for the exact supported scope. Validate acoustic quality
and transcript accuracy on your own data before preparing a training corpus.
