# Audio Language Identification Ensemble

Identify the spoken language in each row of a NeMo-style audio manifest with
SpeechBrain, Indic Canary, and Whisper, then select a final language through a
deterministic agreement policy.

The tutorial uses the generic YAML runner in `nemo_curator/config/run.py` and
the same stage-adapter boundary as the maintained sound-event-detection
pipeline. `AudioLIDInferenceStage` owns Curator task I/O, audio normalization,
resume behavior, and result storage. Each adapter owns only its model download,
load, inference, and language normalization.

## Pipeline flow

```text
ManifestReader
    -> SpeechBrainLIDAdapter (primary)
    -> IndicCanaryLIDAdapter (secondary)
    -> WhisperLIDAdapter (tertiary)
    -> SelectAudioLanguageStage
    -> ManifestWriterStage
```

The three inference stages append independent entries to the intermediate
`lid` mapping. The selector does not average model scores; it applies the
precedence rules in [Selection policy](#selection-policy), copies component
evidence into `additional_notes`, and removes `lid` before the final writer.

The implementation supports two interchangeable primary adapters:
SpeechBrain (the maintained YAML default) and AmberNet. AmberNet replaces
SpeechBrain; it is not a fourth simultaneously running ensemble member. Two
maintained configurations cover both reference compositions: `pipeline.yaml`
for primary + Indic Canary + Whisper and `pipeline_non_indic.yaml` for primary
+ Whisper without Canary.

## Prerequisites

- Python 3.12 on x86_64 Linux for the full ensemble.
- A tested, immutable container or isolated environment that combines NeMo
  Curator's audio dependencies with the CUDA stack and
  `tensorrt_llm==1.2.1` required by Indic Canary.
- A CUDA GPU compatible with that environment and the prebuilt engine.
- A prebuilt Indic Canary engine directory available on every executor node.
- Network access on the first run, unless the SpeechBrain and Whisper model
  files are already cached.

Curator's supported `audio_cuda12` extra provides SpeechBrain, Whisper, and
AmberNet for pipelines that omit Indic Canary:

```bash
uv sync --extra audio_cuda12
```

Do not install TensorRT-LLM over Curator's locked environment. Its dependency
pins are intentionally outside the Curator lock. Run the complete Canary
ensemble only from a separately validated Python 3.12 environment or immutable
container that already contains the mutually compatible Curator, CUDA, and
TensorRT-LLM 1.2.1 stack.

The complete ensemble is GPU-oriented. SpeechBrain and Whisper can be used
without Canary for non-Indic routing, subject to the adapter's CPU support and
available memory, but that is a different pipeline composition and does not
enable Indic acceptance.

### Indic Canary engine layout

Set `canary_engine_dir` to the root of a prebuilt engine with these files:

```text
<engine_dir>/
├── encoder/encoder.plan
├── encoder/config.json
├── decoder/rank0.engine
├── decoder/config.json
├── decoder/vocab.json
├── preprocessor/config.json
└── preprocessor/mel_basis.pt
```

The tutorial does not build this engine. Keep the engine on a shared filesystem
or copy the identical build to the same path on every worker node.

### Optional Whisper TensorRT encoder

Whisper uses its PyTorch encoder by default. To build the optional TensorRT
encoder in a Curator environment separate from the full Canary runtime, install
the supported TensorRT extra and build an engine for the deployment GPU:

```bash
uv sync --extra audio_tensorrt

uv run --extra audio_tensorrt python scripts/build_whisper_encoder_tensorrt_engine.py \
  --checkpoint medium \
  --output /models/whisper-medium-encoder.plan \
  --min-batch 1 \
  --opt-batch 8 \
  --max-batch 16 \
  --fp16
```

Make the resulting plan visible at the same path on every executor. TensorRT
engines are specific to the TensorRT version and target GPU. In the separately
validated full-ensemble runtime, select it with
`whisper_backend=tensorrt` and
`whisper_tensorrt_engine_path=/models/whisper-medium-encoder.plan`. The YAML
forwards `batch_size` to Whisper's model-batch limit. The TensorRT wrapper
automatically splits a larger stage window into calls that fit the engine's
maximum batch profile.

## Input manifest

The tutorial does not download a dataset. Supply a JSONL manifest with one
object per line and an `audio_filepath` field:

```json
{"audio_filepath": "/absolute/shared/audio/example.wav", "duration": 8.2}
```

Paths must be readable by every executor worker. `ManifestReader` does not
preflight audio files; `AudioLIDInferenceStage` loads and validates each file in
its worker. The stage converts supported input to contiguous mono float32 audio
at 16 kHz before calling an adapter.

File mode treats every `audio_filepath` as one complete clip. It does not slice
the file from manifest `offset` or `duration` fields. Pipelines that already
segment recordings (for example, a VAD pipeline) should pass the segment through
`waveform_key` and its native rate through `sample_rate_key`; this also lets all
three model stages share one decoded waveform instead of reopening the file.

For programmatic pipelines, set `waveform_key` to consume an in-memory waveform
and set `sample_rate_key` to its source sample-rate field. The maintained YAML
uses file mode (`waveform_key: null`).

## Quick start

Enter the tested full-ensemble environment described above, then run from the
NeMo Curator repository root. Keep the output path different from the input
path because `ManifestWriterStage` truncates its destination during setup.

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/language_identification \
  --config-name pipeline \
  manifest_path=/absolute/path/to/input.jsonl \
  output_path=/absolute/path/to/audio_lid_output.jsonl \
  canary_engine_dir=/absolute/path/to/indic_canary_engine
```

The first run can take several minutes while SpeechBrain and Whisper populate
their model caches. The node-level prefetch hook performs downloads before
worker-local model setup.

## Configuration

| Setting | Default | Purpose |
|---|---:|---|
| `manifest_path` | required | Input JSONL manifest. |
| `output_path` | `./audio_lid_output.jsonl` | Enriched output JSONL manifest. |
| `canary_engine_dir` | required | Prebuilt Indic Canary TensorRT-LLM engine root. |
| `canary_max_duration_sec` | `40.0` | Canary stage and adapter duration cap; keep it within the engine build profile. |
| `primary_backend` | `speechbrain` | Primary classifier (`speechbrain` or `ambernet`). |
| `speechbrain_source` | `speechbrain/lang-id-voxlingua107-ecapa` | SpeechBrain checkpoint source. |
| `ambernet_model_name` | `langid_ambernet` | NeMo model name when `primary_backend=ambernet`. |
| `whisper_model_size` | `medium` | Whisper checkpoint name. |
| `whisper_model_path` | `null` | Optional local Whisper `.pt` checkpoint; takes precedence over `whisper_model_size`. |
| `whisper_fp16` | `true` | Use FP16 Whisper Mel/model inference when the worker is on CUDA. |
| `whisper_backend` | `torch` | Whisper encoder backend (`torch` or `tensorrt`). |
| `whisper_tensorrt_engine_path` | `null` | Required shared `.plan` path when `whisper_backend=tensorrt`. |
| `batch_size` | `16` | Stage window and Canary/Whisper model-batch limit. |
| `backend` | `xenna` | Executor backend (`xenna` or `ray_data`). |
| `execution_mode` | `batch` | Xenna mode; batch serializes the GPU stages. |

Override any setting on the command line, for example:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/language_identification \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/data/output.jsonl \
  canary_engine_dir=/models/canary_trtllm \
  batch_size=8 \
  whisper_backend=tensorrt \
  whisper_tensorrt_engine_path=/models/whisper-medium-encoder.plan
```

Use a smaller `batch_size` or smaller GPU-memory reservation if model actors do
not fit. Do not specify both `gpus` and `gpu_memory_gb` for the same stage.

## Pipeline stages

### SpeechBrain primary

The primary stage selects `SpeechBrainLIDAdapter`, identifies its output as
`SpeechBrainLangID`, and tags it `primary`. The default VoxLingua107 ECAPA
model has broad language coverage. Audio shorter than one second is not
inferred, and the maintained configuration truncates input after 10 seconds.

SpeechBrain labels such as `ta: Tamil` are normalized to lowercase language
codes such as `ta`. Its confidence is a linear probability derived from the
classifier score.

### Indic Canary secondary

The secondary stage selects `IndicCanaryLIDAdapter`, identifies its output as
`IndicCanaryLangID`, and tags it `secondary`. It uses the supplied engine to
emit an Indic language token and accepts up to 40 seconds of audio in this
tutorial.

The engine contract does not expose calibrated generation logits. Therefore a
recognized language token has confidence `1.0`, and a missing language token
has confidence `0.0`. Do not compare that value numerically with SpeechBrain or
Whisper confidence.

### Whisper tertiary

The tertiary stage selects `WhisperLIDAdapter`, identifies its output as
`WhisperLangID`, and tags it `tertiary`. It performs language detection only;
it does not transcribe the sample. The tutorial uses Whisper's 30-second audio
window. `whisper_backend=torch` is the default; the `tensorrt` option replaces
only Whisper's encoder and retains Whisper's decoder and language-token logic.

### Final selector

`SelectAudioLanguageStage` reads the three entries under `lid`, writes the
selected language to `source_lang`, and records the selected confidence as
`additional_notes.source_lid_confidence`. It also copies each component's
model, prediction, and confidence to role-prefixed notes, then consumes the
intermediate `lid` mapping. A rejected row is retained in the manifest and
marked with `_skipme`; it is not silently dropped. If a row already carries an
unrelated upstream skip reason such as `audio_load_error`, the selector
preserves that reason instead of overwriting it with an ensemble rejection.

## Selection policy

The selector applies these rules in order:

1. If no model predictions exist, mark the row skipped.
2. If Whisper predicts `en`, accept Whisper immediately, regardless of the
   other predictions.
3. If Whisper is missing or its language is empty, mark the row skipped.
4. If primary, Indic Canary, and Whisper all predict the same language, accept
   the Indic Canary language and confidence.
5. If Whisper predicts a non-Indic language and the primary model agrees,
   accept the Whisper language and confidence. Canary may be absent for this
   rule.
6. Otherwise, including any Indic prediction without full three-model
   agreement, mark the row skipped for disagreement.

The selector treats these codes as Indic for rule 5:

```text
hi ta bn ur gu mr ml kn te or as pa ne sa sd si kok mai doi ks mni sat brx bo
```

This policy is intentionally categorical. Confidence values from different
model families are not calibrated against one another and do not change the
precedence order.

## Output format

Before selection, each model writes one JSON-serializable result under its
stable `model_id`:

```json
{
  "lid": {
    "SpeechBrainLangID": {"language": "hi", "confidence": 0.91, "tag": "primary"},
    "IndicCanaryLangID": {"language": "hi", "confidence": 1.0, "tag": "secondary"},
    "WhisperLangID": {"language": "hi", "confidence": 0.88, "tag": "tertiary"}
  }
}
```

The selector consumes `lid`, so the final manifest preserves the evidence as
role-prefixed notes instead:

```json
{
  "audio_filepath": "/data/audio/example.wav",
  "source_lang": "hi",
  "additional_notes": {
    "primary_lid_model": "speechbrain",
    "primary_lid_prediction": "hi",
    "primary_lid_confidence": "0.910",
    "secondary_lid_model": "indic_canary",
    "secondary_lid_prediction": "hi",
    "secondary_lid_confidence": "1.000",
    "tertiary_lid_model": "whisper",
    "tertiary_lid_prediction": "hi",
    "tertiary_lid_confidence": "0.880",
    "source_lid_confidence": 1.0,
    "SelectBestLIDPrediction": "used secondary, agreement between all 3 langID models."
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `lid` | object | Intermediate component results keyed by configured `model_id`; consumed by the selector. |
| `lid.<model>.language` | string | Normalized lowercase language code, or an empty string when no language was emitted. |
| `lid.<model>.confidence` | float | Adapter-native confidence; values are not cross-model calibrated. |
| `lid.<model>.tag` | string | Ensemble role: `primary`, `secondary`, or `tertiary`. |
| `source_lang` | string | Selector's final language, or its rejection sentinel on a skipped row. |
| `additional_notes.<role>_lid_*` | string | Final-manifest copy of each component's model, prediction, and three-decimal confidence. |
| `additional_notes.source_lid_confidence` | float | Confidence belonging to the selected component result. |
| `_skipme` | string | Authoritative reason the row was not accepted, when present and non-empty. |

Downstream stages must check `_skipme`; `source_lang` alone is not the skip
signal.

## Resume semantics

Every inference stage in the maintained YAML sets
`skip_if_output_exists: true`. When a task reaches a model stage with an
intermediate `lid` mapping, that stage reuses membership of its own
`lid[model_id]` as the completion marker. Empty language with confidence `0.0`
is a completed result. Another model's result does not count, while any mapping
already stored for this model—including an empty result—is reused.

The final selector deliberately removes `lid`. Therefore the final manifest is
not a component-level resume manifest. Feeding any final row back through the
maintained YAML reruns its models. For a previously rejected row, the inference
stages clear only the selector's three exact rejection reasons before retrying;
an unrelated upstream `_skipme` remains terminal and untouched. To persist
model-level restart data instead, write an intermediate manifest before
`SelectAudioLanguageStage`, then resume from that manifest and write the
completed run to a different path.

With `fail_on_audio_error: false`, unreadable or missing audio produces a
completed empty component result instead of immediately setting `_skipme` or
failing the batch. The selector later applies its normal missing/empty and
agreement rules. Set `fail_on_audio_error: true` in a custom pipeline when an
audio preparation failure must set `_skipme=audio_load_error` immediately. That
upstream reason is preserved by the selector.

## Alternate primary: AmberNet

AmberNet can fill the same primary role as SpeechBrain. Select it in the
maintained YAML rather than adding it as a fourth inference stage:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/language_identification \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/data/output.jsonl \
  canary_engine_dir=/models/canary_trtllm \
  primary_backend=ambernet \
  ambernet_model_name=langid_ambernet
```

AmberNet uses NeMo ASR's `EncDecSpeakerLabelModel`, which is already included in
the audio extra. See the
[NVIDIA LangID AmberNet NGC model card and license](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/langid_ambernet/-).
Its supported language inventory is smaller than VoxLingua107; verify that it
covers the dataset before substituting it.

## Non-Indic mode without Canary

For a dataset where Indic acceptance is not required, run the maintained
two-model configuration. It omits Canary and tags Whisper `secondary`, matching
the reference pipeline without `--indic`:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/language_identification \
  --config-name pipeline_non_indic \
  manifest_path=/data/input.jsonl \
  output_path=/data/output.jsonl
```

The remaining accepted routes are:

- Whisper English, which wins immediately.
- A non-Indic language on which the primary and Whisper agree.

An Indic language cannot be accepted in this mode because the selector
requires full primary + Canary + Whisper agreement for Indic predictions.

## Performance and scaling

Model download and setup dominate very small manifests. For larger manifests,
throughput depends on clip duration, Whisper model size, engine build, GPU, and
batch size. Benchmark the exact deployment rather than relying on a
cross-hardware files-per-second number.

The maintained single-GPU configuration uses
`backend=xenna execution_mode=batch`. Batch mode materializes each model stage
before the next one starts, so the full-GPU Canary actor does not compete with
resident fractional SpeechBrain or Whisper actors. A multi-GPU deployment can
select Ray Data streaming only after sizing the aggregate concurrent GPU
reservations. Keep one identical model cache and Canary engine available to
each node in multi-node runs.

The maintained YAML caps SpeechBrain, Indic Canary, and Whisper at 2, 1, and 2
workers, respectively. These explicit actor caps prevent each heavyweight model
from scaling to every available slot. Tune them together with GPU reservations,
model-cache placement, and the Canary engine profile for your cluster.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Hydra reports that `canary_engine_dir` is missing | The full pipeline requires a local engine | Pass `canary_engine_dir=/absolute/path/to/engine`, or use the documented non-Indic composition. |
| Import reports that `tensorrt_llm` is missing | The full ensemble was started from Curator's standard locked environment | Use the tested immutable container or isolated Python 3.12 full-ensemble environment; do not overlay TensorRT-LLM onto the Curator venv. |
| Canary setup reports missing files | Engine directory is incomplete or points one level too high/low | Compare it with the required seven-file layout above. |
| First run appears idle | Model weights are downloading during node prefetch | Inspect the Hugging Face, NeMo, and Whisper caches and wait for setup to finish. |
| CUDA out of memory | Batch size, Whisper size, or actor reservations are too large | Reduce `batch_size`, use a smaller Whisper checkpoint, or avoid overlapping model actors. |
| Row has `_skipme` despite predictions | The categorical agreement policy rejected it | Inspect the role-prefixed LID entries in `additional_notes` and apply the precedence list above. |
| A final manifest reruns every model | The selector consumes its intermediate `lid` mapping | Persist a manifest before the selector when component-level resume is required. |
| Output is empty or overwritten | Input and output paths were the same | Restore the input manifest and write to a different output path. |

## Models and licenses

- [SpeechBrain VoxLingua107 ECAPA model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- [NVIDIA LangID AmberNet NGC model card and license](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/langid_ambernet/-)
- [OpenAI Whisper repository](https://github.com/openai/whisper)
- [NVIDIA Canary model collection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/canary)

Review each model card, dataset provenance, and license before redistributing
weights or generated metadata. NeMo Curator's Apache-2.0 license does not
replace the terms of third-party checkpoints or engine artifacts.
