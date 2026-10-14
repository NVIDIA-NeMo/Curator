# Hindi Indic ASR Pipeline

Transcribe Hindi audio with Indic Canary, produce a Parakeet recovery
transcript, check hallucinations, select and clean the final text, and calculate
word error rate (WER) against the input reference.

Like the [audio-tagging tutorial](../tagging/README.md), this tutorial lists each
stage explicitly in [pipeline.yaml](pipeline.yaml). It uses Curator's shared
YAML runner; there is no tutorial-specific inference wrapper.

## Prerequisites

- Linux x86_64, Docker, and the NVIDIA Container Toolkit.
- A GPU compatible with **both** supplied TensorRT engine bundles. The retained
  configuration reserves 50 GiB for Canary and 24 GiB for Parakeet; one 80 GB
  H100 is a suitable memory target when both engines were built for H100.
  Two short clips reduce processing work, not static model memory.
- A driver compatible with the CUDA 13 native libraries. Supported data-center
  deployments may supply NVIDIA's forward-compatibility libraries; this is a
  system prerequisite, not a Python package.
- Compatible **prebuilt Indic Canary and Hindi Parakeet engine bundles**.
  These model artifacts are not included or downloaded by this tutorial.
  Owners of the matching source checkpoints can build them separately; see
  [model files](#model-files) before attempting inference.

The audio samples and their references are public and need no Hugging Face
token. The complete inference run still requires the separately supplied model
artifacts and suitable hardware; it is not a model-download-from-scratch guide.

## Install the environment

Run from the root of this Curator checkout:

```bash
docker build -f docker/audio/Dockerfile.trt_llm -t curator-indic-asr .
docker run --rm --gpus all curator-indic-asr python docker/audio/check_trt_llm.py
```

The [dedicated audio Dockerfile](../../../docker/audio/Dockerfile.trt_llm)
creates one clean Python 3.12 environment with Torch 2.9.1, TensorRT-LLM 1.2.1,
and TensorRT 10.14.1.48.post1. It includes Curator, the audio dependencies,
SoundFile, and the required WAV/FLAC decoding libraries. No pre-existing host
Python environment is used.
The native versions above are pinned; other allowed dependency versions are
resolved at build time. Reuse the built image to retain the same environment.

The build checks installed versions. The second command additionally checks
the native model libraries and performs a CUDA operation; it does not load
your model engines. Fix any failure there before running the pipeline.
Do **not** run root `uv sync` inside this image: it would replace the dedicated
native stack with Curator's normal dependency profile.

Create persistent data and output directories, then start the container.
Replace the engine mount with the absolute path to your model directory:

```bash
mkdir -p indic-asr-work/data indic-asr-work/output indic-asr-work/ray-logs
docker run --rm -it --gpus all --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$PWD/indic-asr-work/data:/data" \
  -v "$PWD/indic-asr-work/output:/outputs" \
  -v "$PWD/indic-asr-work/ray-logs:/tmp/ray" \
  -v /absolute/path/to/engines:/models:ro \
  curator-indic-asr bash
```

Run the following commands **inside this container**, from `/opt/Curator`.
Data, outputs, and Ray worker logs remain in `indic-asr-work` after the
container exits. Docker writes these files as root; host-side edits may need
elevated permissions.

## Download two public Hindi clips

The quick start uses the Hindi validation split of
[Google FLEURS](https://huggingface.co/datasets/google/fleurs), licensed
[CC BY 4.0](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md).
The downloader pins revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`,
verifies file hashes, and preserves the original reference transcripts.

```bash
python tutorials/audio/indic_asr/download_samples.py --output-dir /data/indic_asr
```

It fetches only a small prefix of the audio archive (at most 2 MiB) and the
transcript table, not the entire dataset. It creates these two original mono
16 kHz WAV files and a two-row `manifest.jsonl`:

| Audio file | Duration |
| --- | ---: |
| `audio/10003386135529782184.wav` | 12.24 seconds |
| `audio/10028896103651079035.wav` | 10.26 seconds |
| Total | 22.50 seconds |

The original Hindi references are:

1. **10003386135529782184.wav:** मेजरकेन खाना, भूमध्य सागर में समान क्षेत्रों की तरह, रोटी, सब्जियों और मांस (विशेष रूप से सूअर का मांस) पर आधारित है, और जैतून के तेल का उपयोग करता है।
2. **10028896103651079035.wav:** वैज्ञानिकों का मानना है कि ओसेलोट्स गंध के द्वारा जानवरों का पीछा करते और खाते (शिकार) हैं, यह सूँघते हुए कि वे ज़मीन पर कहाँ रहे होंगे।

Repeating the download command reuses identical files. It refuses to overwrite
conflicting audio or manifests.

## Model files

Mount your compatible engine bundles with this layout:

```text
/models/
├── indic_canary/
│   ├── encoder/encoder.plan
│   ├── encoder/config.json
│   ├── decoder/config.json
│   ├── decoder/rank0.engine
│   ├── decoder/vocab.json
│   ├── preprocessor/config.json
│   └── preprocessor/mel_basis.pt
└── parakeet/
    ├── encoder.plan
    ├── metadata.json
    └── model.nemo
```

Each directory is a complete bundle, not a Hugging Face model ID. TensorRT
plans must match the GPU architecture and native runtime used here; a random
Canary or Parakeet checkpoint is not a replacement for these Hindi engines.
The runtime does not build missing engines automatically.

### Checkpoint provenance and public releases

The [reference launcher at commit `3931b631`](https://gitlab-master.nvidia.com/nkoluguri/granary-v2-asr/-/blob/3931b631b1a9f23215abdb240b647ef6a3a54b34/slurm_pipeline/asr/run_asr_indic_slurm.sh)
names an internal Parakeet `epoch_37.nemo` and prebuilt Canary engines. It does
not identify the Canary source checkpoint or provide public download URLs and
checkpoint hashes. Consequently, it does not establish that the exact weights
used for this tutorial are publicly downloadable.

Related public model pages include NVIDIA's
[Parakeet RNNT collection, with an Indic variant](https://catalog.ngc.nvidia.com/orgs/nvidia/-/collections/parakeet-1-1b-rnnt-multilingual/-),
and the Canary-based [Indic-Transcribe Core](https://huggingface.co/bodhan-ai/indic-transcribe-core)
and [Flex](https://huggingface.co/bodhan-ai/indic-transcribe-flex) releases.
These have their own access and license conditions; the Indic-Transcribe files
are gated. None has been established here as the identical source checkpoint
or a validated replacement for these engines. Do not substitute a model based
only on its family name: for example,
[`nvidia/canary-1b-flash`](https://huggingface.co/nvidia/canary-1b-flash) does not
support Hindi.

### Optional: build bundles from matching checkpoints

**This is a separate model-owner workflow, not part of the two-clip quick
start.** You need the matching checkpoints, a GPU and build stack compatible
with the intended inference deployment, and permission to use the weights.
Build into a new writable directory, then mount its completed bundles at
`/models:ro` for inference. The inference container above deliberately mounts
that destination read-only.

**Parakeet:** this checkout already includes
[`build_indic_parakeet_rnnt_tensorrt_engine.py`](../../../nemo_curator/stages/audio/inference/scripts/build_indic_parakeet_rnnt_tensorrt_engine.py).
In a build environment with CUDA, NeMo ASR, ONNX export support, and the same
TensorRT version as inference, run from the Curator root, replacing both
absolute paths:

```bash
python nemo_curator/stages/audio/inference/scripts/build_indic_parakeet_rnnt_tensorrt_engine.py \
  --model /absolute/path/to/indic-parakeet.nemo \
  --output-dir /absolute/path/to/engines/parakeet \
  --min-batch 1 --opt-batch 8 --max-batch 16 \
  --min-frames 8 --opt-frames 800 --max-frames 4001 \
  --max-symbols-per-step 10 --workspace-gb 8
```

It exports the FP16 encoder, builds TensorRT, checks encoder numerical
agreement, and packages `encoder.plan`, `metadata.json`, and the original
checkpoint as `model.nemo`. Success prints
`INDIC_PARAKEET_RNNT_TENSORRT_ENCODER_PARITY_PASSED`; this checks the encoder,
not full transcription parity. The checkpoint must restore with the installed
NeMo and contain the expected RNN-T encoder, decoder, and joint network.

**Canary:** its converter is **not included in this checkout**. Obtain the
[reference conversion directory pinned at `1e2d6399`](https://github.com/nithinraok/Curator/tree/1e2d639946f28dfd289a0a9456480385d4480c28/scripts/audio/indic_canary_trtllm_conversion)
and follow its README to prepare a separate build environment. It requires
the matching Indic Canary `.nemo` **and the AI4Bharat NeMo fork containing
`CanaryMultilingualTokenizer`**. Its documented stack uses Python 3.12,
official TensorRT-LLM 1.2.1, NumPy 1.26.4, MPI, and CUDA build tools. The
tutorial's inference image uses stock NeMo and NumPy 2.3.5; it is not a
validated substitute for that conversion environment. Do not downgrade or
replace packages inside the inference image to perform conversion.

After that separate setup, run from the reference conversion directory,
replacing the checkpoint, NeMo fork, and output paths:

```bash
python build_engine.py \
  --nemo_model_path /absolute/path/to/indic-canary.nemo \
  --nemo_repo_path /absolute/path/to/indic-canary-NeMo \
  --engine_dir /absolute/path/to/engines/indic_canary \
  --dtype bfloat16 --max_batch_size 64 --max_beam_width 4 \
  --max_audio_seconds 40 --max_prompt_tokens 10
```

Success prints `CANARY_TRTLLM_ENGINE_BUILD_PASSED`. Always pass the matching
Hindi checkpoint; the converter's default public model is not Hindi-capable.
Check all seven Canary files in the layout above, including
`encoder/config.json` and `decoder/rank0.engine`: the converter's final
presence check covers only five files, not the complete runtime contract.
These profiles retain the 40-second window and capacity for the tutorial's
four-beam, 347-new-token decoding settings. Both build commands were checked
against their CLI definitions, but the tutorial validation reuses existing
bundles: it does **not** validate fresh conversion of a public checkpoint.

## Run the two-clip pipeline

The YAML lists all eleven stages, their input/output text keys, model settings,
and file paths. The quick-start defaults use one actor for each ASR model and
a model batch size of two; no stage is skipped.
The Parakeet TensorRT adapter enables NeMo's batched-greedy decoder CUDA
graphs, matching the reference implementation; its non-TensorRT adapter is
unchanged.

```bash
set -o pipefail
test ! -e /outputs/indic_asr_xenna.jsonl && \
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/indic_asr \
  --config-name pipeline \
  dataset_dir=/data/indic_asr \
  indic_canary_engine_dir=/models/indic_canary \
  parakeet_tensorrt_engine_dir=/models/parakeet \
  output_path=/outputs/indic_asr_xenna.jsonl \
  2>&1 | tee /outputs/indic_asr_xenna.log
```

Alternatively, run Ray Data on the same input files and stages:

```bash
set -o pipefail
test ! -e /outputs/indic_asr_ray_data.jsonl && \
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/indic_asr \
  --config-name pipeline \
  dataset_dir=/data/indic_asr \
  indic_canary_engine_dir=/models/indic_canary \
  parakeet_tensorrt_engine_dir=/models/parakeet \
  backend=ray_data output_path=/outputs/indic_asr_ray_data.jsonl \
  2>&1 | tee /outputs/indic_asr_ray_data.log
```

The `test` guard refuses to run if its output JSONL already exists. For another
run, choose a new JSONL and log filename in the command, and use that JSONL in
the check below. The manifest writer itself overwrites existing output, so do
not remove the guard. Run only one pipeline at a time in this container.

`tee` saves console output under `/outputs`; `pipefail` keeps an inference or
log-write failure from appearing successful. Inspect the console log and the
persisted `ray-logs` directory if a command fails. Model loading may dominate
the runtime of this short sample.

Check the result without assuming that either model reproduces the reference
exactly. This checks identity, reference and metadata preservation, usable
unflagged transcripts, and independently recomputes word-level WER without
normalizing spelling or punctuation. For Ray Data, pass
`/outputs/indic_asr_ray_data.jsonl` instead:

```bash
python - /outputs/indic_asr_xenna.jsonl <<'PY'
import json
import math
import sys
from pathlib import Path

inputs = [json.loads(line) for line in Path("/data/indic_asr/manifest.jsonl").read_text().splitlines()]
outputs = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
inputs_by_id = {row["audio_item_id"]: row for row in inputs}
assert len(inputs) == len(inputs_by_id) == len(outputs) == 2
assert set(inputs_by_id) == {row["audio_item_id"] for row in outputs}
for row in outputs:
    original = inputs_by_id[row["audio_item_id"]]
    assert row["reference_text"] == row["granary_v1_prediction"] == original["text"]
    assert row["audio_filepath"] == str((Path("/data/indic_asr") / original["audio_filepath"]).resolve())
    for key in ("duration", "sampling_rate", "source_lang", "dataset", "dataset_revision", "license"):
        assert row[key] == original[key], (row["audio_item_id"], key)
    assert row["_skipme"] == "", (row["audio_item_id"], row["_skipme"], row["additional_notes"])
    for key in ("primary_model_prediction", "fallback_model_prediction",
                "best_prediction", "cleaned_text", "abbreviated_text"):
        assert isinstance(row[key], str) and row[key].strip(), (row["audio_item_id"], key)
    source = row["best_prediction_source"]
    assert source in ("primary", "fallback"), source
    assert row["best_prediction"] == row[f"{source}_model_prediction"]
    wer = row["wer_pct"]
    assert type(wer) in (int, float) and math.isfinite(wer) and wer >= 0
    reference, prediction = original["text"].split(), row["abbreviated_text"].split()
    assert reference
    costs = list(range(len(prediction) + 1))
    for i, reference_word in enumerate(reference, 1):
        next_costs = [i]
        for j, predicted_word in enumerate(prediction, 1):
            next_costs.append(min(costs[j] + 1, next_costs[j - 1] + 1,
                                  costs[j - 1] + (reference_word != predicted_word)))
        costs = next_costs
    assert wer == round(costs[-1] / len(reference) * 100.0, 4), row["audio_item_id"]
    assert "waveform" not in row and "text" not in row
    print(row["audio_item_id"], row["abbreviated_text"], row["wer_pct"])
PY
```

A nonempty `_skipme` means the pipeline flagged that row; it can still finish
successfully while this usability check fails. Inspect `additional_notes` and
both predictions rather than treating a flagged row as a crash. WER may exceed
100% when there are enough inserted words; a passing check validates the
calculation and data flow, not transcription quality.

## Use your own audio

Provide a JSONL manifest with one row per clip:

```json
{"audio_filepath":"audio/my_clip.wav","audio_item_id":"my_clip","duration":6.26,"sampling_rate":16000,"source_lang":"hi","text":"इस ऑडियो का वास्तविक संदर्भ प्रतिलेख"}
```

`duration` must match decoded audio duration within 1 ms; replace the example
value with your file's duration. Relative paths resolve against `dataset_dir`.
Supply Hindi audio at 16 kHz and a nonempty reference in `text`. The preparation
stage converts loaded audio to mono, validates it, preserves the reference,
and initializes the model fields. References are used for WER, not supplied
as ASR prompts.

Use clips between **0.5 and 40 seconds** for this configuration. Split longer
recordings before creating the manifest: the Canary adapter truncates input
after 40 seconds, even though Parakeet's engine mode can process longer audio.

## Output and troubleshooting

Each output row retains its identity and audio metadata, plus both model
predictions, `best_prediction`, `best_prediction_source`, `cleaned_text`,
`abbreviated_text`, `wer_pct`, `_skipme`, and `additional_notes`.

- **Missing TensorRT-LLM or native import error:** use the dedicated container
  and its smoke check, not the normal `all` or `audio_tensorrt` environment.
- **CUDA or engine deserialization failure:** check the host driver, GPU
  architecture, and the versions used to build both engine bundles.
- **Insufficient GPU memory:** shortening the manifest does not shrink the
  engines; retain the documented model-memory requirements.
- **Missing audio:** verify the manifest and its paths under `dataset_dir`.
- **Missing model file:** compare your bundle to the full layout above.

## What each stage does: one recorded row

This example uses `parakeet_hindi_asr_train_hindi_219834`, a 6.26-second Hindi
clip from [ketav/parakeet-hindi-asr](https://huggingface.co/datasets/ketav/parakeet-hindi-asr/tree/35376a112c4b79318eeaba0c0dd1b6f1a9bf0ea0).
It is a historical input/output pair recorded with Curator commit
[`89cccf46`](https://github.com/NVIDIA-NeMo/Curator/commit/89cccf463f14f382a0be276d8d6cad9096ee70a5),
not expected output for the two FLEURS clips above. Both executors produced
the same final bytes for this identity.

### What was recorded, and what is reconstructed

The run retained the input manifest and final output JSONL, including both ASR
predictions, selected/cleaned text, and stage notes. It did **not** retain a full
`AudioTask` snapshot after every stage: only the final JSONL rows were persisted.
The saved task archives are empty, and the execution logs do not contain this
row's intermediate contents either.

Consequently, the two endpoint JSONL lines below are original recorded bytes;
the stage-by-stage account is a reconstruction from the implementation and the
fields preserved in the final row. The model predictions are recorded values,
not newly generated examples. Waveform arrays, model features, decoder tokens,
and complete earlier task states cannot be recovered as original snapshots.
The waveform is described by shape and type instead of inventing its samples.

For this example, a CPU-only reconstruction decoded the original audio and
replayed the preparation, filtering, selection, cleanup, WER, and writer stages,
using the recorded predictions at the two model-result boundaries. The replayed
writer output matched the captured final line byte for byte. This validates
the reconstruction; it is not a new model-inference run or an original trace.

The hashes below cover each UTF-8 JSONL record **including its trailing LF**;
the JSON code blocks preserve that line, its key order, spacing, and spelling.

| Record | Bytes | SHA-256 |
| --- | ---: | --- |
| Original input | 395 | `e13552b3090921d501773a158387aabcdfeec421809c2864b112bd8af9a2afcd` |
| Final output | 1741 | `35db141c11388becf519185ff56c549c4558935edcea8b853ce66121e292674d` |

### Original input: verbatim

```json
{"audio_filepath":"audio/hindi_219834.flac","audio_item_id":"parakeet_hindi_asr_train_hindi_219834","corpus":"Parakeet Hindi ASR","duration":6.26,"sampling_rate":16000,"source_audio_filepath":"/data/hindi_data/audio/hindi_219834.wav","source_lang":"hi","text":"पिछले दो महीनों की मेरी फाश्टेग बिल रीडिंग दिखाएँ"}
```

The recorded run resolved the relative input path to
`/tmp/curator/datasets/audio_indic_asr_parakeet_hindi_531h_35376a11/audio/hindi_219834.flac`
before its reader. This tutorial resolves relative paths in
`PrepareIndicASRInputStage` using `dataset_dir`; `source_audio_filepath` remains
unchanged. An input path in this historical example is provenance, not a file
that the quick-start commands require.

### Stage-by-stage mutations

To avoid repeating long strings in every row, the table uses these aliases
for the exact recorded text values; the aliases are not fields in the data:

```text
R (reference): पिछले दो महीनों की मेरी फाश्टेग बिल रीडिंग दिखाएँ
P (Canary):    पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएँ
F (Parakeet):  पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएं।
```

Each row below describes changes to `AudioTask.data` for the **same input row**; fields not
mentioned remain unchanged. Nested notes accumulate rather than replacing the
whole `additional_notes` object. Model-stage text and final notes are recorded;
the placement of changes at each boundary is reconstructed. The explicit
[`pipeline.yaml`](pipeline.yaml) lists the stages in this order.

| Stage / function | What it does | Mutation for this row |
| --- | --- | --- |
| 1. `ManifestReader` → `ManifestReaderStage.process()` | Partitions manifest files, parses JSONL, and creates one `AudioTask` per row; it does not decode audio or run ASR. | Carries the input fields and the already-normalized absolute `audio_filepath`. No transcript fields have been added yet. |
| 2. `PrepareIndicASRInputStage.process()` | Loads audio, requires Hindi and a nonempty reference, validates mono 16 kHz audio and duration within 1 ms, and initializes the pipeline fields. | Removes `text`; adds `reference_text=R`, `granary_v1_prediction=R`, `_skipme=""`, and `waveform` (float32, shape `[1, 100159]`). Keeps `sampling_rate=16000`. Initializes notes with `primary_model="indic_canary"` and `recovery_model="parakeet_riva"`. |
| 3. `InferenceIndicCanaryStage` → `ASRStage.process_batch()` / `assemble()` | Transcribes with the TensorRT encoder and TensorRT-LLM decoder: beam size 4, at most 347 new tokens, `pnc=false`. Keeps the waveform for the next model. | Adds `primary_model_prediction=P` and `asr_language="hi"`; reference text is not supplied to the model. The 6.26-second clip does not need the 40-second truncation path. |
| 4. `WhisperHallucinationStage.process()` — primary | Checks repeated words, unusually long words, known hallucination phrases, and excessive character rate. It annotates a row rather than dropping it. | Adds `additional_notes.WhisperHallucination_primary="passed"`; `_skipme` stays empty and `P` is unchanged. |
| 5. `InferenceParakeetStage` → `ASRStage.process_batch()` / `assemble()` | Runs the Parakeet TensorRT encoder with the NeMo/PyTorch RNN-T decoder. It runs on eligible Hindi rows even when the primary check passed. | Adds `fallback_model_prediction=F`, keeps `asr_language="hi"`, and removes `waveform` because `keep_waveform=false`. The different spelling/punctuation in `F` is genuine model output. |
| 6. `WhisperHallucinationStage.process()` — recovery | Checks the fallback transcript. With `overwrite=true`, a valid fallback can clear a prior hallucination flag and record recovery. | Adds `additional_notes.WhisperHallucination_asr="passed"`; no recovery is needed here, so `_skipme` remains empty. |
| 7. `SelectBestPredictionStage.process()` | Chooses a transcript using support/recovery notes and, when needed, cross-model agreement. Ordinary valid rows keep the primary result. Short-audio ground-truth substitution is disabled in this configuration. | Adds `best_prediction=P`, `best_prediction_source="primary"`, and `additional_notes.SelectBestPrediction="used primary"`. No agreement score is calculated for this unflagged row; neither `F` nor `R` replaces `P`. |
| 8. `RegexSubstitutionStage.process()` | Applies the ordered Granary regex rules to the selected transcript, including punctuation/formatting normalization, bracketed-content removal, character filtering, and whitespace cleanup. Already-flagged rows are not normalized. | Adds `cleaned_text=P`. This particular text needs no changes; the stage does not correct the reference spelling. |
| 9. `AbbreviationConcatStage.process()` | Joins eligible spaced single-letter abbreviations using its language-aware rules, and records a note when a join occurs. | Adds `abbreviated_text=P`. There is no eligible abbreviation in this row, so no abbreviation note is added. |
| 10. `GetPairwiseWerStage.process()` | Compares `abbreviated_text` with `reference_text` using word-level edit distance, stores a percentage rounded to four decimal places, and leaves both strings unchanged. | Adds `wer_pct=11.1111`: one substitution (`फाश्टेग` → `फास्टैग`) among nine reference words, or `100 × 1 / 9`. |
| 11. `ManifestWriterStage.process()` | Serializes `task.data` as UTF-8 JSONL using `ensure_ascii=False`, through the single configured writer. | Writes the complete recorded row below, including both predictions and notes; no waveform is present. It does not rewrite the transcripts or omit this row. |

The decoded clip contains 100,159 samples at 16 kHz: 6.2599375 seconds. The
manifest's `duration=6.26` is rounded and remains unchanged in the output.

“Recovery” describes the fallback-selection policy, not conditional execution
of Parakeet. Both models ran on this row. This example follows the normal
primary-selection path: it does not demonstrate hallucination recovery, a regex
edit, or an abbreviation join.

### Final output: verbatim

```json
{"audio_filepath": "/tmp/curator/datasets/audio_indic_asr_parakeet_hindi_531h_35376a11/audio/hindi_219834.flac", "audio_item_id": "parakeet_hindi_asr_train_hindi_219834", "corpus": "Parakeet Hindi ASR", "duration": 6.26, "sampling_rate": 16000, "source_audio_filepath": "/data/hindi_data/audio/hindi_219834.wav", "source_lang": "hi", "additional_notes": {"primary_model": "indic_canary", "recovery_model": "parakeet_riva", "WhisperHallucination_primary": "passed", "WhisperHallucination_asr": "passed", "SelectBestPrediction": "used primary"}, "granary_v1_prediction": "पिछले दो महीनों की मेरी फाश्टेग बिल रीडिंग दिखाएँ", "reference_text": "पिछले दो महीनों की मेरी फाश्टेग बिल रीडिंग दिखाएँ", "_skipme": "", "primary_model_prediction": "पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएँ", "asr_language": "hi", "fallback_model_prediction": "पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएं।", "best_prediction": "पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएँ", "best_prediction_source": "primary", "cleaned_text": "पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएँ", "abbreviated_text": "पिछले दो महीनों की मेरी फास्टैग बिल रीडिंग दिखाएँ", "wer_pct": 11.1111}
```
