# Granary v2 full pipelines

Turn recordings into speech clips, ASR transcripts and enriched text in one Ray Data
cluster. One entrypoint, [main.py](main.py), runs one of three pipelines, chosen with `--config-name`:

| Pipeline | Languages | Language ID | ASR (primary → recovery) | Image |
| --- | --- | --- | --- | --- |
| [en_granary_pipeline.yaml](en_granary_pipeline.yaml) | English | SpeechBrain + Whisper | Qwen3-Omni → Qwen3-ASR | [Dockerfile](Dockerfile) |
| [european_granary_pipeline.yaml](european_granary_pipeline.yaml) | 24 European languages | SpeechBrain + Whisper | Qwen3-Omni → Qwen3-ASR (`de es fr it pt ru nl`); Parakeet-v3 → Whisper (`pl cs ro hu el fi da sv`); Whisper → Parakeet-v3 (`lt lv hr et bg sk sl mt uk`) | [Dockerfile](Dockerfile) |
| [indic_granary_pipeline.yaml](indic_granary_pipeline.yaml) | 22 Indic languages | SpeechBrain + Indic Canary + Whisper | Indic Canary (TensorRT-LLM) → Parakeet-Riva (`hi`) or IndicConformer (other languages) | [Dockerfile.trt_llm_granary](Dockerfile.trt_llm_granary) |

The language groups and model pairs follow the production Granary v2 routing. The stages around
the ASR step are identical in all three pipelines, so they and the shared settings live once in
[common.yaml](common.yaml); each pipeline YAML adds only its language ID, ASR stages and output file.
They differ only where the table does:

```mermaid
flowchart TD
    A[Recording manifest] --> B[Decode once: 16 kHz mono waveform + Sortformer diarization]
    B --> C[Silero VAD speech segments]
    C --> D[Sound events + language ID]
    D --> F[Keep this pipeline's languages]
    F --> G[Primary ASR on the segment waveform]
    G --> R{Primary hallucination flagged?}
    R -->|Yes| S[Recovery ASR + recheck]
    S --> H[Selection]
    R -->|No| H
    H --> E[Write Opus clips + clips.jsonl]
    E --> J[Cleanup]
    J --> I[Nine text operations on the Dynamo server]
    I --> L[english / european / indic .jsonl]
```

The recording is decoded and resampled once; each segment carries its own waveform through language ID and ASR,
and the Opus clip is written afterwards. Only clips whose selected `source_lang` is in the pipeline's
language set continue to ASR, so rows for other languages and no-speech recordings are dropped without an
audit row. Unrecovered ASR and
text rejection flags stay in the final manifest, so the output is an audit manifest, not a guarantee
that every row suits training.

**Language ID.** `SelectAudioLanguageStage` accepts Whisper's answer when it is `en`; otherwise it
needs SpeechBrain and Whisper to agree, and Indic languages need Indic Canary to agree as well. The
English and European pipelines therefore drop the Canary stage and its TensorRT-LLM engine, which the
Indic pipeline keeps.

**ASR routing.** Each European ASR stage handles only its language group
(`supported_language_codes`) and skips rows that already have a prediction
(`skip_if_output_exists`). By default recovery stages run only on rows the primary check flagged; pass
`recovery_only_if_hallucinated=false` to run them on every row, as the reference pipeline does. A stage that
does not support a row's language adds the audit note `skipped (unsupported language: xx)` under its
own name in `additional_notes`; the stage that does support the language replaces any such
prediction-key note, so selection sees only the model that ran.

Use `backend=ray_data` (the default here). Curator raises an error for GPU stages under Xenna while the managed inference server is active.

## 1. Build the image

Run commands from this checkout's root. The default configuration needs **16 H100 80 GB GPUs**,
256 logical CPUs, 500 GiB shared memory for Ray's object store and more host RAM (2 TiB
recommended): four Dynamo replicas reserve two GPUs each, and the other eight GPUs serve the audio
stages. Docker and the NVIDIA Container Toolkit are required.

Build the images from the repository root, from public base images. English and European:

```bash
docker build -f docker/Dockerfile --target nemo_curator \
  --build-arg CURATOR_EXTRA="audio_tensorrt --extra inference_server" --build-arg CURATOR_GROUPS="" \
  -t curator-base .
docker build -f tutorials/audio/granary_v2/full_pipeline/Dockerfile -t curator-granary-full .
```

Indic:

```bash
docker build -f docker/audio/Dockerfile.trt_llm --target granary -t curator-granary-base .
docker build -f tutorials/audio/granary_v2/full_pipeline/Dockerfile.trt_llm_granary -t curator-granary-indic .
```

- [Dockerfile](Dockerfile) extends the repository's [docker/Dockerfile](../../../../docker/Dockerfile)
  (CUDA 12.9, Python 3.13, Torch 2.11, vLLM 0.22, Dynamo, etcd/NATS), built with the `audio_tensorrt`
  and `inference_server` extras. It adds the NeMo encoder the eight-speaker Sortformer checkpoint needs
  and Ubuntu's `ffmpeg` (libopus included), which torchcodec and the Opus clips need.
- [Dockerfile.trt_llm_granary](Dockerfile.trt_llm_granary) extends the Granary image from
  [docker/audio/Dockerfile.trt_llm](../../../../docker/audio/Dockerfile.trt_llm) (`--target granary`: Torch 2.9.1,
  TensorRT-LLM 1.2.1) with `ai-dynamo`, etcd/NATS and the CUDA 12.9 compiler. It adds no in-process vLLM:
  the Indic ASR stages run on TensorRT-LLM and NeMo.

The English and European image needs a host driver that supports CUDA 12.9 (R575 or newer).
Dynamo's text-model workers run in isolated Ray environments (Transformers 5.8.0 from the YAML) and
resolve their packages at startup, so neither image freezes them.

`main.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` before Ray starts, because vLLM
tensor-parallel CUDA graphs need CUDA-IPC-compatible allocations.

## 2. Supply models and a recording manifest

`model_dir` holds the prepared models described in the metadata tutorial's
[model preparation](../metadata/README.md#1-build-and-prepare-the-shared-environment). Build the Silero and SED
TensorRT engines inside the image you will run (`curator-granary-full` for English and European,
`curator-granary-indic` for Indic): TensorRT engines only load under the TensorRT version that built them,
and the two images differ. The directory holds: `sortformer.nemo`, `silero.plan`,
`sed.pth`, `sed.plan`, `speechbrain/` and `whisper.pt`. The Indic pipeline also needs
`indic_canary/` (the prebuilt TensorRT-LLM engine) and `indic_parakeet/` (`model.nemo` and its
TensorRT engine). Qwen3-Omni, the text model and, for the Indic and European recovery models,
IndicConformer, Parakeet-v3 and Whisper large-v3 download from Hugging Face on first use unless
already cached; set `model_id` and `model_identifier` to local snapshot directories to avoid that.

Use a UTF-8 JSONL manifest, one recording per line, with a unique `audio_item_id` and an absolute
`audio_filepath` visible to every audio worker:

```json
{"audio_item_id":"recording-001","audio_filepath":"/data/recording-001.wav"}
```

Extra provenance fields are preserved. No reference transcript is required, and none is used as a
segment label. Mount the model and recording directories at the **same absolute path** inside the
container so manifest paths still resolve.

## 3. Run

Choose a fresh output directory; the entrypoint refuses to overwrite an existing run.

```bash
export MODELS=/absolute/path/to/prepared/models
export DATA=/absolute/path/to/recordings
export OUTPUT="$PWD/granary-output"
mkdir -p "$PWD/.cache/granary" "$OUTPUT"
docker run --rm --gpus all --ipc=host \
  -v "$MODELS:$MODELS:ro" -v "$DATA:$DATA:ro" \
  -v "$PWD/.cache/granary:/cache" -v "$OUTPUT:/output" \
  curator-granary-full bash -lc '
    source /opt/venv/env.sh
    set -euo pipefail
    python -u tutorials/audio/granary_v2/full_pipeline/main.py "$@" 2>&1 | tee /output/pipeline.log
  ' -- --config-name en_granary_pipeline \
  manifest_path="$DATA/manifest.jsonl" model_dir="$MODELS" \
  output_dir=/output/run cache_root=/cache/compilation
```

Use `--config-name european_granary_pipeline` with the same image, or `--config-name
indic_granary_pipeline` with `curator-granary-indic`. For fewer GPUs, override
`ray_client.num_gpus`, `ray_client.num_cpus` and `inference_server.models.0.num_replicas`.

The entrypoint starts one Ray cluster, downloads the pinned text model snapshot and enters the YAML's
`InferenceServer` context. Dynamo handles placement, readiness and routing; the pipeline starts after
the endpoint is ready and reuses it for all nine text operations. On completion or failure the
entrypoint closes the server before Ray. Model downloads and worker environment setup can take
several minutes on first use.

## 4. Outputs

`$OUTPUT/run/` contains:

| Path | Contents |
| --- | --- |
| `clips.jsonl` | Opus clips with primary/recovery predictions and selection, before text enrichment |
| `english.jsonl`, `european.jsonl` or `indic.jsonl` | Selected clips with primary/recovery predictions, selection source, enriched text and rejection notes |
| `audio/` | Opus clips for the selected languages |
| `rttm/` | Recording-level speaker turns |
| `speechbrain_cache/` | Language-model loading cache |

Check that final rows map to clips and inspect flags separately from successful enrichment
(pass the output file name and the pipeline's language codes):

```bash
docker run --rm -i -v "$OUTPUT:/output:ro" --entrypoint python curator-granary-full - \
  english.jsonl en <<'PY'
import json
import sys
from pathlib import Path

root = Path('/output/run')
name, langs = sys.argv[1], set(sys.argv[2].split(','))
clips = [json.loads(line) for line in (root / 'clips.jsonl').read_text().splitlines()]
final = [json.loads(line) for line in (root / name).read_text().splitlines()]
expected = {row['audio_item_id']: row for row in clips}
actual = {row['audio_item_id']: row for row in final}
assert len(actual) == len(final) == len(expected)
assert actual.keys() == expected.keys()
fields = (
    'primary_model_prediction', 'best_prediction',
    'abbreviated_text', 'entity_recovered_text',
    'pnc_text', 'llm_language_prediction', 'tn_raw', 'captioning_text',
    'code_switched_text', 'speech_qa_text', 'itn_raw', 'itn_clean',
)
flagged = 0
for clip_id, row in actual.items():
    assert row['source_lang'] in langs
    if row.get('best_prediction_source') == 'fallback':
        assert row.get('fallback_model_prediction')
    for key in ('audio_filepath', 'original_file', 'start_ms', 'end_ms'):
        assert row[key] == expected[clip_id][key], (clip_id, key)
    assert Path(row['audio_filepath']).is_file()
    if row.get('_skipme'):
        flagged += 1
        print('FLAGGED', clip_id, row['_skipme'], row.get('additional_notes'))
    else:
        assert all(isinstance(row.get(key), str) and row[key].strip() for key in fields), row
print(f'{len(clips)} clips; {len(final)} final rows; {flagged} flagged')
assert final, 'No clips selected; inspect the input audio and language routing'
PY
```

For the other pipelines pass `european.jsonl` or `indic.jsonl` and the matching comma-separated
language codes (for example `de,es,fr,it,pt,ru,nl,pl,cs,ro,hu,el,fi,da,sv,lt,lv,hr,et,bg,sk,sl,mt,uk`).

Empty output can reflect no speech, language rejection or the wrong pipeline for the input; run the
`granary_v2/metadata` pipeline on the same input to see rejected and no-speech rows before changing
settings. Nonempty fields and successful requests do not establish
transcript accuracy or semantic quality; review generated text and `additional_notes` first.

## 5. Validation status

CPU checks cover configuration loading, building each YAML's pipeline and importing every stage and
adapter class with its third-party dependencies in the matching image. On a local GPU, language ID, Parakeet-v3,
Whisper and Qwen3-ASR (vLLM 0.22) transcribed real European audio. Multi-GPU execution, Qwen3-Omni, the
Indic TensorRT-LLM engines and the end-to-end single-cluster runs are **not** verified: GPU scaling,
transcript quality and the language-group routing on real data remain unproven. The Indic image inherits
dependency declaration conflicts from the Granary base, so plain `uv pip check` is not clean there.
