# Curator Branch Comparison

**Our branch**: `aaftabv/standardize-audio-stages` (on `mohammadaaftabv/Curator`)
**Reference branch**: `nkoluguri/integration-test` (on `nithinraok/Curator`)
**Common ancestor**: `1cbb4a62` (28 commits in ref since, 15 in ours since)

**Diff summary**: 72 files changed, +5668 / -2124 lines

---

## Table of Contents

1. [High-Level Philosophy Differences](#1-high-level-philosophy-differences)
2. [Features Only in Reference (Removed by Us)](#2-features-only-in-reference-removed-by-us)
3. [Features Only in Ours (Added by Us)](#3-features-only-in-ours-added-by-us)
4. [Shared Features with Different Implementations](#4-shared-features-with-different-implementations)
5. [Naming & Convention Differences](#5-naming--convention-differences)
6. [Dependency Differences](#6-dependency-differences)
7. [Test Coverage Differences](#7-test-coverage-differences)
8. [Entry Point & Pipeline Assembly Differences](#8-entry-point--pipeline-assembly-differences)
9. [Recommended Merge Strategy](#9-recommended-merge-strategy)

---

## 1. High-Level Philosophy Differences

| Aspect | Reference (`nkoluguri/integration-test`) | Ours (`aaftabv/standardize-audio-stages`) |
|---|---|---|
| **Decision tracking** | `additional_notes` dict with per-stage entries via `pipeline_utils.set_note()` | Removed entirely — stages set `_skip_me` but don't log reasons to a dict |
| **Skip key name** | `_skipme` (NeMo/lhotse convention per latest PR #8) | `_skip_me` (Python naming convention) |
| **Multi-node scaling** | `num_workers` parameter on GPU stages + `xenna_stage_spec()` override | Removed `num_workers` and `xenna_stage_spec()` — lets autoscaler decide |
| **SED (Sound Event Detection)** | Full CNN14/PANNs inference + postprocessing pipeline | Completely removed — SED is out of scope for Granary v2 |
| **Segment extraction** | Has `segment_extractor.py` in segmentation/ | Has `SegmentExtractionStage` in io/ (605 lines, much more comprehensive) |
| **Module organization** | Some I/O stages in `alm/` subpackage | Moved I/O stages to dedicated `io/` subpackage |
| **Sample rate key** | `sampling_rate` (NeMo convention) | `sample_rate` (simpler, more standard) |
| **Language support** | `LANG_CODE_TO_NAME` in shared `pipeline_utils.py` | Inlined into `qwen_omni.py`, imported from there by `qwen_asr.py` |
| **Lazy imports** | Segmentation uses `__getattr__` pattern | Direct imports (simpler but adds hard deps) |
| **PnC per-language** | `source_lang_key` on PnC stage + `{language}` placeholder in prompts | Removed — single prompt template without `{language}` placeholder |
| **Hydra entry point** | Not present (only `run_pipeline.py` argparse) | `tutorials/audio/qwen_omni_inprocess/main.py` (Hydra config-driven) |

---

## 2. Features Only in Reference (Removed by Us)

### 2.1 Sound Event Detection (SED)

**Files in reference only:**
- `nemo_curator/stages/audio/inference/sed.py` (274 lines)
- `nemo_curator/stages/audio/inference/sed_models/__init__.py`
- `nemo_curator/stages/audio/inference/sed_models/cnn14.py` (352 lines)
- `nemo_curator/stages/audio/postprocessing/sed_postprocessing.py` (160 lines)
- `nemo_curator/stages/audio/postprocessing/sed_utils.py` (275 lines)

**What it does:**
- `SEDInferenceStage` — runs PANNs CNN14 model on each audio waveform, produces per-frame (T×527) AudioSet class probabilities
- `SEDPostprocessingStage` — applies thresholds, minimum duration, merge gaps to produce event labels (speech, music, noise, etc.)
- Supports saving results as `.npz` sidecar files

**Why we removed it:**
- SED is not part of the Granary v2 transcription pipeline
- Adds a large dependency (`torchlibrosa`) and GPU memory requirement
- The `run_pipeline.py` in ref conditionally adds SED stages if `--sed_checkpoint` is provided; we removed all SED CLI args

### 2.2 `pipeline_utils.py` — Decision Tracking System

**File in reference only:** `nemo_curator/stages/audio/pipeline_utils.py` (53 lines)

**What it provides:**
```python
def set_note(task_data: dict, stage_name: str, value: str, notes_key: str = "additional_notes") -> None:
    """Write a stage decision note into the additional_notes dict."""
    notes = task_data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        task_data[notes_key] = notes
    notes[stage_name] = value
```

Plus `LANG_CODE_TO_NAME` dict (language codes → English names).

**How reference uses it:**
Every stage (WhisperHallucination, PnC, ITN, InitializeFields) calls `set_note()` to record WHY a decision was made:
- `"skipped (flagged)"`, `"skipped (empty)"`, `"applied (modified)"`, `"applied (unchanged)"`, `"hallucination (unique_words, long_word)"`, `"fallback (length_mismatch)"`, etc.

**Why we removed it:**
- Adds per-sample overhead (dict creation + string assignment for every task)
- Output JSONL already has `_skip_me` field which indicates skipped status
- For debugging, log messages provide the same info without polluting data
- Keeping output manifests clean (fewer keys = smaller JSONL)

### 2.3 `num_workers` + `xenna_stage_spec()` — Manual Worker Scaling

**In reference, every GPU stage has:**
```python
num_workers: int | None = None

def xenna_stage_spec(self) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    if self.num_workers is not None:
        spec["num_workers"] = self.num_workers
    return spec
```

Plus CLI args: `--omni_num_workers`, `--asr_num_workers`, `--pnc_num_workers`, `--itn_num_workers`

**In ours:** Completely removed. The Xenna autoscaler handles worker count based on available GPUs and resource requests.

**Why we removed it:**
- Manual worker counts are fragile (wrong value = underutilization or OOM)
- Autoscaler adapts to different hardware (L40S vs A100 vs H100)
- Removes 4 CLI args and 4 repeated method definitions

### 2.4 `autoscale_interval_s` — Scaling Interval Parameter

**In reference:** `--autoscale_interval_s` arg (default 180s) passed to XennaExecutor config.

**In ours:** Removed — uses Xenna default.

### 2.5 `pnc_source_lang_key` — Per-Sample Language for PnC

**In reference:** PnC stage accepts `source_lang_key` and passes `languages` to `QwenTextLLM.generate()`. The prompt template uses `{language}` placeholder for per-sample language resolution.

**In ours:** Removed. PnC prompt uses `{text}` only. All samples get the same language-agnostic PnC restoration.

**Impact:** Reference supports multilingual PnC where the prompt says e.g. "Add punctuation to this Italian text: {text}". Ours assumes English or language-agnostic prompts.

### 2.6 Granary v1 `_skipme` Preservation

**In reference (PR #8):**
```python
def _init_task(self, task: AudioTask) -> None:
    v1_skipme = task.data.get(self.skip_me_key, "")
    notes = task.data.get(self.notes_key, {})
    if v1_skipme:
        notes["v1_skipme"] = v1_skipme
    task.data[self.notes_key] = notes
```

InitializeFields preserves any existing `_skipme` value into `additional_notes["v1_skipme"]` before resetting.

**In ours:** Just sets `_skip_me = ""` unconditionally. No preservation.

### 2.7 Extra `drop_keys` in InitializeFields

**In reference:** `_DEFAULT_DROP_KEYS` includes `"selected_transcript"`, `"taskname"`, `"orig_text"`, `"orig_answer"` (4 extra keys).

**In ours:** Only drops `"pnc"`, `"itn"`, `"timestamp"`.

### 2.8 `segment_extractor.py` (in segmentation/)

**In reference:** `nemo_curator/stages/audio/segmentation/segment_extractor.py` (86 lines) — simpler segment extraction.

**In ours:** Removed from segmentation, replaced by comprehensive `io/extract_segments.py` (605 lines).

---

## 3. Features Only in Ours (Added by Us)

### 3.1 Hydra Entry Point (`tutorials/audio/qwen_omni_inprocess/main.py`)

**321 lines.** A Hydra-based entry point that wraps the same pipeline as `run_pipeline.py` but accepts configuration via YAML + overrides instead of argparse.

**Key features:**
- `build_granary_v2_pipeline(cfg: DictConfig)` — constructs the full stage chain from config
- `_prefetch_models()` — parallel HuggingFace model download with `ThreadPoolExecutor`
- Hydra config at `tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml` (118 lines)

**Why added:** NvLLMOps Kratos `run_curator.py` invokes the pipeline via:
```python
subprocess.run([python, "tutorials/audio/{pipeline}/main.py",
    "--config-path=...", "--config-name=...",
    "workspace_dir=...", "input_manifest=..."])
```
Hydra overrides are cleaner than building long argparse command lines.

### 3.2 `SegmentExtractionStage` (`io/extract_segments.py`)

**605 lines.** Comprehensive segment extraction that auto-detects pipeline combo:
- Combo 2: No VAD / VAD only — extracts by `original_start_ms`/`original_end_ms`
- Combo 3: Speaker diarization — extracts per-speaker intervals from `diar_segments`
- Combo 4: VAD + speaker — extracts speaker-segment timestamps

Supports: WAV/FLAC output, parallel processing, manifest rewriting with new paths.

### 3.3 Speaker Diarization Stage (`inference/speaker_diarization/pyannote.py`)

**304 lines.** GPU-accelerated speaker diarization using Pyannote:
- Wraps `pyannote/speaker-diarization-3.1` model
- Produces per-sample `diar_segments` list with speaker labels and timestamps
- Configurable min/max speakers, min segment duration

### 3.4 WhisperX VAD Stage (`inference/vad/whisperx_vad.py`)

**188 lines.** Voice Activity Detection using WhisperX's silero-based VAD:
- Produces VAD segments with start/end timestamps
- Configurable threshold, min speech duration, min silence duration

### 3.5 NeMo ASR Stage (`inference/asr/asr_nemo.py`)

**130 lines.** NeMo framework ASR inference stage:
- Wraps pretrained NeMo ASR models (Conformer, etc.)
- GPU-accelerated batch inference
- Alternative to Qwen-based ASR for certain pipelines

### 3.6 `ALMManifestReader` / `ALMManifestWriter` (moved to `io/`)

Moved from `alm/` to `io/` subpackage for better organization. The `io/` package now contains all I/O stages:
- `alm_manifest_reader.py` (105 lines)
- `alm_manifest_writer.py` (84 lines)
- `nemo_tarred_reader.py`
- `sharded_manifest_writer.py` (118 lines)
- `extract_segments.py` (605 lines)

### 3.7 `ManifestReader` / `ManifestReaderStage` / `ManifestWriterStage` (in `common.py`)

Generic manifest I/O stages added to `common.py`:
- `ManifestReaderStage` — reads JSONL files line-by-line via fsspec
- `ManifestReader` — CompositeStage wrapping FilePartitioning + reading
- `ManifestWriterStage` — appends AudioTasks to JSONL output

### 3.8 `ShardedManifestWriterStage` (moved to `io/`)

**118 lines.** Was in `alm/sharded_manifest_writer.py`, moved to `io/sharded_manifest_writer.py`.

### 3.9 `PIPELINE_DEEP_DIVE.md`

**1161 lines.** Comprehensive documentation of the entire Granary v2 pipeline:
- Stage-by-stage breakdown with input/output schemas
- GPU resource requirements per stage
- Data flow diagrams
- Configuration reference

### 3.10 Benchmarking Script

`benchmarking/scripts/audio_postprocessing_benchmark.py` (221 lines) — performance benchmarking for the audio postprocessing pipeline.

### 3.11 Comprehensive Test Suite

New test files (all `tests/stages/audio/`):
- `inference/test_qwen_asr.py` (121 lines)
- `inference/test_qwen_omni.py` (131 lines)
- `io/test_extract_segments.py` (157 lines)
- `io/test_nemo_tarred_reader.py` (139 lines)
- `io/test_sharded_manifest_writer.py` (97 lines)
- `text_filtering/test_abbreviation_concat.py` (102 lines)
- `text_filtering/test_disfluency_wer_guard.py` (100 lines)
- `text_filtering/test_itn_restoration.py` (135 lines)
- `text_filtering/test_pnc_content_guard.py` (115 lines)
- `text_filtering/test_pnc_restoration.py` (110 lines)
- `text_filtering/test_select_best_prediction.py` (106 lines)

### 3.12 `granary_v2_postprocessing/README.md` and `requirements.txt`

Added documentation and dependency listing for the tutorial.

---

## 4. Shared Features with Different Implementations

### 4.1 `InferenceQwenOmniStage`

| Aspect | Reference | Ours |
|---|---|---|
| `sample_rate_key` | `"sampling_rate"` | `"sample_rate"` |
| `skip_me_key` field | Present (sets `"empty_audio"` on empty waveforms) | Removed — doesn't flag empty audio |
| `num_workers` | Present | Removed |
| `xenna_stage_spec()` | Returns `{"num_workers": N}` | Removed |
| `setup_on_node()` | Creates full model + calls `setup()` | Only does `snapshot_download()` (weight caching) |
| `process_batch()` input validation | No explicit validation | Calls `self.validate_input(task)` |
| Empty audio handling | Returns `skipped_indices`, sets `_skipme="empty_audio"` | Model returns empty string, no flagging |
| `generate()` return | `(pred_texts, disfluency_texts, skipped_indices)` | `(pred_texts, disfluency_texts)` |

### 4.2 `InferenceQwenASRStage`

| Aspect | Reference | Ours |
|---|---|---|
| `sample_rate_key` | `"sampling_rate"` | `"sample_rate"` |
| `gpu_memory_utilization` default | `0.7` | `0.95` |
| `num_workers` | Present | Removed |
| `setup_on_node()` | Creates full model | Only `snapshot_download()` |
| Run-index selection | Inline in `process_batch` | Extracted to `_select_run_indices()` method |
| Language resolution | Inline in `process_batch` | Extracted to `_resolve_languages()` method |
| Input validation | None | `self.validate_input(task)` per task |

### 4.3 `PnCRestorationStage`

| Aspect | Reference | Ours |
|---|---|---|
| `skip_me_key` | `"_skipme"` | `"_skip_me"` |
| `notes_key` | Present (writes decision notes) | Removed |
| `source_lang_key` | Present (passes languages to model) | Removed |
| `num_workers` | Present | Removed |
| `xenna_stage_spec()` | Present | Removed |
| Batch processing | Calls `self._model.generate(eligible_texts, languages=eligible_langs)` once | Chunks into `self.batch_size` sub-batches with for-loop |
| Decision tracking | `set_note()` for "applied (modified)", "applied (unchanged)", "kept as-is (incomplete)" | None |
| Input validation | None | `self.validate_input(task)` per task |
| `_partition_tasks()` | Inline logic | Extracted to dedicated method |

### 4.4 `ITNRestorationStage`

| Aspect | Reference | Ours |
|---|---|---|
| `skip_me_key` | `"_skipme"` | `"_skip_me"` |
| `notes_key` | Present | Removed |
| `num_workers` | Present | Removed |
| `max_output_tokens` default | 512 | 4096 |
| `max_model_len` default | 2048 | 4096 |
| `max_num_seqs` default | 64 | 16 |
| `process()` | Delegates to `process_batch` | Raises `NotImplementedError` |
| Batch logic | Inline in `process_batch` | Extracted to `_collect_prompts()` and `_apply_output()` methods |
| Decision tracking | `set_note()` calls | None |
| Input validation | None | `self.validate_input(task)` per task |

### 4.5 `WhisperHallucinationStage`

| Aspect | Reference | Ours |
|---|---|---|
| `skip_me_key` | `"_skipme"` | `"_skip_me"` |
| `name` field position | After other fields | First field (Python dataclass ordering) |
| Decision tracking | `set_note()` for "hallucination (...)", "recovered", etc. | None |
| Phrase file parsing | Reads lines as-is (`{line.strip() for line in f}`) | Parses trailing integers (removes count suffixes from phrase lists) |

### 4.6 `InitializeFieldsStage`

| Aspect | Reference | Ours |
|---|---|---|
| Skip key | `"_skipme"` | `"_skip_me"` |
| Notes key | Present, preserves v1 `_skipme` into notes | Removed |
| Extra drop keys | `"selected_transcript"`, `"taskname"`, `"orig_text"`, `"orig_answer"` | Not present |
| Internal method | `_init_task()` called by both `process()` and `process_batch()` | Logic duplicated inline in both methods |
| `name` field position | After other fields | First field |

### 4.7 `NemoTarShardReaderStage`

| Aspect | Reference | Ours |
|---|---|---|
| Sample rate key output | Both `"sampling_rate"` and `"sample_rate"` | Only `"sample_rate"` |
| `outputs()` | Includes `"sampling_rate"` | Only `"sample_rate"`, `"corpus"`, `"num_channels"` |
| Tar file handling | No try/finally (tar.close() at end) | `try/finally` ensures tar.close() even on exception |
| RayStageSpecKeys import | Try/except `ModuleNotFoundError` fallback | Direct import from `nemo_curator.backends.utils` |

### 4.8 `QwenTextLLM` (model wrapper)

| Aspect | Reference | Ours |
|---|---|---|
| `_prepare_single()` | Accepts `language` parameter, formats `{text}` + `{language}` | No `language` parameter, formats `{text}` only |
| `_prepare_batch()` | Accepts `languages: list[str] | None` | No `languages` parameter |
| `generate()` | Accepts `languages: list[str] | None` | No `languages` parameter |
| Batch mapping | `pool.map(_prepare_single, texts, templates, langs)` | `pool.map(_prepare_single, texts, templates)` |

### 4.9 `QwenOmni` (model wrapper, `models/qwen_omni.py`)

| Aspect | Reference | Ours |
|---|---|---|
| `generate()` return | `(pred_texts, disfluency_texts, skipped_indices)` | `(pred_texts, disfluency_texts)` |
| Empty audio handling | Tracks `skipped_indices` set, returns it | Returns empty string for empty audio, no tracking |

### 4.10 `QwenASR` (model wrapper, `models/qwen_asr.py`)

| Aspect | Reference | Ours |
|---|---|---|
| Main difference | Minor - mostly same | Same core logic, slightly different error handling |

---

## 5. Naming & Convention Differences

| Convention | Reference | Ours |
|---|---|---|
| Skip key | `_skipme` | `_skip_me` |
| Sample rate key | `sampling_rate` | `sample_rate` |
| Dataclass field ordering | `name` after functional fields | `name` as first field |
| Decision tracking | `additional_notes` dict per task | Not tracked |
| Language lookup location | `pipeline_utils.py` (shared module) | Inlined in `qwen_omni.py` |
| Segmentation imports | Lazy via `__getattr__` | Direct imports |
| `process()` on batch-only stages | Some delegate to `process_batch` | All raise `NotImplementedError` |
| Input validation | Not enforced | `validate_input()` check at start of `process_batch()` |

---

## 6. Dependency Differences

### `pyproject.toml` — `audio_cuda12` extra

| Package | Reference | Ours |
|---|---|---|
| `vllm` | Not in extras (runtime install) | Added to `audio_cuda12` |
| `qwen-omni-utils` | Not in extras | Added to `audio_cuda12` |
| `qwen-asr` | Not in extras | Added to `audio_cuda12` |
| `fasttext` | Not in extras | `fasttext==0.9.3` added to `audio_cuda12` |
| `transformers` constraint | `>=4.56.0,<5.0` (comment: "breaks with hf_hub<1.0") | `>=4.56.0,<5.0` (comment: "upper-bounded by nemo-toolkit[asr] constraint") |

**Impact:** Our branch tries to make `uv sync --extra audio_cuda12` install everything needed for the Granary v2 pipeline. Reference expects runtime `pip install` commands (as in the Slurm sbatch script).

---

## 7. Test Coverage Differences

| Test File | Reference | Ours |
|---|---|---|
| `test_qwen_asr.py` | Not present | 121 lines |
| `test_qwen_omni.py` | Not present | 131 lines |
| `test_extract_segments.py` | Not present | 157 lines |
| `test_nemo_tarred_reader.py` | Not present | 139 lines |
| `test_sharded_manifest_writer.py` | Not present | 97 lines |
| `test_abbreviation_concat.py` | Not present | 102 lines |
| `test_disfluency_wer_guard.py` | Not present | 100 lines |
| `test_itn_restoration.py` | Not present | 135 lines |
| `test_pnc_content_guard.py` | Not present | 115 lines |
| `test_pnc_restoration.py` | Not present | 110 lines |
| `test_select_best_prediction.py` | Not present | 106 lines |
| `test_initialize_fields.py` | Present (modified) | Modified to match `_skip_me` naming |
| `test_fasttext_lid.py` | Present | Modified (75 lines of changes) |
| `test_regex_substitution.py` | Present | Modified (91 lines of changes) |
| `test_whisper_hallucination.py` | Present | Modified (52 lines of changes) |
| `test_common.py` | Present | Modified (10 lines of changes) |

---

## 8. Entry Point & Pipeline Assembly Differences

### Reference: `examples/audio/qwen_omni_inprocess/run_pipeline.py`

- Pure argparse-based CLI
- Optionally includes SED stages (if `--sed_checkpoint` given)
- `InferenceQwenOmniStage` is appended after optional SED block
- Has `--autoscale_interval_s` and `--*_num_workers` args
- Has `--pnc_source_lang_key` arg
- `itn_max_output_tokens` default: 512
- XennaExecutor config includes `autoscale_interval_s`

### Ours: `examples/audio/qwen_omni_inprocess/run_pipeline.py`

- Same argparse CLI but:
  - No SED args or stages
  - No `num_workers` args
  - No `autoscale_interval_s`
  - No `pnc_source_lang_key`
  - `itn_max_output_tokens` default: 4096
  - `InferenceQwenOmniStage` is part of the initial `stages` list (not appended after conditional block)

### Ours (additional): `tutorials/audio/qwen_omni_inprocess/main.py`

- Hydra-based entry point (does NOT exist in reference)
- Uses `build_granary_v2_pipeline(cfg)` function
- `_prefetch_models()` for parallel model downloads
- Designed for NvLLMOps Kratos integration

---

## 9. Recommended Merge Strategy

### Must bring from reference into ours:

1. **`_skipme` → `_skipme`** — Reference's PR #8 standardized to `_skipme` (NeMo/lhotse convention). We should adopt this and drop `_skip_me`.

2. **`sampling_rate` key** — NeMo convention is `sampling_rate`. Since NemoTarredAudioReader feeds into stages that then pass data downstream, we should match NeMo's convention to avoid key mismatches with existing NeMo manifests.

3. **`additional_notes` tracking** — Valuable for debugging production runs. Consider keeping it as optional (controlled by a flag) rather than hardcoded.

4. **Extra `drop_keys`** — `"selected_transcript"`, `"taskname"`, `"orig_text"`, `"orig_answer"` are common in NeMo manifests and should be dropped.

5. **Granary v1 `_skipme` preservation** — Important for incremental processing where data may already have v1 flags.

### Keep from ours (do NOT revert):

1. **Hydra entry point** (`main.py`) — Required for NvLLMOps Kratos integration
2. **`SegmentExtractionStage`** in `io/` — More comprehensive than reference's simpler version
3. **Speaker diarization + WhisperX VAD stages** — New capability not in reference
4. **NeMo ASR stage** — Additional inference backend
5. **Comprehensive test suite** — 11 new test files
6. **`PIPELINE_DEEP_DIVE.md`** — Documentation
7. **`validate_input()` calls** — Better error messages
8. **Extracted helper methods** (`_partition_tasks`, `_collect_prompts`, `_apply_output`, `_select_run_indices`, `_resolve_languages`) — Better code organization
9. **`try/finally` for tar handling** — Prevents resource leaks
10. **Dependencies in `pyproject.toml`** — Makes `uv sync` work without runtime pip

### Items to discuss:

1. **SED stages** — Do we want them? They're useful for audio quality assessment but add complexity and dependencies.
2. **`num_workers` / manual scaling** — Do we need manual override for multi-node? Autoscaler may not always be optimal for bursty GPU workloads.
3. **Per-language PnC** — Do we need multilingual PnC prompt support? Currently only English is targeted.
4. **`phrases.txt`** format — Reference added a `phrases.txt` (65 lines) with trailing integers (counts?). Our parser handles this via `rsplit`; reference reads raw lines. Need to verify format compatibility.

---

## Appendix: File-by-File Quick Reference

### Files only in reference (we deleted):
```
nemo_curator/stages/audio/inference/sed.py
nemo_curator/stages/audio/inference/sed_models/__init__.py
nemo_curator/stages/audio/inference/sed_models/cnn14.py
nemo_curator/stages/audio/postprocessing/sed_postprocessing.py
nemo_curator/stages/audio/postprocessing/sed_utils.py
nemo_curator/stages/audio/pipeline_utils.py
nemo_curator/stages/audio/segmentation/segment_extractor.py
tutorials/audio/granary_v2_postprocessing/phrases.txt
```

### Files only in ours (we added):
```
benchmarking/scripts/audio_postprocessing_benchmark.py
nemo_curator/stages/audio/inference/asr/__init__.py
nemo_curator/stages/audio/inference/asr/asr_nemo.py
nemo_curator/stages/audio/inference/speaker_diarization/__init__.py
nemo_curator/stages/audio/inference/speaker_diarization/pyannote.py
nemo_curator/stages/audio/inference/vad/__init__.py
nemo_curator/stages/audio/inference/vad/whisperx_vad.py
nemo_curator/stages/audio/io/alm_manifest_reader.py
nemo_curator/stages/audio/io/alm_manifest_writer.py
nemo_curator/stages/audio/io/extract_segments.py
nemo_curator/stages/audio/io/sharded_manifest_writer.py
tests/stages/audio/inference/test_qwen_asr.py
tests/stages/audio/inference/test_qwen_omni.py
tests/stages/audio/io/test_extract_segments.py
tests/stages/audio/io/test_nemo_tarred_reader.py
tests/stages/audio/io/test_sharded_manifest_writer.py
tests/stages/audio/text_filtering/test_abbreviation_concat.py
tests/stages/audio/text_filtering/test_disfluency_wer_guard.py
tests/stages/audio/text_filtering/test_itn_restoration.py
tests/stages/audio/text_filtering/test_pnc_content_guard.py
tests/stages/audio/text_filtering/test_pnc_restoration.py
tests/stages/audio/text_filtering/test_select_best_prediction.py
tutorials/audio/granary_v2_postprocessing/README.md
tutorials/audio/granary_v2_postprocessing/requirements.txt
tutorials/audio/qwen_omni_inprocess/PIPELINE_DEEP_DIVE.md
tutorials/audio/qwen_omni_inprocess/main.py
tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml
```

### Files modified in both (diverged):
```
examples/audio/qwen_omni_inprocess/run_pipeline.py
nemo_curator/models/qwen_asr.py
nemo_curator/models/qwen_omni.py
nemo_curator/models/qwen_text_llm.py
nemo_curator/stages/audio/__init__.py
nemo_curator/stages/audio/alm/alm_manifest_reader.py
nemo_curator/stages/audio/alm/alm_manifest_writer.py
nemo_curator/stages/audio/alm/sharded_manifest_writer.py
nemo_curator/stages/audio/common.py
nemo_curator/stages/audio/inference/__init__.py
nemo_curator/stages/audio/inference/qwen_asr.py
nemo_curator/stages/audio/inference/qwen_omni.py
nemo_curator/stages/audio/io/nemo_tarred_reader.py
nemo_curator/stages/audio/metrics/get_wer.py
nemo_curator/stages/audio/segmentation/__init__.py
nemo_curator/stages/audio/segmentation/speaker_separation.py
nemo_curator/stages/audio/segmentation/vad_segmentation.py
nemo_curator/stages/audio/text_filtering/abbreviation_concat.py
nemo_curator/stages/audio/text_filtering/disfluency_wer_guard.py
nemo_curator/stages/audio/text_filtering/fasttext_lid.py
nemo_curator/stages/audio/text_filtering/finalize_fields.py
nemo_curator/stages/audio/text_filtering/initialize_fields.py
nemo_curator/stages/audio/text_filtering/itn_restoration.py
nemo_curator/stages/audio/text_filtering/pnc_content_guard.py
nemo_curator/stages/audio/text_filtering/pnc_restoration.py
nemo_curator/stages/audio/text_filtering/prompts/pnc_prompt.md
nemo_curator/stages/audio/text_filtering/regex_substitution.py
nemo_curator/stages/audio/text_filtering/select_best_prediction.py
nemo_curator/stages/audio/text_filtering/whisper_hallucination.py
pyproject.toml
tutorials/audio/README.md
tutorials/audio/granary_v2_postprocessing/post_processsing_pipeline.py
```
