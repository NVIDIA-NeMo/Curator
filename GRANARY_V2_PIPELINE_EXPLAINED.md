# Granary v2 Pipeline — Complete Technical Explanation

This document explains the entire Granary v2 audio transcription and text filtering pipeline as implemented in the Curator audio pipeline.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Entry Points](#2-entry-points)
3. [Stage-by-Stage Breakdown](#3-stage-by-stage-breakdown)
4. [Model Wrappers](#4-model-wrappers)
5. [Data Flow & Key Mappings](#5-data-flow--key-mappings)
6. [Execution Engine (Xenna)](#6-execution-engine-xenna)
7. [Resume & Checkpointing](#7-resume--checkpointing)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Pipeline Overview

The Granary v2 pipeline takes NeMo-tarred audio datasets (sharded `.tar` archives + JSONL manifests) and produces high-quality transcriptions with punctuation, capitalization, and inverse text normalization.

### High-Level Data Flow

```
NeMo Tarred Data (YAML config → .tar + .jsonl per shard)
    │
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1: NemoTarredAudioReader (CPU)                              │
│   Discover shards → Stream tar → Decode audio in memory           │
│   Output: AudioTask with waveform (float32 numpy) + metadata      │
├────────────────────────────────────────────────────────────────────┤
│ Stage 2: InitializeFieldsStage (CPU)                              │
│   Set _skip_me="", rename text→granary_v1_prediction, drop keys   │
├────────────────────────────────────────────────────────────────────┤
│ Stage 3: InferenceQwenOmniStage (GPU, TP=2)                      │
│   Batched vLLM inference with Qwen3-Omni-30B                     │
│   Output: qwen3_prediction_s1 (+ s2 if follow-up prompt)         │
├────────────────────────────────────────────────────────────────────┤
│ Stage 4: [Optional] DisfluencyWerGuardStage (CPU)                 │
│   Compare Turn 1 vs Turn 2 WER; revert s2 if divergence > 50%    │
├────────────────────────────────────────────────────────────────────┤
│ Stage 5: WhisperHallucinationStage (CPU)                          │
│   Flag hallucinated transcripts via phrase matching, uniqueness,   │
│   long-word detection, char-rate check                            │
├────────────────────────────────────────────────────────────────────┤
│ Stage 6: [Optional] InferenceQwenASRStage (GPU)                   │
│   Re-transcribe ONLY hallucinated samples with Qwen3-ASR-0.6B    │
├────────────────────────────────────────────────────────────────────┤
│ Stage 7: [Optional] WhisperHallucinationStage (CPU, 2nd instance) │
│   Re-check QwenASR output for hallucination                       │
├────────────────────────────────────────────────────────────────────┤
│ Stage 8: SelectBestPredictionStage (CPU)                          │
│   Pick: recovered ASR > cross-model agreement > primary (omni)    │
├────────────────────────────────────────────────────────────────────┤
│ Stage 9: FastTextLIDStage (CPU)                                   │
│   Language ID check; flag wrong-language samples                   │
├────────────────────────────────────────────────────────────────────┤
│ Stage 10: RegexSubstitutionStage (CPU)                            │
│   Apply regex cleanup rules from YAML                             │
├────────────────────────────────────────────────────────────────────┤
│ Stage 11: AbbreviationConcatStage (CPU)                           │
│   "A P I" → "API", handles 40+ languages                         │
├────────────────────────────────────────────────────────────────────┤
│ Stage 12: [Optional] PnCRestorationStage (GPU, TP=1)              │
│   Restore punctuation & capitalization via Qwen3.5-35B-FP8        │
├────────────────────────────────────────────────────────────────────┤
│ Stage 13: [Optional] PnCContentGuardStage (CPU)                   │
│   Revert PnC output if LLM changed actual words                  │
├────────────────────────────────────────────────────────────────────┤
│ Stage 14: [Optional] ITNRestorationStage (GPU, TP=1)              │
│   Inverse text normalization (numbers, dates, symbols)            │
├────────────────────────────────────────────────────────────────────┤
│ Stage 15: ShardedManifestWriterStage (CPU)                        │
│   Write per-shard JSONL + .done markers for resume support        │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output: Per-shard JSONL manifests with all predictions + metadata
```

---

## 2. Entry Points

### 2.1 Argparse CLI — `examples/audio/qwen_omni_inprocess/run_pipeline.py`

**Used for**: Slurm jobs, local development, direct invocation.

```bash
python examples/audio/qwen_omni_inprocess/run_pipeline.py \
    --data_config /path/to/data_config.yaml \
    --output_dir /path/to/output \
    --model_id Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --hall_phrases /path/to/en.txt \
    --regex_yaml /path/to/common.yaml \
    --tensor_parallel_size 2 \
    --batch_size 32 \
    --execution_mode streaming
```

**Key behaviors**:
- Sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `VLLM_LOGGING_LEVEL=ERROR` at module load
- Reads prompt files if `--ml_prompt_file` / `--en_prompt_file` / `--pnc_prompt_file` are provided
- Constructs stage chain based on flags (`--asr_model_id`, `--skip_pnc`, `--enable_itn`, `--followup_prompt`)
- Creates `XennaExecutor` with configured `execution_mode`
- Runs pipeline via `pipeline.run(executor=executor)`

### 2.2 Hydra CLI — `tutorials/audio/qwen_omni_inprocess/main.py`

**Used for**: NvLLMOps Kratos integration (invoked by `run_curator.py`).

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
    --config-path=/path/to/configs \
    --config-name=qwen_omni_inprocess \
    workspace_dir=/work \
    input_manifest=/data/staged_config.yaml \
    hf_token=hf_xxx
```

**Key behaviors**:
- All parameters come from YAML config + Hydra overrides
- `_prefetch_models(cfg)` — downloads all models in parallel before pipeline starts
- Sets `HF_TOKEN` env var from config for HuggingFace Hub access
- Same stage construction logic as argparse entry point but reads from `DictConfig`

### 2.3 Hydra Config — `tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml`

Default config with all tunable parameters. Key sections:
- **CORE**: `input_manifest`, `workspace_dir`, `execution_mode`
- **QWEN-OMNI**: `model_id`, `tensor_parallel_size`, `batch_size`, `max_model_len`
- **TEXT FILTERING**: `hall_phrases`, `fasttext_model`, `regex_yaml`
- **QWEN-ASR**: `asr_model_id` (null = disabled)
- **PNC**: `pnc_model_id`, `pnc_batch_size`, `skip_pnc`
- **ITN**: `enable_itn`, `itn_model_id`

---

## 3. Stage-by-Stage Breakdown

### Stage 1: NemoTarredAudioReader

**File**: `nemo_curator/stages/audio/io/nemo_tarred_reader.py`
**Type**: CompositeStage (decomposes into 2 sub-stages)
**Compute**: CPU only
**Resources**: 4.0 CPUs per worker (configured via `.with_()`)

#### Sub-stage 1a: NemoTarShardDiscoveryStage

**What it does**:
1. Reads the Granary YAML data config (lists corpora with manifest + tar paths)
2. Expands NeMo brace patterns (`_OP_0..511_CL_` → 512 individual paths)
3. Pairs each manifest file with its corresponding tar file
4. Checks for `.done` markers in `output_dir` to skip already-completed shards
5. Emits one `FileGroupTask` per shard with `data = [manifest_path, tar_path]`

**Resume logic**: Scans `output_dir` for `*.jsonl.done` files. If a shard's done file exists, it's skipped entirely. Partial (incomplete) `.jsonl` files are deleted before re-processing.

**Corpus filtering**: If `corpus_filter` is set, only shards matching those corpus names are emitted.

#### Sub-stage 1b: NemoTarShardReaderStage

**What it does**:
1. Reads the JSONL manifest file for the shard (maps filenames → metadata)
2. Opens the tar file via `lhotse`'s `open_best()` (supports local files and S3/AIS streaming via `pipe:curl`)
3. Iterates through tar entries, matching each audio file to its manifest entry
4. Decodes audio in-memory using `soundfile` → float32 numpy array
5. Converts stereo to mono (mean of channels)
6. Emits one `AudioTask` per utterance with:
   - `waveform`: 1-D float32 numpy array (mono)
   - `sample_rate`: int (native rate, typically 16000 or 44100)
   - `corpus`: string (corpus name from config)
   - `num_channels`: int (original channel count)
   - All manifest metadata (duration, text, audio_filepath, etc.)
   - `_metadata["_shard_key"]`: relative path for output routing
   - `_metadata["_shard_total"]`: total utterances in this shard

**S3 support**: Converts `s3://bucket/key` to `pipe:curl -sL 'endpoint/v1/objects/bucket/key?provider=s3'` for streaming. Uses `AIS_ENDPOINT` env var or explicit `s3_endpoint_url`.

---

### Stage 2: InitializeFieldsStage

**File**: `nemo_curator/stages/audio/text_filtering/initialize_fields.py`
**Compute**: CPU
**Resources**: 1.0 CPU

**What it does** (per task):
1. Sets `_skip_me = ""` (empty = not skipped; downstream stages write reason strings here)
2. If `source_lang` key missing, defaults to `"en"`
3. Renames `text` → `granary_v1_prediction` (preserves any existing v1 transcription)
4. Drops stale keys: `answer`, `target_lang`, `decodercontext`, `emotion`, `diarize`, `pnc`, `itn`, `timestamp`

**Why rename text**: The original `text` field may contain a Granary v1 prediction (from a previous pipeline run). Renaming prevents confusion with the new predictions generated by this pipeline.

---

### Stage 3: InferenceQwenOmniStage

**File**: `nemo_curator/stages/audio/inference/qwen_omni.py`
**Model**: `nemo_curator/models/qwen_omni.py` (QwenOmni class)
**Compute**: GPU (TP=2 default → 2 GPUs)
**Resources**: `Resources(gpus=float(tensor_parallel_size))`

**What it does**:
1. **setup_on_node()**: Pre-downloads model weights via `huggingface_hub.snapshot_download()` (no GPU allocation — just caching)
2. **setup()**: Creates vLLM `LLM` engine with specified TP, model_len, memory utilization
3. **process_batch()** (per batch of 32 tasks):
   a. Extracts waveforms + sample rates from tasks
   b. Maps `source_lang` codes to full language names via `_LANG_CODE_TO_NAME` dict
   c. Calls `self._model.generate(waveforms, sample_rates, languages)`
   d. Writes predictions to `qwen3_prediction_s1` (and `qwen3_prediction_s2` if follow-up prompt)
   e. Drops waveform from task data (unless `keep_waveform=True` for ASR recovery)

**Model internals (QwenOmni.generate())**:
1. Resamples each waveform to 16 kHz using `librosa.resample()` (in thread pool)
2. Builds multi-turn conversations:
   - Turn 1: `[system_prompt] + [audio + prompt_text]` → generates `qwen3_prediction_s1`
   - Turn 2 (if `followup_prompt`): `[context of turn 1] + [audio + followup_prompt]` → generates `qwen3_prediction_s2`
3. Uses `qwen_omni_utils.process_mm_info()` to create vLLM-compatible inputs (accepts numpy arrays directly)
4. Batched inference via `vllm.LLM.generate()` with `SamplingParams(temperature=0, top_k=1, max_tokens=256)`

**Prompt strategy**:
- Multilingual: Uses `ml_prompt` (e.g. "Transcribe the audio.") with `{language}` placeholder resolved per-sample
- English-specific: If `en_prompt_text` is set, English samples get this prompt, others get `ml_prompt`
- Follow-up (disfluency): Turn 2 asks model to add filler words, false starts, colloquial forms

#### Qwen-Omni Two-Turn Inference — Deep Dive

The core of the pipeline is `QwenOmni.generate()` (`nemo_curator/models/qwen_omni.py`). It performs **two sequential vLLM inference calls** on the same batch, using the Turn 1 output as context for Turn 2.

##### Turn 1: Clean Transcription

```
┌─────────────────────────────────────────────────────────────────┐
│ For each audio sample in the batch:                             │
│                                                                 │
│ 1. Resample waveform to 16 kHz (via librosa.resample)          │
│ 2. Select prompt:                                               │
│    - If language == "English" and en_prompt_text exists → use it│
│    - Else → use ml_prompt with {language} placeholder resolved  │
│ 3. Build chat messages:                                         │
│    messages = [                                                  │
│      {"role": "system", "content": system_prompt},   (optional) │
│      {"role": "user", "content": [                              │
│        {"type": "text", "text": prompt},                        │
│        {"type": "audio", "audio": waveform_16k_numpy}           │
│      ]}                                                         │
│    ]                                                            │
│ 4. Apply chat template via Qwen3OmniMoeProcessor:              │
│    text = processor.apply_chat_template(messages, ...)          │
│ 5. Extract multimodal data via qwen_omni_utils.process_mm_info │
│    audios, images, videos = process_mm_info(messages)           │
│ 6. Build vLLM input dict:                                       │
│    {"prompt": text, "multi_modal_data": {"audio": audios}}      │
└─────────────────────────────────────────────────────────────────┘
```

All of step 1-6 happens in a **ThreadPoolExecutor** (`prep_workers=16` threads) for parallelism. The preprocessing is CPU-bound (resampling + tokenization), so threading bypasses the GIL via C extensions.

Then: `t1_outputs = self._llm.generate(valid_inputs, sampling_params, use_tqdm=False)`

The vLLM engine processes all samples in a single batched call with continuous batching (up to `max_num_seqs=16` concurrent sequences). Results are stored as `pred_texts[i] = output.outputs[0].text.strip()`.

**If `followup_prompt` is None**: Returns `(pred_texts, [""] * n)` — Turn 2 is skipped entirely.

##### Turn 2: Disfluency Refinement

Only runs when `followup_prompt` is set. Only processes samples where Turn 1 produced non-empty text.

```
┌─────────────────────────────────────────────────────────────────┐
│ For each sample that got a Turn 1 prediction:                   │
│                                                                 │
│ 1. Build multi-turn conversation history:                       │
│    messages = [                                                  │
│      {"role": "system", "content": system_prompt},   (optional) │
│      {"role": "user", "content": [                              │
│        {"type": "text", "text": original_prompt},               │
│        {"type": "audio", "audio": waveform_16k_numpy}  ← SAME  │
│      ]},                                                        │
│      {"role": "assistant", "content": [                         │
│        {"type": "text", "text": turn1_prediction}    ← FROM T1  │
│      ]},                                                        │
│      {"role": "user", "content": [                              │
│        {"type": "text", "text": followup_prompt}                │
│      ]}                                                         │
│    ]                                                            │
│ 2. Apply chat template + extract multimodal data (same as T1)  │
│ 3. Build vLLM input dict                                        │
└─────────────────────────────────────────────────────────────────┘
```

Key observations:
- The **audio is passed again** in the Turn 2 messages (the model needs to "re-listen")
- The **Turn 1 prediction is injected as assistant response** in the conversation history
- The follow-up prompt is a separate user message AFTER the assistant response
- This creates a natural multi-turn dialogue: "transcribe" → (model answers) → "now add disfluencies"

Then: `t2_outputs = self._llm.generate(t2_inputs, sampling_params, use_tqdm=False)`

Results stored as `disfluency_texts[i]`. Samples that failed Turn 2 preprocessing get empty strings.

##### vLLM Engine Configuration

```python
LLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    trust_remote_code=True,            # Required for Qwen architecture
    gpu_memory_utilization=0.95,       # Use 95% of GPU VRAM
    tensor_parallel_size=2,            # Split across 2 GPUs
    limit_mm_per_prompt={"image": 1, "video": 1, "audio": 2},  # 2 audio slots (T1 + T2)
    max_num_seqs=16,                   # Max concurrent sequences
    max_model_len=32768,               # Context window (audio tokens are large)
    seed=1234,                         # Reproducibility
    enable_prefix_caching=True,        # Cache shared prefixes across requests
    prefix_caching_hash_algo="xxhash", # Fast hashing for cache lookups
)

SamplingParams(
    temperature=0.0,    # Greedy decoding (deterministic)
    top_k=1,            # Single token selection
    max_tokens=256,     # Max output length per sample
)
```

**Why `limit_mm_per_prompt={"audio": 2}`**: In Turn 2, the conversation contains the audio from Turn 1 + the same audio re-referenced. vLLM needs to know the maximum multimodal items per prompt for memory allocation.

**Why `enable_prefix_caching=True`**: When processing the same audio in Turn 2, the prefix (system prompt + audio tokens) is identical to Turn 1. Prefix caching avoids recomputing attention for the shared prefix, significantly speeding up Turn 2.

##### Complete Two-Turn Timeline

```
Time →
────────────────────────────────────────────────────────────────────

[Preprocessing Phase — CPU, 16 threads]
  ├── Sample 0: resample → build_messages → apply_template → process_mm_info
  ├── Sample 1: resample → build_messages → apply_template → process_mm_info
  ├── ...
  └── Sample 31: resample → build_messages → apply_template → process_mm_info

[Turn 1 Inference — GPU, vLLM continuous batching]
  └── _llm.generate(32 inputs) → 32 predictions (qwen3_prediction_s1)

[Turn 2 Preprocessing — CPU, 16 threads]
  ├── Sample 0: build_turn2_messages(waveform, prediction_s1) → template → mm_info
  ├── ...
  └── Sample 31: build_turn2_messages(waveform, prediction_s1) → template → mm_info

[Turn 2 Inference — GPU, vLLM continuous batching]
  └── _llm.generate(32 inputs) → 32 predictions (qwen3_prediction_s2)

[Post-processing — write to task.data]
  ├── task.data["qwen3_prediction_s1"] = turn1_text
  ├── task.data["qwen3_prediction_s2"] = turn2_text
  └── task.data.pop("waveform")  (unless keep_waveform=True)
```

##### Error Handling

- If preprocessing fails for a sample (corrupt audio, shape mismatch): that sample gets `""` for both turns, other samples proceed normally
- If ALL samples fail preprocessing: returns `([""] * n, [""] * n)` with a warning
- The `_prep_pool.map()` call returns `None` for failed samples; these are filtered out before `_llm.generate()`
- Turn 2 only runs on samples that produced non-empty Turn 1 output

##### Memory Considerations

- **Waveforms stay in CPU memory** during preprocessing (numpy arrays)
- After Turn 1, the **16 kHz resampled waveforms** are kept in a dict (`waveforms_16k`) for Turn 2 (avoids re-resampling)
- After both turns complete, `InferenceQwenOmniStage.process_batch()` pops the waveform from task data (frees memory)
- `teardown()` explicitly deletes the LLM engine and calls `torch.cuda.empty_cache()` + `gc.collect()`

---

### Stage 4: DisfluencyWerGuardStage (Optional)

**File**: `nemo_curator/stages/audio/text_filtering/disfluency_wer_guard.py`
**Compute**: CPU
**Active when**: `followup_prompt` is set (two-turn mode)

**What it does**:
- Computes WER between `qwen3_prediction_s1` (clean) and `qwen3_prediction_s2` (disfluent)
- If WER > `max_wer_pct` (default 50%), the disfluent version diverged too much → reverts `s2` to `s1`
- Prevents cases where the model hallucinates completely different text in Turn 2

---

### Stage 5: WhisperHallucinationStage (1st instance)

**File**: `nemo_curator/stages/audio/text_filtering/whisper_hallucination.py`
**Compute**: CPU
**Resources**: 1.0 CPU

**What it does** (4 detection methods):

1. **Phrase matching**: Loads hallucination phrases from file (e.g. "Thanks for watching", "Subscribe to my channel"). If the entire transcript matches a known phrase → hallucination.

2. **Unique word ratio**: `unique_words / total_words`. If ratio < `unique_words_threshold` (0.4), text is repetitive nonsense (e.g. "the the the the...").

3. **Long word detection**: Any word longer than `long_word_threshold` (25 chars) that is also > `long_word_rel_threshold` × median word length → hallucination (garbled text).

4. **Character rate**: `len(text) / duration`. If > `max_char_rate` (40 chars/sec), text is impossibly dense for spoken audio (e.g. full paragraph from 0.1s audio).

**Output**: Sets `_skip_me = "Hallucination:WhisperHallucination_omni"` if flagged.

**Phrase file parsing**: Lines ending with integers are treated as phrase + count format — the trailing integer is stripped.

---

### Stage 6: InferenceQwenASRStage (Optional)

**File**: `nemo_curator/stages/audio/inference/qwen_asr.py`
**Model**: `nemo_curator/models/qwen_asr.py` (QwenASR class)
**Compute**: GPU (~2 GB VRAM for 0.6B model)
**Active when**: `asr_model_id` is set

**What it does**:
- Only processes tasks where `_skip_me` starts with `"Hallucination"` (run_only_if_key/prefix)
- Non-hallucinated tasks pass through unchanged
- Runs Qwen3-ASR-0.6B on the original waveform (still in memory if `keep_waveform=True`)
- Writes result to `qwen3_asr_prediction` and detected language to `qwen3_asr_language`

**Why a separate ASR model**: Qwen-Omni (30B) is a generalist that sometimes hallucinates on challenging audio. Qwen-ASR (0.6B) is specialized for ASR and less prone to hallucination. If ASR produces clean output on a sample that Omni hallucinated, we can recover it.

---

### Stage 7: WhisperHallucinationStage (2nd instance, Optional)

Same logic as Stage 5 but configured differently:
- `text_key = "qwen3_asr_prediction"` (checks ASR output)
- `overwrite = True` (can clear the hallucination flag)
- `recovery_value = "Recovered:QwenASR"` (if ASR output is clean, marks as recovered)

If ASR also hallucinates → flag remains set (both models failed).
If ASR is clean → sets notes to "Recovered:QwenASR", clears `_skip_me`.

---

### Stage 8: SelectBestPredictionStage

**File**: `nemo_curator/stages/audio/text_filtering/select_best_prediction.py`
**Compute**: CPU

**Selection priority** (highest to lowest):
1. **Recovered via ASR**: If notes contain "Recovered" and `qwen3_asr_prediction` is non-empty → use ASR prediction
2. **Cross-model agreement**: If BOTH were flagged as hallucinated, but their texts have WER ≤ 20% → use omni (two independent models agreeing = likely correct). Clears `_skip_me`.
3. **Fallback**: Use `qwen3_prediction_s1` (or `s2` if follow-up mode) as-is

**Output**: Writes to `best_prediction` field, which all downstream stages read from.

---

### Stage 9: FastTextLIDStage

**File**: `nemo_curator/stages/audio/text_filtering/fasttext_lid.py`
**Compute**: CPU
**Model**: `facebook/fasttext-language-identification` (130 MB, downloaded from HF Hub)

**What it does**:
1. Runs FastText language identification on `best_prediction` text
2. Compares detected language against `source_lang` from manifest metadata
3. If detection confidence < `min_lang_prob` (0.8) OR detected language ≠ expected → sets `_skip_me = "LID:{reason}"`

**Purpose**: Catches samples where the model produced text in the wrong language (e.g. Omni outputs French for an English audio).

---

### Stage 10: RegexSubstitutionStage

**File**: `nemo_curator/stages/audio/text_filtering/regex_substitution.py`
**Compute**: CPU

**What it does**:
1. Loads regex rules from YAML file (e.g. `tutorials/audio/granary_v2_postprocessing/common.yaml`)
2. Applies each regex pattern → replacement in sequence on `best_prediction`
3. Writes result to `cleaned_text`

**Example rules**:
- Remove multiple spaces
- Normalize unicode characters
- Strip special tokens left by ASR models
- Language-specific cleanup (Arabic diacritics, CJK normalization)

---

### Stage 11: AbbreviationConcatStage

**File**: `nemo_curator/stages/audio/text_filtering/abbreviation_concat.py`
**Compute**: CPU

**What it does**:
- Detects sequences of spaced single letters: `"A P I"` → `"API"`
- Language-aware: uses per-language character classes (Cyrillic, Greek, Latin variants)
- Handles edge cases:
  - Trailing possessives: `"U K's"` → `"UK's"`
  - Single-letter particles (e.g. Italian "e", English "a") are not joined
  - Mixed case pairs (`"xI"`) are not abbreviations
  - DNA/RNA sequences are preserved
- Reads from `cleaned_text`, writes to `abbreviated_text`
- Records found abbreviations in `abbreviations` field

---

### Stage 12: PnCRestorationStage (Optional)

**File**: `nemo_curator/stages/audio/text_filtering/pnc_restoration.py`
**Model**: `nemo_curator/models/qwen_text_llm.py` (QwenTextLLM class)
**Compute**: GPU (TP=1, ~20 GB for Qwen3.5-35B-FP8)
**Active when**: `skip_pnc` is False (default)

**Two-step inference process**:

**Step 1 — Completeness check**:
- Prompt: "Is this text a complete utterance? Reply only 'complete' or 'incomplete': {text}"
- If "incomplete" → text is kept as-is (don't add punctuation to fragments)
- If "complete" → proceed to Step 2

**Step 2 — PnC restoration**:
- Prompt (from `pnc_prompt.md`) includes per-sample language context:
  `Add punctuation and capitalization to the raw ASR transcription in {language} below...`
- Output: Text with restored punctuation and capitalization

**Batching**: Chunks eligible texts into `batch_size` sub-batches, runs vLLM inference per chunk.

**Key parameters**:
- `model_id`: Qwen/Qwen3.5-35B-A3B-FP8 (18 GB, FP8 quantized)
- `kv_cache_dtype`: "fp8" (halves KV-cache memory)
- `max_model_len`: 4096 (sufficient for single-utterance texts)

---

### Stage 13: PnCContentGuardStage (Optional)

**File**: `nemo_curator/stages/audio/text_filtering/pnc_content_guard.py`
**Compute**: CPU
**Active when**: PnC is enabled

**What it does**:
- Compares `abbreviated_text` (input to PnC) with `pnc_text` (PnC output)
- Normalizes both (lowercase, strip punctuation) and checks if underlying words changed
- If words changed → reverts `pnc_text` to `abbreviated_text` (PnC hallucinated/edited content)
- Stores rejected PnC text in `rejected_pnc_text` field for debugging

**Purpose**: LLMs sometimes "improve" text by changing words, not just adding punctuation. This guard catches those cases.

---

### Stage 14: ITNRestorationStage (Optional)

**File**: `nemo_curator/stages/audio/text_filtering/itn_restoration.py`
**Model**: Same `QwenTextLLM` wrapper, separate vLLM engine instance
**Compute**: GPU (TP=1)
**Active when**: `enable_itn` is True

**What it does**:
- Converts spoken-form numbers/dates/symbols to written form:
  - "one hundred twenty three" → "123"
  - "January first twenty twenty six" → "January 1st, 2026"
  - "five dollars" → "$5"
- Uses FP8 KV-cache with prefix caching (shared system prompt cached across requests)

**Validation** (if `enable_validation=True`):
- Checks output isn't dramatically different from input (length ratio, word overlap)
- Falls back to input text on validation failure
- Stores failure reason in `itn_filtered` field

---

### Stage 15: ShardedManifestWriterStage

**File**: `nemo_curator/stages/audio/io/sharded_manifest_writer.py`
**Compute**: CPU (single worker)

**What it does**:
1. Extracts `_shard_key` from task metadata (set by NemoTarShardReaderStage)
2. Constructs output path: `{output_dir}/{shard_key}.jsonl`
3. Appends task data as JSON line to the shard's output file
4. When all tasks for a shard are written (`_shard_total` reached):
   - Creates `{shard_key}.jsonl.done` marker file
   - Done file contains the utterance count

**Important**: Uses `xenna_stage_spec = {"num_workers": 1}` to ensure single-writer access (no concurrent file corruption).

**Output JSONL fields** (per line):
```json
{
  "audio_filepath": "audio_001.opus",
  "duration": 4.2,
  "sample_rate": 16000,
  "corpus": "mmlpc",
  "source_lang": "en",
  "granary_v1_prediction": "original text from v1",
  "qwen3_prediction_s1": "transcribed text turn 1",
  "qwen3_prediction_s2": "transcribed text with disfluencies",
  "best_prediction": "final selected prediction",
  "cleaned_text": "after regex cleanup",
  "abbreviated_text": "after abbreviation concat",
  "pnc_text": "After punctuation restoration.",
  "itn_text": "After ITN: $5 on January 1st.",
  "_skip_me": "",
  "abbreviations": ["API", "UK"]
}
```

---

## 4. Model Wrappers

### 4.1 QwenOmni (`nemo_curator/models/qwen_omni.py`)

**Architecture**: Qwen3-Omni-30B-A3B-Instruct (30B params, FP8, multimodal)
**Backend**: vLLM with `trust_remote_code=True`
**Input**: Audio waveforms (numpy float32, any sample rate)
**Output**: Text predictions

**Key methods**:
- `setup()`: Creates `vllm.LLM` engine, loads `qwen_omni_utils` processor
- `generate(waveforms, sample_rates, languages)`:
  - Creates thread pool for parallel preprocessing
  - Resamples audio to 16 kHz
  - Builds conversation messages with audio tokens
  - Runs vLLM batched generation
  - Optionally runs Turn 2 with follow-up prompt
  - Returns `(pred_texts, disfluency_texts)`
- `teardown()`: Deletes LLM engine, forces garbage collection

### 4.2 QwenASR (`nemo_curator/models/qwen_asr.py`)

**Architecture**: Qwen3-ASR-0.6B (0.6B params, specialized ASR)
**Backend**: `transformers` pipeline (not vLLM — model too small to benefit)
**Input**: Audio waveforms + optional context text + language
**Output**: Transcribed text + detected language

### 4.3 QwenTextLLM (`nemo_curator/models/qwen_text_llm.py`)

**Architecture**: Qwen3.5-35B-A3B-FP8 (text-only, 35B params FP8)
**Backend**: vLLM with FP8 KV-cache
**Used by**: PnCRestorationStage, ITNRestorationStage
**Input**: Text strings + prompt templates
**Output**: Transformed text

**Two-step flow** (PnC):
1. Completeness check: classifies text as complete/incomplete
2. PnC application: adds punctuation to complete texts only

---

## 5. Data Flow & Key Mappings

### Task Data Keys (progressive)

| After Stage | New Keys Added | Keys Read |
|---|---|---|
| NemoTarShardReader | `waveform`, `sample_rate`, `corpus`, `num_channels`, + all manifest fields | - |
| InitializeFields | `_skip_me=""`, `granary_v1_prediction`, `source_lang` | `text` (renamed) |
| QwenOmni | `qwen3_prediction_s1`, `qwen3_prediction_s2` | `waveform`, `sample_rate`, `source_lang` |
| DisfluencyWerGuard | (may revert `s2`) | `qwen3_prediction_s1`, `qwen3_prediction_s2` |
| WhisperHallucination | `_skip_me` (if flagged) | `qwen3_prediction_s1`/`s2`, `duration` |
| QwenASR | `qwen3_asr_prediction`, `qwen3_asr_language` | `waveform`, `sample_rate`, `_skip_me` |
| WhisperHallucination (2nd) | `_skip_me` (may clear) | `qwen3_asr_prediction`, `duration` |
| SelectBestPrediction | `best_prediction`, `omni_asr_agreement_wer` | `qwen3_prediction_s1`/`s2`, `qwen3_asr_prediction`, `_skip_me` |
| FastTextLID | `_skip_me` (if wrong lang) | `best_prediction`, `source_lang` |
| RegexSubstitution | `cleaned_text` | `best_prediction` |
| AbbreviationConcat | `abbreviated_text`, `abbreviations` | `cleaned_text`, `source_lang` |
| PnCRestoration | `pnc_text` | `abbreviated_text`, `_skip_me` |
| PnCContentGuard | `rejected_pnc_text` (if reverted) | `abbreviated_text`, `pnc_text` |
| ITNRestoration | `itn_text`, `itn_filtered` (if failed) | `pnc_text` (or `abbreviated_text`) |
| ShardedManifestWriter | writes to disk | all task data |

### Key Flow Diagram

```
text (from manifest)
  → granary_v1_prediction (renamed by InitializeFields)

waveform + sample_rate (from tar)
  → qwen3_prediction_s1 (from QwenOmni)
  → qwen3_prediction_s2 (from QwenOmni, if follow-up)

qwen3_prediction_s1/s2
  → best_prediction (selected by SelectBestPrediction)

best_prediction
  → cleaned_text (by RegexSubstitution)
  → abbreviated_text (by AbbreviationConcat)
  → pnc_text (by PnCRestoration)
  → itn_text (by ITNRestoration)
```

---

## 6. Execution Engine (Xenna)

The pipeline runs on **Xenna**, NVIDIA's internal distributed execution engine (part of `cosmos-xenna`). Key behaviors:

### Streaming Mode (default)
- Each stage runs as a Ray actor
- Tasks flow through stages one-at-a-time (or in configurable batches)
- GPU stages get dedicated actors with GPU resources
- CPU stages share a pool of CPU actors
- Backpressure: if a downstream GPU stage is full, upstream stages pause

### Batch Mode
- All tasks for a stage are collected before the next stage starts
- Higher memory usage but simpler debugging
- Useful for small datasets

### Resource Allocation
- GPU stages declare `Resources(gpus=N)` where N = tensor_parallel_size
- Xenna assigns GPUs to actors based on resource requests
- On `gpu.l40s.4` (4× L40S 48GB): QwenOmni uses 2 GPUs, PnC uses 1, ITN uses 1

---

## 7. Resume & Checkpointing

The pipeline supports **shard-level resume**:

1. **Discovery**: `NemoTarShardDiscoveryStage` scans `output_dir` for `.done` files
2. **Skip**: Completed shards are not emitted → their tasks never enter the pipeline
3. **Partial cleanup**: If a shard has a `.jsonl` but no `.done`, the partial file is deleted and the shard is reprocessed from scratch
4. **Done markers**: `ShardedManifestWriterStage` writes `.done` only when ALL utterances for a shard have been written

This means a crashed pipeline can be restarted and will automatically skip all fully-processed shards.

---

## 8. Configuration Reference

### GPU Memory Budget (4× L40S 48GB each)

| Model | Stage | VRAM | GPUs | Notes |
|---|---|---|---|---|
| Qwen3-Omni-30B-FP8 | InferenceQwenOmni | ~40 GB | 2× L40S (TP=2) | Primary inference |
| Qwen3.5-35B-FP8 | PnCRestoration | ~20 GB | 1× L40S | FP8 KV-cache |
| Qwen3.5-35B-FP8 | ITNRestoration | ~20 GB | 1× L40S | Shared weights if same model |
| Qwen3-ASR-0.6B | InferenceQwenASR | ~2 GB | 1× L40S | Tiny model, shares GPU |
| FastText LID | FastTextLIDStage | CPU | - | 130 MB model |

### Critical Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Prevent CUDA fork deadlocks |
| `VLLM_LOGGING_LEVEL` | `ERROR` | Suppress verbose vLLM logs |
| `XENNA_RESPECT_CUDA_VISIBLE_DEVICES` | `1` | Correct GPU assignment |
| `RAY_TMPDIR` | `/tmp` | Ray temp files location |
| `HF_TOKEN` | (secret) | HuggingFace Hub authentication |
| `HF_HOME` | `/tmp/hf_home` | Model cache location |
| `AIS_ENDPOINT` | (S3 endpoint URL) | For streaming from S3/AIS |

### Throughput Estimates (per GPU-hour on L40S)

| Stage | Throughput | Bottleneck |
|---|---|---|
| NemoTarShardReader | ~10,000 utterances/min | I/O (tar streaming) |
| QwenOmni (TP=2) | ~50-80 audio-sec/GPU-sec | GPU compute |
| QwenASR | ~200-400 audio-sec/GPU-sec | GPU compute |
| PnC (TP=1) | ~100-200 samples/min | GPU compute |
| ITN (TP=1) | ~150-300 samples/min | GPU compute |
| Text filtering (CPU) | ~10,000+ samples/min | CPU (negligible) |

The pipeline is **GPU-bound** — QwenOmni is always the bottleneck. Text filtering stages process faster than GPU stages can feed them.
