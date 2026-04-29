# Granary v2 Pipeline — Complete Deep Dive

A line-by-line, function-by-function, file-by-file explanation of the entire
Granary v2 Qwen-Omni in-process audio pipeline.

---

## File Map

```
tutorials/audio/qwen_omni_inprocess/
├── main.py                          ← Hydra entry point (~320 lines, includes _prefetch_models)
├── qwen_omni_inprocess.yaml         ← Hydra config (119 lines)
├── PIPELINE_DEEP_DIVE.md            ← This file
└── prompts/
    └── en_qwen3_omni_disfluency_asr.md  ← English disfluency prompt

nemo_curator/stages/audio/
├── io/
│   ├── nemo_tarred_reader.py        ← NeMo-tarred data reader (393 lines)
│   └── sharded_manifest_writer.py   ← Per-shard JSONL output writer (119 lines)
├── inference/
│   ├── qwen_omni.py                 ← Qwen3-Omni GPU inference stage (223 lines)
│   └── qwen_asr.py                  ← Qwen3-ASR GPU inference stage (optional)
└── text_filtering/
    ├── initialize_fields.py         ← Field initializer (86 lines)
    ├── whisper_hallucination.py      ← Hallucination detector (184 lines)
    ├── select_best_prediction.py     ← Best prediction selector (95 lines)
    ├── fasttext_lid.py              ← Language ID via FastText (191 lines)
    ├── regex_substitution.py         ← Regex-based text cleanup (103 lines)
    ├── abbreviation_concat.py        ← Abbreviation joining (261 lines)
    ├── pnc_restoration.py            ← Punctuation/capitalization via LLM (232 lines)
    └── pnc_content_guard.py          ← PnC output validation (92 lines)

nemo_curator/models/
├── qwen_omni.py                     ← QwenOmni vLLM wrapper
└── qwen_text_llm.py                 ← QwenTextLLM vLLM wrapper (used by PnC)
```

---

## 1. Entry Point: `main.py`

**File:** `tutorials/audio/qwen_omni_inprocess/main.py`

### Lines 38-39: Environment setup (before any imports)

```python
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
```

These MUST be set before vLLM is imported anywhere. `spawn` is required for
multi-GPU tensor parallelism (fork + CUDA = deadlock). `setdefault` means
explicit env vars (e.g. from NvLLMOps) take precedence.

### Lines 41-63: Imports

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader
...
```

All stage classes are imported at module level. This means the module must be
importable without GPU — stages don't touch GPU until `setup()`.

### Lines 66-73: `_read_file_or_str(value)`

```python
def _read_file_or_str(value: str | None) -> str | None:
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read().strip()
    return value
```

Utility that transparently handles prompt values that are either:
- A file path (e.g. `/src/Curator/.../en_qwen3_omni_disfluency_asr.md`) → reads contents
- A raw string (e.g. `"Transcribe the audio."`) → returns as-is
- `None` → returns `None`

Used for all prompt resolution (lines 86-91).

### Lines 76-235: `build_granary_v2_pipeline(cfg)`

This is the core function that constructs the stage chain. It receives the
fully-resolved Hydra `DictConfig` and returns a `Pipeline` object.

#### Lines 78-80: Data config resolution

```python
data_config = cfg.get("data_config") or cfg.get("input_manifest")
if not data_config:
    raise ValueError("Either 'data_config' or 'input_manifest' must be set")
```

NvLLMOps passes `input_manifest=<path>` as a Hydra override. The YAML has
`data_config: ${input_manifest}`, so both resolve to the same path. The
fallback chain ensures either works.

#### Lines 82-84: Output directory and corpus filter

```python
output_dir = cfg.get("output_dir") or cfg.get("workspace_dir", "./output")
corpus_filter = OmegaConf.to_container(cfg.corpus, resolve=True) if "corpus" in cfg and cfg.corpus else None
```

`corpus_filter` limits which corpora from the YAML data config are processed.
`None` means process all. `OmegaConf.to_container` converts the Hydra ListConfig
to a plain Python list.

#### Lines 86-91: Prompt resolution

```python
ml_prompt = _read_file_or_str(cfg.get("ml_prompt_file")) or cfg.get("ml_prompt", "Transcribe the audio.")
en_prompt = _read_file_or_str(cfg.get("en_prompt_file"))
followup_prompt = _read_file_or_str(cfg.get("followup_prompt_file")) or cfg.get("followup_prompt")
system_prompt = _read_file_or_str(cfg.get("system_prompt"))
pnc_prompt = _read_file_or_str(cfg.get("pnc_prompt_file")) or cfg.get("pnc_prompt")
itn_prompt = _read_file_or_str(cfg.get("itn_prompt_file"))
```

Each prompt has a `_file` variant (path to a `.md` file) and a direct string
variant. File takes priority. This is how prompts baked into the Docker image
(as `.md` files) get loaded.

#### Lines 93-95: Key configuration

```python
omni_text_key = "qwen3_prediction_s2" if followup_prompt else "qwen3_prediction_s1"
source_lang_key = cfg.get("source_lang_key", "source_lang")
asr_model_id = cfg.get("asr_model_id")
```

- `omni_text_key`: Which QwenOmni output to use downstream. If two-turn mode
  (followup prompt), use the disfluency-aware second turn. Otherwise, first turn.
- `asr_model_id`: If set, enables QwenASR recovery stages.

#### Lines 97-125: Stage chain — Reader + QwenOmni

```python
stages = [
    NemoTarredAudioReader(
        yaml_path=data_config,
        corpus_filter=corpus_filter,
        s3_endpoint_url=cfg.get("s3_endpoint_url"),
        output_dir=output_dir,
    ).with_({"nemo_tar_shard_reader": {"resources": Resources(cpus=4.0)}}),

    InitializeFieldsStage(),

    InferenceQwenOmniStage(
        model_id=cfg.get("model_id", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        prompt_text=ml_prompt,
        en_prompt_text=en_prompt,
        ...
        tensor_parallel_size=cfg.get("tensor_parallel_size", 2),
        batch_size=cfg.get("batch_size", 32),
        ...
        keep_waveform=bool(asr_model_id),
    ),
]
```

`.with_()` overrides sub-stage resources. Here it gives the shard reader 4 CPU
cores for parallel tar streaming. `keep_waveform=bool(asr_model_id)` keeps
audio data in memory only if ASR recovery is enabled (saves memory otherwise).

#### Lines 127-132: Optional DisfluencyWerGuard

```python
if followup_prompt:
    stages.append(DisfluencyWerGuardStage(
        ref_text_key="qwen3_prediction_s1",
        hyp_text_key="qwen3_prediction_s2",
        max_wer_pct=cfg.get("max_wer_pct", 50.0),
    ))
```

Only active in two-turn mode. Compares turn 1 (clean ASR) with turn 2
(disfluency-aware). If WER > 50%, the disfluency turn is likely corrupt.

#### Lines 134-142: WhisperHallucinationStage

```python
stages.append(WhisperHallucinationStage(
    name="WhisperHallucination_omni",
    common_hall_file=cfg.hall_phrases,
    text_key=omni_text_key,
    unique_words_threshold=cfg.get("unique_words_threshold", 0.4),
    long_word_threshold=cfg.get("long_word_threshold", 25),
    long_word_rel_threshold=cfg.get("long_word_rel_threshold", 3.0),
    max_char_rate=cfg.get("max_char_rate", 40.0),
))
```

Always present. Checks QwenOmni output for hallucination patterns.

#### Lines 144-164: Optional QwenASR recovery

```python
if asr_model_id:
    stages.extend([
        InferenceQwenASRStage(model_id=asr_model_id, ...),
        WhisperHallucinationStage(
            name="WhisperHallucination_asr",
            text_key="qwen3_asr_prediction",
            overwrite=True,
            recovery_value="Recovered:QwenASR",
            ...
        ),
    ])
```

Only if `asr_model_id` is set. Runs a second ASR model (QwenASR) on the
same audio. If the omni model hallucinated but ASR didn't, the ASR output
can recover the sample. The second `WhisperHallucinationStage` has
`overwrite=True` — it can clear a previously set hallucination flag.

#### Lines 166-169: SelectBestPrediction

```python
stages.append(SelectBestPredictionStage(
    primary_text_key=omni_text_key,
    asr_text_key="qwen3_asr_prediction",
))
```

Always present. Picks the best available prediction and writes to
`best_prediction`. Handles three cases: ASR recovery, cross-model agreement,
or fallback to primary.

#### Lines 171-188: Text cleanup chain

```python
stages.extend([
    FastTextLIDStage(
        model_path=cfg.get("fasttext_model", "facebook/fasttext-language-identification"),
        text_key="best_prediction",
        source_lang_key=source_lang_key,
        min_lang_prob=cfg.get("min_lang_prob", 0.8),
    ),
    RegexSubstitutionStage(
        regex_params_yaml=cfg.regex_yaml,
        text_key="best_prediction",
        output_text_key="cleaned_text",
    ),
    AbbreviationConcatStage(
        text_key="cleaned_text",
        output_text_key="abbreviated_text",
        source_lang_key=source_lang_key,
    ),
])
```

Three sequential CPU stages that clean up the text:
1. FastText LID — flags wrong-language entries
2. Regex — applies pattern-based cleanup rules from `common.yaml`
3. AbbreviationConcat — joins "A P I" → "API"

#### Lines 190-214: Optional PnC restoration

```python
if not cfg.get("skip_pnc", False):
    stages.extend([
        PnCRestorationStage(
            model_id=cfg.get("pnc_model_id", "Qwen/Qwen3.5-35B-A3B-FP8"),
            text_key="abbreviated_text",
            output_text_key="pnc_text",
            tensor_parallel_size=cfg.get("pnc_tensor_parallel_size", 2),
            ...
        ),
        PnCContentGuardStage(
            text_key="abbreviated_text",
            pnc_text_key="pnc_text",
            rejected_text_key="rejected_pnc_text",
        ),
    ])
```

GPU-based PnC restoration followed by content guard. PnC uses a separate
text-only LLM (Qwen3.5-35B-FP8) to add punctuation and capitalization.
ContentGuard reverts if the LLM changed actual word content.

#### Lines 216-231: Optional ITN

```python
if cfg.get("enable_itn", False):
    stages.append(ITNRestorationStage(...))
```

Inverse Text Normalization — converts spoken-form numbers/dates to
written form. Disabled by default.

#### Line 233: Output writer

```python
stages.append(ShardedManifestWriterStage(output_dir=output_dir))
```

Always last. Writes processed tasks to per-shard JSONL files.

### Lines 239-291: `_prefetch_models(cfg)` — Parallel model pre-download

```python
def _prefetch_models(cfg: DictConfig) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import snapshot_download

    tasks = {}
    # Qwen3-Omni model (15GB, 6 safetensor shards)
    omni_model = cfg.get("model_id", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    tasks["qwen_omni"] = lambda m=omni_model: snapshot_download(m)

    # FastText LID model (130MB)
    tasks["fasttext"] = lambda: snapshot_download("facebook/fasttext-language-identification")

    # PnC text LLM (18GB, FP8)
    if cfg.get("pnc_model_id"):
        pnc_model = cfg.pnc_model_id
        tasks["pnc"] = lambda m=pnc_model: snapshot_download(m)

    # ASR recovery model (optional)
    if cfg.get("asr_model_id"):
        asr_model = cfg.asr_model_id
        tasks["asr"] = lambda m=asr_model: snapshot_download(m)

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            future.result()  # raises on failure
            logger.info(f"Pre-fetched: {name}")
```

Downloads ALL models concurrently before `pipeline.run()`. Each `snapshot_download()` uses ~8 HF hub threads internally, giving ~24 concurrent HTTP connections total on `gpu.l40s.4` (218 CPUs, 980Gi RAM).

**Performance impact:**
- Wall-clock download = `max(15GB, 18GB)` ≈ 12 min instead of sequential 15+18 ≈ 22 min
- **Saves ~10-13 min per first-run job**
- Zero impact on subsequent runs (all cached in `/tmp/hf_home/hub/`)

**Lambda closure:** Note `lambda m=omni_model: ...` — the default argument captures the current value at definition time, avoiding a common Python closure bug where all lambdas would reference the same variable.

### Lines 293-313: `main(cfg)` — Hydra entry point

```python
@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    hf_token = cfg.get("hf_token", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HF_HOME", "/tmp/hf_home")

    _prefetch_models(cfg)                # ← NEW: parallel model pre-download

    pipeline = build_granary_v2_pipeline(cfg)
    executor = XennaExecutor(config={"execution_mode": cfg.get("execution_mode", "streaming")})

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min.")
```

1. Sets `HF_TOKEN` from Hydra config (passed by NvLLMOps from Vault, or CLI)
2. Sets `HF_HOME` to `/tmp/hf_home` (writable ephemeral disk inside containers)
3. **Pre-fetches all models in parallel** — ensures `setup_on_node()` in each stage gets instant cache hits
4. Builds the pipeline
5. Creates `XennaExecutor` in streaming mode (stages run concurrently)
6. Runs and times the pipeline

---

## 2. Data Reader: `nemo_tarred_reader.py`

**File:** `nemo_curator/stages/audio/io/nemo_tarred_reader.py`

### Lines 44-52: `_expand_nemo_path(pattern)`

```python
def _expand_nemo_path(pattern: str) -> list[str]:
    match = re.search(r"_OP_(\d+)\.\.(\d+)_CL_", pattern)
    if not match:
        return [pattern]
    start, end = int(match.group(1)), int(match.group(2))
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
```

Expands NeMo's brace-expansion syntax:
- Input: `s3://playground/.../manifest__OP_0..1_CL_.json`
- Output: `["s3://playground/.../manifest_0.json", "s3://playground/.../manifest_1.json"]`

The `__OP_` and `_CL_` markers replace `{` and `}` because YAML doesn't
allow literal braces in certain contexts.

### Lines 55-69: `_s3_to_pipe(tar_path, s3_endpoint_url)`

```python
def _s3_to_pipe(tar_path: str, s3_endpoint_url: str | None = None) -> str:
    parsed = urlparse(tar_path)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    endpoint = s3_endpoint_url or os.environ.get("AIS_ENDPOINT")
    url = f"{endpoint.rstrip('/')}/v1/objects/{bucket}/{key}?provider=s3"
    return f"pipe:curl -sL '{url}'"
```

Converts an S3 URL to a `pipe:` command that lhotse can open as a streaming
file. The AIS (AIStore) gateway URL format
`{endpoint}/v1/objects/{bucket}/{key}?provider=s3` tells the S3-compatible
endpoint to fetch from its S3 backend.

For our setup: `s3://playground/aaftabv/.../audio_0.tar` with
`s3_endpoint_url=https://pdx.s8k.io` becomes:
```
pipe:curl -sL 'https://pdx.s8k.io/v1/objects/playground/aaftabv/.../audio_0.tar?provider=s3'
```

### Lines 72-88: `_open_tar(tar_path, s3_endpoint_url)`

```python
def _open_tar(tar_path: str, s3_endpoint_url: str | None = None) -> tarfile.TarFile:
    if tar_path.startswith("s3://"):
        pipe_path = _s3_to_pipe(tar_path, s3_endpoint_url)
        fileobj = open_best(pipe_path, mode="rb")
    else:
        fileobj = open_best(tar_path, mode="rb")
    return tarfile.open(fileobj=fileobj, mode="r|*")
```

Opens a tar file in streaming mode (`r|*`) — sequential read without seeking.
For S3 paths, it pipes through `curl`. For local paths, it opens directly.
`lhotse.open_best` handles the `pipe:` protocol transparently.

### Lines 96-231: `NemoTarShardDiscoveryStage`

This stage runs once at the start. It parses the YAML data config and emits
one `FileGroupTask` per shard pair (manifest + tar).

#### Lines 180-231: `process(_task) -> list[FileGroupTask]`

```python
def process(self, _task: _EmptyTask) -> list[FileGroupTask]:
    with open(self.yaml_path) as f:
        config = yaml.safe_load(f)

    tasks = []
    for group in config:
        for cfg in group.get("input_cfg", []):
            corpus = cfg.get("corpus", "unknown")
            manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
            tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
            for mp, tp in zip(manifest_paths, tar_paths):
                tasks.append(FileGroupTask(
                    task_id=shard_key,
                    data=[mp, tp],
                    reader_config={"corpus": corpus, "shard_key": shard_key},
                ))
    return tasks
```

For our 2-shard sample, this emits 2 `FileGroupTask` objects:
- `FileGroupTask(data=["s3://.../manifest_0.json", "s3://.../audio_0.tar"])`
- `FileGroupTask(data=["s3://.../manifest_1.json", "s3://.../audio_1.tar"])`

It also handles:
- **Checkpointing** (lines 126-142, 183-186): Scans `output_dir` for `.done`
  markers and skips already-completed shards. Enables crash recovery.
- **Partial output cleanup** (lines 214-218): Removes partially-written JSONL
  files for incomplete shards to avoid duplicates on restart.

### Lines 238-335: `NemoTarShardReaderStage`

This is the fanout stage — it receives one `FileGroupTask` (a shard) and
emits many `AudioTask` objects (one per utterance).

#### Lines 273-281: `_read_manifest(path)`

```python
def _read_manifest(self, path: str) -> dict[str, dict]:
    entries = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            entry = json.loads(raw_line.strip())
            entries[entry[self.filepath_key]] = entry
    return entries
```

Reads a NeMo manifest JSON file. Each line is a JSON object with fields like
`audio_filepath`, `duration`, `text`, `source_lang`. The `audio_filepath`
is the filename INSIDE the tar archive (e.g. `audio_0_8600.wav`). Returns
a dict keyed by filename for O(1) lookup during tar iteration.

#### Lines 283-335: `process(task) -> list[AudioTask]`

```python
def process(self, task: FileGroupTask) -> list[AudioTask]:
    manifest_path, tar_path = task.data[0], task.data[1]
    manifest = self._read_manifest(manifest_path)

    tar = _open_tar(tar_path, self.s3_endpoint_url)
    results = []
    for tar_info in tar:
        if not tar_info.isfile() or tar_info.name not in manifest:
            continue
        raw_audio = tar.extractfile(tar_info).read()       # read bytes into memory
        audio, sample_rate = sf.read(BytesIO(raw_audio))   # decode to numpy float32
        if audio.ndim > 1:
            audio = audio.mean(axis=1)                      # stereo → mono
        entry = dict(manifest[tar_info.name])
        entry["waveform"] = audio
        entry["sample_rate"] = sample_rate
        results.append(AudioTask(task_id=..., data=entry))
    return results
```

For each tar member:
1. Check if it's a file and has a manifest entry (skip directories, metadata)
2. Read raw bytes into memory (no disk write)
3. Decode audio via `soundfile` to numpy float32 array
4. Convert stereo to mono by averaging channels
5. Merge manifest metadata with audio data into an `AudioTask`

The `_shard_key` and `_shard_total` metadata (lines 320-332) enable the
`ShardedManifestWriterStage` to write per-shard output files and create
`.done` markers when all utterances from a shard are processed.

### Lines 342-393: `NemoTarredAudioReader` (composite)

```python
class NemoTarredAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    def __post_init__(self):
        self._stages = [
            NemoTarShardDiscoveryStage(yaml_path=..., corpus_filter=..., output_dir=...),
            NemoTarShardReaderStage(filepath_key=..., s3_endpoint_url=...),
        ]

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
```

User-facing API. A `CompositeStage` that decomposes into discovery + reader.
Xenna calls `decompose()` to get the sub-stages and schedules them independently.

---

## 3. GPU Inference: `qwen_omni.py`

**File:** `nemo_curator/stages/audio/inference/qwen_omni.py`

### Lines 30-44: Language code mapping

```python
_LANG_CODE_TO_NAME = {"en": "English", "de": "German", ...}
```

Maps ISO 639-1 codes to full language names. Used to tell the Qwen model
which language to expect (some prompts are language-conditioned).

### Lines 47-112: `InferenceQwenOmniStage` dataclass

Key fields:
- `model_id` (line 91): HuggingFace model ID, default `"Qwen/Qwen3-Omni-30B-A3B-Instruct"`
- `tensor_parallel_size` (line 104): Number of GPUs. Set at line 117 to `Resources(gpus=float(tp))`
- `batch_size` (line 111): Batches for `process_batch`, default 32
- `keep_waveform` (line 109): Whether to keep audio data after inference. Only needed if QwenASR runs later.

### Lines 113-117: `__post_init__()`

```python
def __post_init__(self):
    self._model = None
    tp = self.tensor_parallel_size
    if tp and tp > 0:
        self.resources = Resources(gpus=float(tp))
```

Sets GPU resource requirement based on `tensor_parallel_size`. With TP=2,
the stage declares it needs 2 GPUs. Xenna uses this for scheduling.

### Lines 119-134: `_create_model()`

```python
def _create_model(self) -> QwenOmni:
    return QwenOmni(
        model_id=self.model_id,
        prompt_text=self.prompt_text,
        ...
    )
```

Creates the `QwenOmni` model wrapper (defined in `nemo_curator/models/qwen_omni.py`).
Does NOT load GPU memory — that happens in `setup()`.

### Lines 140-157: `setup_on_node()`

```python
def setup_on_node(self, ...):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(self.model_id)
    except Exception:
        logger.warning("snapshot_download failed; setup() will download")
```

Called **once per physical node** by Xenna, before any GPU allocation.
Downloads model weights (~15GB) to the HuggingFace cache. This prevents
multiple GPU workers on the same node from triggering parallel downloads.

### Lines 159-163: `setup()`

```python
def setup(self, ...):
    if self._model is None:
        self._model = self._create_model()
        self._model.setup()
```

Called **per worker replica**. Creates the vLLM engine and loads the model
into GPU memory. If `setup_on_node()` already cached the weights, this
just loads from local cache.

### Lines 165-168: `teardown()`

```python
def teardown(self):
    if self._model is not None:
        self._model.teardown()
        self._model = None
```

Frees GPU memory. In streaming mode, Xenna calls this when the stage
finishes processing all data, freeing GPUs for later stages (e.g. PnC).

### Lines 191-222: `process_batch(tasks)`

```python
def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
    waveforms = [t.data[self.waveform_key] for t in tasks]
    sample_rates = [t.data[self.sample_rate_key] for t in tasks]
    languages = [
        _LANG_CODE_TO_NAME.get(code, code)
        for code in (t.data.get(self.source_lang_key) for t in tasks)
    ]

    pred_texts, disfluency_texts = self._model.generate(waveforms, sample_rates, languages)

    for task, pred, disfl in zip(tasks, pred_texts, disfluency_texts):
        task.data[self.pred_text_key] = pred
        if self.followup_prompt:
            task.data[self.disfluency_text_key] = disfl
        if not self.keep_waveform:
            task.data.pop(self.waveform_key, None)
    return tasks
```

1. Extracts waveforms, sample rates, and language codes from all tasks
2. Calls `self._model.generate()` which runs batched vLLM inference
3. Writes predictions back to task data
4. Optionally drops waveform data to free memory (line 218-219)

---

## 4. Text Filtering Stages

### 4a. `initialize_fields.py` — InitializeFieldsStage

**File:** `nemo_curator/stages/audio/text_filtering/initialize_fields.py`

#### Lines 66-74: `process(task)`

```python
def process(self, task):
    task.data["_skip_me"] = ""
    if "source_lang" not in task.data:
        task.data["source_lang"] = "en"
    if "text" in task.data:
        task.data["granary_v1_prediction"] = task.data.pop("text")
    for key in self.drop_keys:
        task.data.pop(key, None)
    return task
```

Three operations:
1. Initialize `_skip_me` to empty string (not skipped)
2. Set default `source_lang` if missing
3. Rename `text` → `granary_v1_prediction` (preserve any pre-existing text)
4. Drop irrelevant keys (`answer`, `target_lang`, `diarize`, etc.)

### 4b. `whisper_hallucination.py` — WhisperHallucinationStage

**File:** `nemo_curator/stages/audio/text_filtering/whisper_hallucination.py`

#### Lines 65-79: `setup()`

```python
def setup(self, ...):
    phrases = set()
    with open(self.common_hall_file) as f:
        for line in f:
            tokens = line.strip().rsplit(maxsplit=1)
            if len(tokens) == 2 and tokens[1].lstrip("-").isdigit():
                phrases.add(tokens[0])      # "Thank you 1297" → "Thank you"
            else:
                phrases.add(line.strip())
    self._phrases = phrases
```

Loads hallucination phrases from `en.txt`. Each line may have a trailing
frequency count (e.g. `"Thank you 1297"`). The count is stripped, keeping
only the phrase text.

#### Lines 87-100: Hallucination checks

Four independent checks, any one triggers flagging:

1. **`_repeated_ngrams`** (lines 87-90): `len(set(words)) / len(words) <= 0.4`
   — low unique-word ratio means repetitive text
2. **`_long_word`** (lines 92-100): Single word >= 25 chars, OR longest word
   is 3x longer than second-longest — garbled model output
3. **`_frequent_single_word`** (lines 105-112): Full text matches a known
   hallucination phrase (exact or prefix match for phrases >= 8 chars)
4. **`_high_char_rate`** (lines 114-118): `total_chars / duration > 40`
   — impossible speech rate (e.g. 200 characters in 0.1 seconds)

#### Lines 120-160: `_process_single(task)`

```python
def _process_single(self, task):
    current_flag = task.data.get("_skip_me", "")
    if not self.overwrite and current_flag:
        return task                          # already flagged, skip

    text = task.data[self.text_key]
    words = text.split()
    duration = task.data.get("duration", 0.0)

    is_hallucinated = repeated or long_w or phrase or high_rate
    if is_hallucinated:
        task.data["_skip_me"] = f"Hallucination:{self.name}"
    elif self.overwrite and was_flagged:
        task.data["_skip_me"] = ""           # recovery: clear flag
        task.data["additional_notes"] = "Recovered:QwenASR"
    return task
```

The `overwrite` flag is key for the ASR recovery path: the second
hallucination check (after QwenASR) can CLEAR a previous hallucination
flag if the ASR output is clean.

### 4c. `select_best_prediction.py` — SelectBestPredictionStage

**File:** `nemo_curator/stages/audio/text_filtering/select_best_prediction.py`

#### Lines 69-94: `process(task)`

Three-priority selection:

```python
def process(self, task):
    primary_pred = task.data.get("qwen3_prediction_s1", "")
    asr_pred = task.data.get("qwen3_asr_prediction", "")
    notes = task.data.get("additional_notes", "")
    skip_me = task.data.get("_skip_me", "")

    # Priority 1: ASR recovery
    if "Recovered" in notes and asr_pred:
        task.data["best_prediction"] = asr_pred
        return task

    # Priority 2: Cross-model agreement
    if skip_me.startswith("Hallucination") and asr_pred and primary_pred:
        wer = get_wer(normalize(primary_pred), normalize(asr_pred))
        if wer <= 20.0:  # (100 - min_agreement_pct=80)
            task.data["best_prediction"] = primary_pred
            task.data["_skip_me"] = ""                    # unflag!
            task.data["additional_notes"] = "Recovered:CrossModelAgreement"
            return task

    # Priority 3: Fallback
    task.data["best_prediction"] = primary_pred
    return task
```

Cross-model agreement: If BOTH models were flagged as hallucinated, but
their outputs are nearly identical (WER <= 20%), the text is likely correct
because two independent models independently produced the same output.

### 4d. `fasttext_lid.py` — FastTextLIDStage

**File:** `nemo_curator/stages/audio/text_filtering/fasttext_lid.py`

#### Lines 95-119: `_resolve_model_path()`

Three-level model path resolution:
1. **Local file** (line 96-97): If `model_path` is an existing file, use it
2. **Legacy name** (lines 98-106): `lid.176.bin` or `lid.176.ftz` — downloads
   from Facebook's CDN to `~/.cache/nemo_curator/fasttext/`
3. **HuggingFace repo** (lines 107-114): `facebook/fasttext-language-identification`
   — downloads via `huggingface_hub.hf_hub_download(filename="model.bin")`

#### Lines 135-140: `setup()`

```python
def setup(self, ...):
    import fasttext
    resolved = self._resolve_model_path()
    self._model = fasttext.load_model(resolved)
```

Loads the FastText model into memory (~130MB). The `import fasttext` is
deferred to avoid import errors when fasttext isn't installed.

#### Lines 152-172: `_process_single(task)`

```python
def _process_single(self, task):
    if task.data.get("_skip_me", ""):
        return task                              # already flagged
    text = task.data["best_prediction"].strip().replace("\n", " ")
    if not text:
        task.data["_skip_me"] = "Empty text"
        return task
    if len(text.split()) < self.min_word_count:  # default 2
        return task                              # too short for reliable LID

    lang, prob = self._predict(text)
    expected = task.data["source_lang"]
    if lang != expected.lower():
        task.data["_skip_me"] = "Wrong language"
    elif prob < self.min_lang_prob:               # default 0.8
        task.data["_skip_me"] = "Low probability of language"
    return task
```

Short texts (< 2 words) are passed through because FastText is unreliable
on single words.

### 4e. `regex_substitution.py` — RegexSubstitutionStage

**File:** `nemo_curator/stages/audio/text_filtering/regex_substitution.py`

#### Lines 57-62: `setup()`

```python
def setup(self, ...):
    with open(self.regex_params_yaml) as f:
        raw = yaml.safe_load(f)
    self._rules = raw if isinstance(raw, list) else []
```

Loads a YAML file containing a list of `{pattern, repl, count}` dicts.

#### Lines 70-84: `_process_single(task)`

```python
def _process_single(self, task):
    if task.data.get("_skip_me", ""):
        task.data.setdefault("cleaned_text", "")
        return task
    text = " " + task.data["best_prediction"] + " "  # pad for edge patterns
    for rule in self._rules:
        text = re.sub(rule["pattern"], rule["repl"], text, count=rule.get("count", 0))
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    task.data["cleaned_text"] = text
    if not text and not task.data["_skip_me"]:
        task.data["_skip_me"] = "Empty after regex cleaning"
    return task
```

Applies regex rules sequentially. Padding with spaces ensures patterns
that match at word boundaries work correctly at the start/end of text.

### 4f. `abbreviation_concat.py` — AbbreviationConcatStage

**File:** `nemo_curator/stages/audio/text_filtering/abbreviation_concat.py`

#### Lines 166-201: `concat_abbreviations(text, language)`

Core algorithm for joining spaced-out single letters:

```python
def concat_abbreviations(text, language="en"):
    pattern = _get_pattern(language)        # language-specific regex
    particles = _LANG_PARTICLES.get(language, frozenset())

    def _collect_and_join(match):
        raw = match.group(0)                # e.g. "A P I"
        replaced = _join_letters(match, particles)  # e.g. "API"
        if replaced != raw:
            found.append(replaced.strip())
        return replaced

    result = pattern.sub(_collect_and_join, text)
    return result, found
```

Language-aware: uses `_LANG_CHAR_CLASS` (lines 30-59) for language-specific
character sets. For example, German includes ÄÖÜ, Russian uses Cyrillic.

Edge cases handled:
- `_strip_particles` (lines 91-104): Removes articles like "a" from matches
  (e.g. "a B C" → "BC" not "aBC")
- `_is_double_i` (line 113): "I I" is not an abbreviation
- `_is_mixed_case_pair` (line 108): "xI" or "Ia" → not abbreviations
- DNA/RNA guard (lines 77, 99-100): "D N A" keeps trailing letters

### 4g. `pnc_restoration.py` — PnCRestorationStage

**File:** `nemo_curator/stages/audio/text_filtering/pnc_restoration.py`

#### Lines 109-114: `_resolve_pnc_prompt()`

```python
def _resolve_pnc_prompt(self):
    if self.pnc_prompt:
        return self.pnc_prompt
    path = Path(self.pnc_prompt_file) if self.pnc_prompt_file else _DEFAULT_PNC_PROMPT_PATH
    return path.read_text(encoding="utf-8").strip()
```

Falls back to `nemo_curator/stages/audio/text_filtering/prompts/pnc_prompt.md`.

#### Lines 143-160: Lifecycle

```python
def setup_on_node(self, ...):
    self._model = self._create_model()
    self._model.setup()                 # loads vLLM engine on GPU

def setup(self, ...):
    if self._model is None:
        self._model = self._create_model()
        self._model.setup()

def teardown(self):
    self._model.teardown()              # frees GPU memory
    self._model = None
```

Unlike QwenOmni, PnC's `setup_on_node()` directly creates and loads the model
(not just downloading weights). This is because `QwenTextLLM` may not have
a separate download-only path.

#### Lines 180-194: `_partition_tasks(tasks)`

```python
def _partition_tasks(self, tasks):
    eligible_indices = []
    eligible_texts = []
    for i, task in enumerate(tasks):
        skip = task.data.get("_skip_me", "")
        text = task.data.get(self.text_key, "")
        if skip:
            task.data[self.output_text_key] = ""
        elif not text.strip():
            task.data[self.output_text_key] = text
        else:
            eligible_indices.append(i)
            eligible_texts.append(text)
    return eligible_indices, eligible_texts
```

Separates tasks that need PnC (eligible) from those that are skipped or
empty. Skipped/empty tasks get their output set immediately without hitting
the GPU.

#### Lines 196-231: `process_batch(tasks)`

```python
def process_batch(self, tasks):
    eligible_indices, eligible_texts = self._partition_tasks(tasks)
    if not eligible_indices:
        return tasks

    all_complete = []
    all_pnc = []
    for start in range(0, len(eligible_texts), self.batch_size):
        chunk = eligible_texts[start:start + self.batch_size]
        is_complete, pnc_texts = self._model.generate(chunk)
        all_complete.extend(is_complete)
        all_pnc.extend(pnc_texts)

    for idx, _complete, pnc_text in zip(eligible_indices, all_complete, all_pnc):
        tasks[idx].data[self.output_text_key] = pnc_text
    return tasks
```

Two-step LLM process per text (inside `self._model.generate()`):
1. **Completeness check**: "Is this text a complete sentence? Answer yes/no."
2. **PnC restoration**: If yes, apply punctuation and capitalization.

### 4h. `pnc_content_guard.py` — PnCContentGuardStage

**File:** `nemo_curator/stages/audio/text_filtering/pnc_content_guard.py`

#### Lines 36-42: Normalization

```python
def _normalise(t):
    return t.lower().translate(_PUNCT_TABLE).replace(" ", "")

def _words_match(a, b):
    return _normalise(a) == _normalise(b)
```

Strips punctuation, lowercases, and removes spaces. If two strings match
after this normalization, they have the same word content — the LLM only
changed punctuation/capitalization (correct behavior).

#### Lines 71-85: `_process_single(task)`

```python
def _process_single(self, task):
    original = task.data.get("abbreviated_text", "")
    pnc = task.data.get("pnc_text", "")

    if original and pnc and not _words_match(original, pnc):
        task.data["rejected_pnc_text"] = pnc       # save bad output
        task.data["pnc_text"] = original            # revert to original
    else:
        task.data["rejected_pnc_text"] = ""
    return task
```

If the LLM changed actual word content (added, removed, or substituted
words), the PnC output is rejected and the original text is used instead.
The rejected output is saved for debugging.

---

## 5. Output Writer: `sharded_manifest_writer.py`

**File:** `nemo_curator/stages/audio/io/sharded_manifest_writer.py`

### Lines 63-69: `setup_on_node()`

```python
def setup_on_node(self, ...):
    os.makedirs(self.output_dir, exist_ok=True)
```

Creates the output directory tree once per node.

### Lines 71-95: `process(task)`

```python
def process(self, task):
    shard_key = task._metadata.get("_shard_key", "unknown/shard_0")
    out_path = os.path.join(self.output_dir, f"{shard_key}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "a") as f:
        f.write(json.dumps(task.data) + "\n")

    self._shard_counts[shard_key] += 1
    shard_total = task._metadata.get("_shard_total", 0)
    if shard_total > 0 and self._shard_counts[shard_key] >= shard_total:
        done_path = f"{out_path}.done"
        with open(done_path, "w") as f:
            f.write(f"{self._shard_counts[shard_key]}\n")
```

Writes each processed utterance as a JSON line to a shard-specific output
file. When all utterances from a shard have been written, creates a `.done`
marker file. This enables the `NemoTarShardDiscoveryStage` to skip
completed shards on restart (crash recovery).

### Lines 114-118: Concurrency control

```python
def num_workers(self):
    return 1

def xenna_stage_spec(self):
    return {"num_workers": 1}
```

Forced to single-worker. File appends are not safe with concurrent writers.

---

## 6. Data Flow Through the Pipeline

For a single utterance, the data dictionary evolves as follows:

```
NemoTarShardReaderStage:
  {audio_filepath, duration, text, source_lang, waveform, sample_rate, corpus}

InitializeFieldsStage:
  {audio_filepath, duration, source_lang, granary_v1_prediction, _skip_me="",
   waveform, sample_rate, corpus}
  (text renamed, _skip_me added, drop_keys removed)

InferenceQwenOmniStage:
  {..., qwen3_prediction_s1="Hello, my name is John.", _skip_me=""}
  (waveform removed if keep_waveform=False)

WhisperHallucinationStage:
  {..., _skip_me="" or "Hallucination:WhisperHallucination_omni"}

SelectBestPredictionStage:
  {..., best_prediction="Hello, my name is John."}

FastTextLIDStage:
  {..., _skip_me="" or "Wrong language:FastTextLID"}

RegexSubstitutionStage:
  {..., cleaned_text="Hello my name is John"}

AbbreviationConcatStage:
  {..., abbreviated_text="Hello my name is John", abbreviations=[]}

PnCRestorationStage:
  {..., pnc_text="Hello, my name is John."}

PnCContentGuardStage:
  {..., pnc_text="Hello, my name is John.", rejected_pnc_text=""}

ShardedManifestWriterStage:
  (writes all fields to JSONL, returns FileGroupTask)
```

At any stage, if `_skip_me` is non-empty, subsequent stages either:
- Pass through without processing (CPU stages check `_skip_me` first)
- Set output to empty string (PnC sets `pnc_text=""` for skipped entries)

The utterance is still written to the output manifest — the `_skip_me`
field allows downstream consumers to filter as needed.

---

## 7. NvLLMOps Integration Notes

This pipeline is invoked by two different orchestration systems in the NvLLMOps repository:

### Kratos (Kubeflow Pipelines)
- **Entry:** `wf_harvest_curator.py` → `run_curator.py` → `subprocess.run("python main.py --config-path=... ...")`
- **Data source:** NeMo-tarred data on Swiftstack S3 (streamed via `s3_endpoint_url: https://pdx.s8k.io`)
- **Models:** Downloaded from HuggingFace Hub at runtime (token from Vault: `HARVEST_SDP_HF_TOKEN`)
- **Output:** Uploaded to Swift via `upload_curator_output()` after pipeline completes
- **Resource:** `gpu.l40s.4` — 4x L40S GPU, 218 CPU, 980Gi RAM, 5680Gi disk

### Slurm (HPC)
- **Entry:** `run_granary_v2.sh` → `srun --container-image=... bash -c "python main.py ..."`
- **Data source:** NeMo-tarred data on Lustre filesystem (mounted into container)
- **Models:** Pre-cached on Lustre (`HF_HUB_OFFLINE=1`, no internet from compute nodes)
- **Output:** Written directly to Lustre output directory
- **Resource:** Configurable via `config.env` (8 GPUs typical)

The pipeline code itself is identical in both paths — only the orchestration, data access, and model caching differ.
