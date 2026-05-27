# Curating the FLEURS Dataset with NeMo Curator

The [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset contains spoken utterances across 100+ languages. This tutorial shows how to:

- Download a FLEURS split for a given language
- Run ASR inference using a NeMo model
- Compute WER and audio duration
- Filter by WER threshold
- Save results as JSONL

Note: Run these examples on GPUs for best performance.

### Quick start (Python script)

The Python pipeline downloads the data, runs ASR, computes WER/duration, filters, and writes JSONL results to `result/` under your `raw_data_dir`.

```bash
python tutorials/audio/fleurs/pipeline.py \
  --raw_data_dir ./example_audio/fleurs \
  --model_name nvidia/stt_hy_fastconformer_hybrid_large_pc \
  --lang hy_am \
  --split dev \
  --wer_threshold 75 \
  --gpus 1 \
  --backend xenna \
  --clean \
  --verbose
```

Key arguments:
- `--raw_data_dir`: Workspace directory for downloaded audio and outputs
- `--model_name`: NeMo ASR model (change per language)
- `--lang`: FLEURS language code (e.g., `hy_am`, `en_us`, etc.)
- `--split`: FLEURS split (`train`, `dev`, or `test`)
- `--wer_threshold`: Keep samples with WER less-or-equal to this value
- `--backend`: `xenna` (default) or `ray_data`
### Choosing a Backend

Both the Python script (`pipeline.py`) and the YAML runner (`run.py`) support two execution backends:

| Backend | Description | When to use |
|---------|-------------|-------------|
| `xenna` | Default executor. Uses Cosmos-Xenna streaming engine with automatic worker allocation. | Most workloads, CI/nightly benchmarks. |
| `ray_data` | Executor built on Ray Data `map_batches`. | Development, machines where Xenna cannot detect GPUs, or when Ray Data integration is preferred. |

**Python script** — pass `--backend`:

```bash
python tutorials/audio/fleurs/pipeline.py \
  --raw_data_dir ./example_audio/fleurs \
  --model_name nvidia/parakeet-tdt-0.6b-v2 \
  --lang en_us --split dev --wer_threshold 75 --gpus 1 \
  --backend ray_data
```

**YAML runner** — override `backend=`:

```bash
python tutorials/audio/fleurs/run.py \
  --config-path . --config-name pipeline.yaml \
  raw_data_dir=./example_audio/fleurs \
  backend=ray_data
```

### Alternative: YAML config + Hydra

You can run the same workflow by instantiating stages from a YAML config.

Option 1: Edit `pipeline.yaml` to set `raw_data_dir`, then run:

```bash
python tutorials/audio/fleurs/run.py \
  --config-path . --config-name pipeline.yaml \
  raw_data_dir=./example_audio/fleurs
```

Option 2: Override values from the command line without editing the file:

```bash
python tutorials/audio/fleurs/run.py \
  --config-path . --config-name pipeline.yaml \
  raw_data_dir=./example_audio/fleurs \
  data_split=dev \
  processors.0.lang=en_us \
  processors.1.model_name=nvidia/stt_en_conformer_ctc_large \
  processors.4.target_value=50.0
```

Note: `--config-path .` tells Hydra to look for configs in the same directory as `run.py`.

Notes on overrides (match indices in `processors` list inside `pipeline.yaml`):
- `processors.0.lang`: language for the FLEURS downloader stage
- `processors.1.model_name`: NeMo ASR model used for inference
- `processors.4.target_value`: WER threshold used for filtering
- `data_split`: top-level variable referenced by the first stage as `split`
- `backend`: `xenna` (default) or `ray_data`
### Output

Results are written as JSONL under `${raw_data_dir}/result`. Each line contains fields like:

```json
{"audio_filepath": "relative/path/to/audio.wav", "text": "reference transcription", "duration": 4.21}
```

Depending on configuration, you may also compute and filter by WER using the predicted text. The example configs keep samples with WER less than or equal to the threshold.

### GPU/CPU, cleaning, and performance notes

- ASR inference is GPU-accelerated. The YAML config requests one GPU via `processors.1.resources.gpus: 1.0`. For CPU fallback with the Python script, pass `--gpus 0`.
- Use `--clean` to remove an existing `result/` directory before writing outputs.
- Use `--verbose` for DEBUG-level logs, helpful for intermittent issues.
- Reduce or increase batch sizes by editing `pipeline.py` or `pipeline.yaml` (e.g., `CreateInitialManifestFleursStage().with_(batch_size=4)`).
- Lower-memory GPUs may require smaller batch sizes; high-memory GPUs can use larger ones for higher throughput.

### Customizing languages and models

- FLEURS language codes follow the dataset convention (e.g., `en_us`, `fr_fr`, `hy_arm`). See the dataset card for a complete list.
- Use a corresponding NeMo model for your target language. For English, for example: `nvidia/stt_en_conformer_ctc_large`. For Armenian (shown above): `nvidia/stt_hy_fastconformer_hybrid_large_pc`.

### What the stages do

Both the Python and YAML flows compose the same stages:
- Download/create initial manifest for the requested FLEURS split
- Run ASR inference with a specified NeMo model
- Compute pairwise WER between reference and predicted text
- Compute audio durations
- Filter samples by WER threshold
- Convert to document format and write JSONL

After running, inspect `${raw_data_dir}/result` to explore your curated manifest(s).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SIGSEGV / actor crash during model load | See [Known Issues](../README.md#known-issues) — set `OTEL_SDK_DISABLED=true` |
