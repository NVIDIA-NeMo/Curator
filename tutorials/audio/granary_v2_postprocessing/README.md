# Granary v2 ASR Postprocessing Pipeline

Postprocessing pipeline for Granary v2 ASR manifests. Reads JSONL manifests produced by ASR inference, cleans and filters transcriptions, and writes output manifests with a `skip_me` field marking low-quality entries.

## Prerequisites

- Python 3.10+
- NeMo Curator installed (see [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- [FastText LID model](https://fasttext.cc/docs/en/language-identification.html) (`lid.176.ftz` or `lid.176.bin`)
- No GPU required — all stages are CPU-only

### System dependencies

None beyond pip. No `sox`, `ffmpeg`, or other system packages needed.

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cpu
source .venv/bin/activate
```

Or with pip:

```bash
pip install -e ".[audio_cpu]"
```

## Quick Start

```bash
python tutorials/audio/granary_v2_postprocessing/post_processing_pipeline.py \
    --input_dir /path/to/input_dir \
    --output_dir /path/to/output_root \
    --fasttext_model /path/to/lid.176.ftz
```

To process specific manifests only:

```bash
python tutorials/audio/granary_v2_postprocessing/post_processing_pipeline.py \
    --input_dir /path/to/input_dir \
    --manifests /path/to/input_dir/corpus/manifest_0.jsonl \
                /path/to/input_dir/corpus/manifest_1.jsonl \
    --output_dir /path/to/output_root \
    --fasttext_model /path/to/lid.176.ftz
```

`--input_dir` is always the root used to compute relative output paths. All `--manifests` paths must be under `--input_dir`.

## Pipeline Architecture

```
ALMManifestReader
  Reads JSONL — one AudioTask per line
      |
      v
InitializeFieldsStage
  Renames text → granary_v1_prediction; sets skip_me = ""
  Drops prompt-engineering fields (answer, source_lang, …)
      |
      v
WhisperHallucinationStage
  Flags: repeated n-grams, long words, known phrases, abnormal char rates
  → skip_me = "Hallucination" or "Empty text"
      |
      v
FastTextLIDStage
  Flags: wrong language or low-confidence language ID
  → skip_me = "Wrong language" or "Low probability of language"
      |
      v
RegexSubstitutionStage
  Applies regex rules from common.yaml to pred_text → cleaned_text
      |
      v
FinalizeFieldsStage
  Drops pnc/itn/timestamp keys
      |
      v
ALMManifestWriterStage
  Writes ALL entries (both clean and flagged) to output JSONL
```

## Output Schema

| Field | Type | Description |
|---|---|---|
| `audio_filepath` | `str` | Path to audio file (preserved from input) |
| `duration` | `float` | Audio duration in seconds (preserved from input) |
| `granary_v1_prediction` | `str` | Original `text` field from input (renamed by `InitializeFieldsStage`) |
| `pred_text` | `str` | Raw ASR prediction (unchanged) |
| `cleaned_text` | `str` | Normalized transcription after regex substitution |
| `skip_me` | `str` | `""` = clean, or a reason string (see below) |
| All other original fields | varies | Preserved as-is (except `pnc`, `itn`, `timestamp` which are dropped) |

### `skip_me` values

| Value | Set by | Meaning |
|---|---|---|
| `""` (empty) | default | Entry passed all checks |
| `"Empty text"` | `WhisperHallucinationStage` | ASR returned empty/whitespace-only transcription |
| `"Hallucination"` | `WhisperHallucinationStage` | One or more hallucination checks triggered |
| `"Wrong language"` | `FastTextLIDStage` | Detected language != `target_lang` |
| `"Low probability of language"` | `FastTextLIDStage` | Confidence < `min_lang_prob` |

Downstream consumers should filter on `skip_me == ""` to get clean entries only.

## All Arguments

### Required

| Argument | Description |
|---|---|
| `--input_dir` | Root input directory; also used as the anchor for mirroring output paths |
| `--output_dir` | Root output directory |

### Optional

| Argument | Default | Description |
|---|---|---|
| `--manifests` | — | Process specific manifests instead of scanning all of `input_dir` (one or more paths, all must be under `--input_dir`) |
| `--fasttext_model` | `lid.176.ftz` | FastText LID model path; downloaded automatically if not found locally |
| `--regex_yaml` | `common.yaml` | Regex substitution rules YAML |
| `--hall_phrases` | `en.txt` | Hallucination phrases file (one phrase per line; optional trailing frequency counts are stripped) |
| `--target_lang` | `en` | Expected language code for LID |
| `--min_lang_prob` | `0.8` | Minimum FastText confidence to keep an entry |
| `--unique_words_threshold` | `0.4` | Unique-word ratio below which repeated n-grams are flagged |
| `--long_word_threshold` | `25` | Character length above which a word is flagged as abnormally long |
| `--long_word_rel_threshold` | `3.0` | Longest/second-longest word ratio for long-word detection |
| `--char_rate_threshold` | `4.0` | chars/s below which text is considered too sparse |
| `--max_char_rate` | `40.0` | chars/s above which text is considered impossibly dense |
| `--verbose` | off | Enable DEBUG logging (shows per-entry flagging reasons) |

## Parameter Tuning

### Hallucination detection thresholds

| Parameter | What it controls | Too low | Too high | Suggested range |
|---|---|---|---|---|
| `unique_words_threshold` | Flags repetitive text (lower = more diverse required) | Misses subtle repetitions | Flags normal text with repeated common words | 0.3–0.5 |
| `long_word_threshold` | Absolute character count for long words | Flags legitimate compound words (e.g. German) | Misses garbled long tokens | 20–30 |
| `long_word_rel_threshold` | How much longer than the 2nd-longest word | Flags texts with one slightly longer word | Misses outlier words | 2.0–4.0 |
| `char_rate_threshold` | Characters per second below which = "too sparse" | Misses silent-audio hallucinations | Flags slow, deliberate speech | 3.0–5.0 |
| `max_char_rate` | Characters per second above which = "impossibly dense" | Flags fast speech / short utterances | Misses confabulated text | 30.0–50.0 |

### Language ID thresholds

| Parameter | What it controls | Typical values |
|---|---|---|
| `min_lang_prob` | Minimum FastText confidence score (0.0–1.0) | **0.8** for general use; **0.6** for code-switched/multilingual data; **0.95** for monolingual corpora |
| `target_lang` | ISO 639-1 language code to expect | `en`, `de`, `fr`, etc. |

### Regex substitution

The `common.yaml` file contains ordered regex rules applied to the predicted text. Rules are applied sequentially — order matters. Edit `common.yaml` to customize normalization (e.g. removing filler annotations, normalizing punctuation).

## Bundled Config Files

| File | Purpose |
|---|---|
| `common.yaml` | Regex substitution rules applied to `cleaned_text` |
| `en.txt` | Known Whisper hallucination phrases (one per line; optional trailing frequency counts are stripped) |

Both are used by default — no need to pass them as arguments.

## Resuming Interrupted Runs

Just rerun the same command. Any manifest whose output file already exists and is non-empty is skipped automatically. Partially written files (from preempted jobs) are ignored and reprocessed.

Check progress before resubmitting:

```bash
INPUT=/path/to/input_dir
OUTPUT=/path/to/output_root

TOTAL=$(find "$INPUT" -name "*.jsonl" | wc -l)
DONE=$(find "$OUTPUT" -name "*.jsonl" ! -name "*.tmp" | wc -l)
echo "Done: $DONE / $TOTAL  (remaining: $((TOTAL - DONE)))"
```

## Performance

- **CPU-only** — no GPU required for text filtering stages
- **I/O-bound** on large manifest directories (reading/writing JSONL dominates)
- Processing is sequential per-manifest; parallelism is within each manifest via XennaExecutor workers
- **Throughput estimate**: ~50k–100k entries/min on a 32-core CPU node (bottlenecked by FastText LID and regex processing)
- FastText model loading happens once per worker in `setup()` — first-batch latency is higher

### When paired with Qwen3-Omni inference

When this pipeline runs downstream of `InferenceQwenOmniStage` (see `examples/audio/qwen_omni_inprocess/`), the GPU inference stage is the bottleneck — text filtering adds negligible overhead. Key tuning guidance from production benchmarks on YODAS (8x GPU, tp=2):

| Knob | Impact |
|---|---|
| `--fp8` + `--enforce_eager` + `--mm_cache_gb` | ~10% relative throughput gain (vLLM-specific args dominate) |
| `--batch_size` (1 → 32 → 512) | Negligible impact — 32 is a good default |
| Pre-download model to `$HF_HOME` | Avoids multi-worker download races on shared clusters |

For multi-node scale (O(million) hours), use the `InferenceServer` abstraction with Ray Serve instead of in-process vLLM. See `nemo_curator.core.serve`.

### Expected filtering ratios

Filtering ratios depend heavily on ASR quality and audio domain. Typical ranges on Granary v2 English data:

| Filter | Typical flag rate |
|---|---|
| Hallucination (all checks) | 2–10% |
| Wrong language | 1–5% |
| Low language probability | 3–8% |
| **Total flagged** | **5–15%** |

Use `--verbose` to inspect per-entry flagging reasons if your flag rate is unexpectedly high or low.

## Troubleshooting

| Issue | Solution |
|---|---|
| `FileNotFoundError: lid.176.ftz` | Download the FastText LID model: `wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz` |
| `AF_UNIX path length` error | `export RAY_TMPDIR=/tmp` |
| High hallucination flag rate | Lower `unique_words_threshold` or raise `char_rate_threshold`; inspect flagged entries with `--verbose` |
| All entries flagged as wrong language | Verify `--target_lang` matches your data; try lowering `--min_lang_prob` |
| Pipeline seems hung | Check log output — first manifest may take longer due to FastText model loading; use `--verbose` for progress |
| Partially written output files | Safe to rerun; the pipeline detects `.tmp` files and reprocesses them |

## Composability

The text filtering stages are reusable building blocks. Common compositions:

```
Qwen3-Omni inference → this pipeline (text filtering) → downstream training
FLEURS pipeline → this pipeline (to clean ASR output before fine-tuning)
Any ASR pipeline → WhisperHallucinationStage + FastTextLIDStage (cherry-pick stages)
```

Individual stages can be imported for custom pipelines:

```python
from nemo_curator.stages.audio.text_filtering import (
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
    FastTextLIDStage,
    FinalizeFieldsStage,
)
```

See `examples/audio/qwen_omni_inprocess/run_pipeline.py` for an end-to-end example that chains `NemoTarredAudioReader` → `InferenceQwenOmniStage` (GPU, with FP8 quantisation, engine retry, and batch chunking) → these text filtering stages → JSONL output.

## Hallucination Detection Details

`WhisperHallucinationStage` applies five checks to the predicted text:

| Check | Triggers when |
|---|---|
| Empty text | Transcription is empty or whitespace-only |
| Repeated n-grams | Unique-word ratio ≤ `unique_words_threshold` |
| Long word (absolute) | Any word ≥ `long_word_threshold` characters |
| Long word (relative) | Longest word is ≥ `long_word_rel_threshold` × second-longest |
| Phrase match | Text matches or starts with a phrase from `en.txt` (prefix match for phrases ≥ 8 chars) |
| Low char rate | `sum(word lengths) / duration ≤ char_rate_threshold` |
| High char rate | `sum(word lengths) / duration > max_char_rate` |

Add new hallucination phrases to `en.txt`, one per line. Trailing frequency counts (e.g. `"Thank you 1297"`) are stripped automatically.
