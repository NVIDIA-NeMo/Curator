# Granary v2 ASR Postprocessing Pipeline

Postprocessing pipeline for Granary v2 ASR manifests. Reads JSONL manifests produced by ASR inference, cleans and filters transcriptions, and writes output manifests with a `skip_me` field marking low-quality entries.

## Prerequisites

- Python 3.10+
- NeMo Curator installed (see [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- No GPU required — all stages are CPU-only

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

## Pipeline stages

| # | Stage | What it does |
|---|---|---|
| 1 | `ALMManifestReader` | Reads JSONL — one `AudioTask` per line |
| 2 | `InitializeFieldsStage` | Renames `text` → `granary_v1_prediction`; sets `skip_me = ""` (empty = not flagged) |
| 3 | `WhisperHallucinationStage` | Sets `skip_me = "Hallucination"` for repeated n-grams, long words, known hallucination phrases, or abnormal char/duration rates |
| 4 | `FastTextLIDStage` | Sets `skip_me = "Wrong language"` or `"Low probability of language"` for non-English or low-confidence entries |
| 5 | `RegexSubstitutionStage` | Reads `pred_text`, applies regex normalization, writes result to `cleaned_text` |
| 6 | `FinalizeFieldsStage` | Drops `pnc`/`itn`/`timestamp` keys |
| 7 | `ALMManifestWriterStage` | Writes **all** entries to output — both clean and flagged |

All entries are written to the output. Use `skip_me` downstream to filter or inspect flagged entries.

## Output schema

| Field | Description |
|---|---|
| `granary_v1_prediction` | Original `text` field from the input manifest (renamed by `InitializeFieldsStage`) |
| `pred_text` | Raw ASR prediction (unchanged) |
| `cleaned_text` | Normalized transcription after regex substitution |
| `skip_me` | `""` = clean, or a reason string (`"Hallucination"`, `"Wrong language"`, etc.) |
| `audio_filepath` | Path to audio file |
| `duration` | Audio duration in seconds |
| All other original fields | Preserved as-is (except `pnc`, `itn`, `timestamp` which are dropped) |

## Bundled config files

| File | Purpose |
|---|---|
| `common.yaml` | Regex substitution rules applied to `cleaned_text` |
| `en.txt` | Known Whisper hallucination phrases (one per line; optional trailing frequency counts are stripped) |

Both are used by default — no need to pass them as arguments.

## Quick Start

From the Curator repository root:

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

### Resuming interrupted runs

Just rerun the same command. Any manifest whose output file already exists and is non-empty is skipped automatically. Partially written files (from preempted jobs) are ignored and reprocessed.

Check progress before resubmitting:

```bash
INPUT=/path/to/input_dir
OUTPUT=/path/to/output_root

TOTAL=$(find "$INPUT" -name "*.jsonl" | wc -l)
DONE=$(find "$OUTPUT" -name "*.jsonl" ! -name "*.tmp" | wc -l)
echo "Done: $DONE / $TOTAL  (remaining: $((TOTAL - DONE)))"
```

## All arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | required | Root input directory; also used as the anchor for mirroring output paths |
| `--output_dir` | required | Root output directory |
| `--manifests` | — | Process specific manifests instead of scanning all of `input_dir` (one or more paths, all must be under `--input_dir`) |
| `--fasttext_model` | `lid.176.ftz` | FastText LID model path (`lid.176.bin` or `lid.176.ftz`); downloaded automatically if not found locally |
| `--regex_yaml` | `common.yaml` | Regex substitution rules YAML |
| `--hall_phrases` | `en.txt` | Hallucination phrases file (one phrase per line) |
| `--target_lang` | `en` | Expected language code for LID |
| `--min_lang_prob` | `0.8` | Minimum FastText confidence to keep an entry |
| `--unique_words_threshold` | `0.4` | Unique-word ratio below which repeated n-grams are flagged |
| `--long_word_threshold` | `25` | Character length above which a word is flagged as abnormally long |
| `--long_word_rel_threshold` | `3.0` | Longest/second-longest word ratio for long-word detection |
| `--char_rate_threshold` | `4.0` | chars/s below which text is considered too sparse (long silence + few words) |
| `--max_char_rate` | `40.0` | chars/s above which text is considered impossibly dense (hallucinated sentence over short audio) |
| `--verbose` | off | Enable DEBUG logging (shows per-entry flagging reasons) |

## Hallucination detection details

`WhisperHallucinationStage` applies five checks to `pred_text`:

| Check | Triggers when |
|---|---|
| Repeated n-grams | Unique-word ratio ≤ `unique_words_threshold` |
| Long word (absolute) | Any word ≥ `long_word_threshold` characters |
| Long word (relative) | Longest word is ≥ `long_word_rel_threshold` × second-longest |
| Phrase match | Text matches or starts with a phrase from `en.txt` (prefix match for phrases ≥ 8 chars) |
| Low char rate | `sum(word lengths) / duration ≤ char_rate_threshold` |
| High char rate | `sum(word lengths) / duration > max_char_rate` |

Add new hallucination phrases to `en.txt`, one per line. Trailing frequency counts (e.g. `"Thank you 1297"`) are stripped automatically.

## Performance Notes

- All stages are CPU-bound (no GPU required)
- Processing is I/O-bound on large manifest directories
- The pipeline processes one manifest at a time sequentially; parallelism is within each manifest via XennaExecutor

## Stage implementation

The filtering stages live in `nemo_curator/stages/audio/text_filtering/` and can be used in any custom pipeline:

```python
from nemo_curator.stages.audio.text_filtering import (
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
    FastTextLIDStage,
    FinalizeFieldsStage,
)
```
