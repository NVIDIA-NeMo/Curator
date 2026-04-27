# Granary v2 ASR Postprocessing Pipeline

Postprocessing pipeline for Granary v2 ASR manifests. Reads JSONL manifests produced by ASR inference, cleans and filters transcriptions, and writes output manifests with a `skip_me` flag marking low-quality entries.

## Pipeline stages

| # | Stage | What it does |
|---|---|---|
| 1 | `ALMManifestReader` | Reads JSONL — one `AudioTask` per line |
| 2 | `InitializeFieldsStage` | Copies `pred_text` → `cleaned_text`; sets `skip_me = 0` |
| 3 | `RegexSubstitutionStage` | Normalizes `cleaned_text` (quotes, dashes, brackets, whitespace) |
| 4 | `WhisperHallucinationStage` | Sets `skip_me = 1` for repeated n-grams, long words, known hallucination phrases, or abnormal char/duration rates |
| 5 | `FastTextLIDStage` | Sets `skip_me = 1` for non-English or low-confidence language ID |
| 6 | `FinalizeFieldsStage` | Renames `text` → `v1_text`, promotes `cleaned_text` → `text`, drops `pnc`/`itn`/`timestamp` |
| 7 | `ALMManifestWriterStage` | Writes **all** entries to output — both clean (`skip_me=0`) and flagged (`skip_me=1`) |

All entries are written to the output. Use `skip_me` downstream to filter or inspect flagged entries.

## Output schema

| Field | Description |
|---|---|
| `text` | Cleaned and normalized transcription |
| `v1_text` | Original reference text from the input manifest |
| `pred_text` | Raw ASR prediction (unchanged) |
| `skip_me` | `0` = clean, `1` = flagged by hallucination or LID filter |
| `audio_filepath` | Path to audio file |
| `duration` | Audio duration in seconds |
| All other original fields | Preserved as-is (except `pnc`, `itn`, `timestamp` which are dropped) |

## Bundled config files

| File | Purpose |
|---|---|
| `common.yaml` | Regex substitution rules applied to `cleaned_text` |
| `en.txt` | Known Whisper hallucination phrases (one per line) |

Both are used by default — no need to pass them as arguments.

## Running on Slurm

### Quick start

```bash
bash tutorials/audio/granary_v2_postprocessing/submit.sh \
    /path/to/output_root \
    /path/to/input_dir
```

`submit.sh` finds every `*.jsonl` under `input_dir` recursively, groups them into chunks of `MANIFESTS_PER_JOB` (default 128), and submits one Slurm job per chunk. All jobs run in parallel. The output directory structure mirrors the input:

```
input:   input_dir/ytc/en2/manifest_0.jsonl
output:  output_root/ytc/en2/manifest_0.jsonl
```

### Tuning chunk size

The default is 128 manifests per job. Override with the `MANIFESTS_PER_JOB` environment variable:

```bash
# Fewer, heavier jobs (large manifests)
MANIFESTS_PER_JOB=256 bash submit.sh /path/to/output /path/to/input

# More, lighter jobs (small manifests, want more parallelism)
MANIFESTS_PER_JOB=32 bash submit.sh /path/to/output /path/to/input
```

For 6552 manifests: `MANIFESTS_PER_JOB=128` → 52 jobs, `MANIFESTS_PER_JOB=32` → 205 jobs.

### Resuming interrupted runs

Just resubmit the same command. Any manifest whose output file already exists and is non-empty is skipped automatically. Partially written files (from preempted jobs) are ignored and reprocessed.

Check progress before resubmitting:

```bash
INPUT=/path/to/input_dir
OUTPUT=/path/to/output_root

TOTAL=$(find "$INPUT" -name "*.jsonl" | wc -l)
DONE=$(find "$OUTPUT" -name "*.jsonl" ! -name "*.tmp" | wc -l)
echo "Done: $DONE / $TOTAL  (remaining: $((TOTAL - DONE)))"
```

### Sequential waves (dependent jobs)

Pass multiple input directories — each wave starts after the previous one finishes:

```bash
bash tutorials/audio/granary_v2_postprocessing/submit.sh \
    /path/to/output_root \
    /path/to/input_dir_batch_1 \
    /path/to/input_dir_batch_2
```

Wave 2 waits for all Wave 1 jobs to finish (`afterany` dependency) before starting.

### Single job (one directory)

```bash
sbatch tutorials/audio/granary_v2_postprocessing/run.sh \
    /path/to/input_dir \
    /path/to/output_root
```

Processes all `*.jsonl` files under `input_dir` sequentially within one job.

## Running locally / interactively

```bash
export PYTHONPATH="/path/to/Curator:${PYTHONPATH:-}"

python tutorials/audio/granary_v2_postprocessing/pipeline.py \
    --input_dir /path/to/input_dir \
    --output_dir /path/to/output_root \
    --fasttext_model /path/to/lid.176.ftz
```

To process specific manifests only:

```bash
python tutorials/audio/granary_v2_postprocessing/pipeline.py \
    --input_dir /path/to/input_dir \
    --manifests /path/to/input_dir/corpus/manifest_0.jsonl \
                /path/to/input_dir/corpus/manifest_1.jsonl \
    --output_dir /path/to/output_root \
    --fasttext_model /path/to/lid.176.ftz
```

`--input_dir` is always the root used to compute relative output paths. All `--manifests` paths must be under `--input_dir`.

## All arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | required | Root input directory; also used as the anchor for mirroring output paths |
| `--output_dir` | required | Root output directory |
| `--manifests` | — | Process specific manifests instead of scanning all of `input_dir` (one or more paths, all must be under `--input_dir`) |
| `--fasttext_model` | `lid.176.ftz` | FastText LID model path (`lid.176.bin` or `lid.176.ftz`) |
| `--regex_yaml` | `common.yaml` | Regex substitution rules YAML |
| `--hall_phrases` | `en.txt` | Hallucination phrases file (one phrase per line) |
| `--target_lang` | `en` | Expected language code for LID |
| `--min_lang_prob` | `0.3` | Minimum FastText confidence to keep an entry |
| `--unique_words_threshold` | `0.4` | Unique-word ratio below which repeated n-grams are flagged |
| `--long_word_threshold` | `25` | Character length above which a word is flagged as abnormally long |
| `--long_word_rel_threshold` | `3.0` | Longest/second-longest word ratio for long-word detection |
| `--max_char_rate` | `40.0` | chars/s above which text is considered impossibly dense (hallucinated sentence over short audio) |
| `--verbose` | off | Enable DEBUG logging (shows per-entry flagging reasons) |

## Hallucination detection details

`WhisperHallucinationStage` applies four checks to `cleaned_text`:

| Check | Triggers when |
|---|---|
| Repeated n-grams | Unique-word ratio ≤ `unique_words_threshold` |
| Long word (absolute) | Any word ≥ `long_word_threshold` characters |
| Long word (relative) | Longest word is ≥ `long_word_rel_threshold` × second-longest |
| Phrase match | Text matches or starts with a phrase from `en.txt` (prefix match for phrases ≥ 8 chars) |
| High char rate | `sum(word lengths) / duration > max_char_rate` |

Add new hallucination phrases to `en.txt`, one per line.

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
