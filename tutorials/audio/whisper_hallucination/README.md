# Whisper Hallucination Filtering

Flag ASR transcripts that look like Whisper hallucinations in an existing
manifest. The pipeline is rule-based and runs on CPU; no model is downloaded.

## Pipeline

`pipeline.yaml` runs three stages:

1. `ManifestReader` reads the input JSONL manifest.
2. `WhisperHallucinationStage` runs four checks on `pred_text`. Any hit flags
   the row:
   - **Repeated n-grams**: the unique-word ratio is at or below
     `unique_words_threshold`.
   - **Long word**: a word reaches `long_word_threshold` characters
     (`agglutinative_long_word_threshold` for agglutinative or compounding
     languages), or the longest word is `long_word_rel_threshold` times longer
     than the next longest. The relative check is off for agglutinative
     languages.
   - **Phrase match**: the transcript is a known hallucination phrase from
     `phrases.txt`, such as "Thank you for watching".
   - **High character rate**: characters per second exceed `max_char_rate`.
3. `ManifestWriterStage` writes every row, flagged or not, to `output_path`.

## Quick start

Each input row needs `pred_text`, `_skipme`, and `duration`. `language` is
optional. From the repository root, run the bundled two-row sample:

```bash
uv sync --extra audio_cpu
source .venv/bin/activate

python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/whisper_hallucination \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/text_filtering/whisper_hallucination_input.jsonl \
  output_path=/tmp/whisper_hallucination_output.jsonl
```

`--config-path` is relative to `nemo_curator/config/run.py`. In the output,
the `"Thank you."` row has `"_skipme": "Hallucination:WhisperHallucination"`
and `additional_notes.WhisperHallucination` set to `hallucination (phrase_match)`;
the other row keeps an empty `_skipme` and gets the note `passed`.

## Output

A flagged row gets `_skipme` set to `Hallucination:<stage name>`, and every row
gets a note under `additional_notes.<stage name>` naming the result and the check
that fired. An existing
non-hallucination `_skipme` value from an earlier filter is kept. Downstream
stages can drop flagged rows by filtering on `_skipme`.

## Re-checking a recovered transcript

Set `overwrite=true` to re-check rows that already carry a hallucination flag,
for example after a second ASR pass. A row that now passes has its flag
cleared, and `additional_notes.WhisperHallucination` is set to `recovered` (or
to the lower-cased `recovery_value`).

## Parameters

| Override | Default | Effect |
|---|---|---|
| `unique_words_threshold` | `0.4` | Lower values flag only heavier repetition |
| `long_word_threshold` | `25` | Absolute long-word length for most languages |
| `agglutinative_long_word_threshold` | `35` | Absolute long-word length for agglutinative languages |
| `long_word_rel_threshold` | `3.0` | Longest word versus the next-longest word |
| `max_char_rate` | `40.0` | Characters per second above which a row is flagged |
| `common_hall_file` | `tutorials/audio/whisper_hallucination/phrases.txt` | Known hallucination phrases, one per line |
