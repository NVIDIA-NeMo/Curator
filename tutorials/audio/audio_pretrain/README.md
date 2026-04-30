# Long-form audio cutting for ALM pretraining

This tutorial cuts long-form diarized + transcribed audio into bounded-duration mono snippets that are suitable as a foundation for **audio LLM (ALM) pretraining**.

The output snippet manifest is intentionally generic. Each row keeps a snippet-relative `segments` list (with speaker, text, ITN text, and word-level timestamps), so the same snippets can be reused to construct:

- interleaved audio/text continuation data,
- ASR training pairs (snippet audio + transcript),
- TTS training pairs (transcript + snippet audio),
- speaker-diarization training data (snippet audio + per-speaker timestamps).

## Inputs

- `--input-manifest`: a JSONL file, one row per long-form audio. Required fields per row: an `id`, an `audio_filepath` (only the basename is used), and a `segments` list. Each segment has `speaker`, `start`, `end`, `text`, `text_ITN`, and a `words` list of `{word, start, end}`.
- `--audio-dir`: directory holding the source audio files. Each row's audio is resolved as `audio_dir / basename(audio_filepath)`.

## Outputs

- `--output-dir`: directory of snippet audio files, one per snippet, named `<original_id>_<start>_<end>.<ext>` (timestamps are seconds with three decimals, in the source audio's coordinate system).
- `--output-manifest`: JSONL where each row is the original metadata of the source audio (with `alignment` removed) plus the new fields `snippet_id`, updated `audio_filepath`, updated `duration`, and a snippet-relative `segments` list. The original `id` is preserved unchanged so snippets are joinable back to their source.
- `--metrics-path`: JSON summary with input/output counts, total durations, dropped-segment breakdowns (empty / overlap / too-long / too-short / no-text), a 30-second-bin histogram of snippet durations, and a per-original breakdown.

## Cutting algorithm

1. **Drop empty segments** — a segment with no text and no words is dropped.
2. **Drop overlapping segments** — two segments overlap (and both are discarded) iff their intersection is at least `--min-overlap-sec` seconds **or** one fully contains the other. Smaller incidental overlaps are kept.
3. **Greedy contiguous packing** — surviving segments are walked in start-time order. The current snippet grows while (a) `last.end - first.start <= --max-duration-sec` AND (b) the gap from the current snippet's last accepted segment's `end` to the next segment's `start` is at most `--max-segment-gap-in-snippet`. Either constraint failing closes the snippet and opens a new one starting from the current segment. Segments are never split.
4. **Drop snippets that don't fit** — a snippet whose span exceeds `--max-duration-sec` (which only happens when a single segment alone is too long), is shorter than `--min-duration-sec`, or has empty concatenated text is dropped and counted under `too_long` / `too_short` / `no_text`.
5. **Audio extraction** — for each surviving snippet the source audio is sliced, channel-averaged to mono if needed, resampled to `--target-sample-rate` (default 16000), and written to `--output-dir`. Snippet `segments` (and word timestamps) are shifted so the snippet starts at `0.0`.

## Example

```bash
python -m tutorials.audio.audio_pretrain.run \
    --input-manifest /path/to/long_form.jsonl \
    --audio-dir /path/to/audios \
    --output-dir /path/to/snippets \
    --output-manifest /path/to/snippets.jsonl \
    --metrics-path /path/to/metrics_summary.json \
    --max-duration-sec 30
```

By default the pipeline runs on the Xenna streaming executor; pass `--backend ray_data` or `--backend ray_actor_pool` to switch.

## Dry-run mode

Pass `--dry-run` to skip all audio I/O — no snippet audio files are written — while still producing the output manifest and metrics summary. This is the right way to size up a real dataset before committing to a full run: you can read the metrics JSON to see how many snippets would be produced, the dropped-segment breakdown, and the duration histogram, in seconds rather than minutes.

```bash
python -m tutorials.audio.audio_pretrain.run \
    --input-manifest /path/to/long_form.jsonl \
    --audio-dir /path/to/audios \
    --output-dir /tmp/snippets_unused \
    --output-manifest /tmp/snippets_dryrun.jsonl \
    --metrics-path /tmp/metrics_dryrun.json \
    --max-duration-sec 30 \
    --dry-run
```

In dry-run the manifest's `audio_filepath` still points at the path the file *would* have been written to, but the file does not exist on disk. The snippet `duration` is the planned `end - start` rather than the resampled-frame-count duration of a real run (the difference is at most one frame at `--target-sample-rate`, ≈62 µs at 16 kHz).
