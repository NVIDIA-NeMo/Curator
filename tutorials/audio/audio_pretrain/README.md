# Long-form audio cutting for ALM pretraining

This tutorial cuts long-form diarized + transcribed audio into bounded-duration mono snippets that are suitable as a foundation for **audio LLM (ALM) pretraining**.

The output snippet manifest is intentionally generic. Each row keeps a snippet-relative `segments` list (with speaker, text, ITN text, and word-level timestamps), so the same snippets can be reused to construct:

- interleaved audio/text continuation data,
- ASR training pairs (snippet audio + transcript),
- TTS training pairs (transcript + snippet audio),
- speaker-diarization training data (snippet audio + per-speaker timestamps).

## Inputs

- `input_manifest`: a JSONL file, one row per long-form audio. Required fields per row: an `id` (non-empty; must be unique within the manifest — duplicates are skipped with a warning), an `audio_filepath` (only the basename is used), and a `segments` list. Each segment has `speaker`, `start`, `end`, `text`, `text_ITN`, and a `words` list of `{word, start, end}`.
- `audio_dir`: directory holding the source audio files. By default each row's audio is resolved as `audio_dir / basename(audio_filepath)` (see `audio_path_resolution`).
- `workflow.0.audio_path_resolution`: how the reader maps a row's `audio_filepath` to an on-disk path. Choices: `basename` (default; flat staging dir, also rejects manifests with duplicate basenames so a hidden collision can't silently route two different source recordings to the same file), `relative` (`audio_dir / value`; preserves subdirectories), `as_is` (trust the manifest's path verbatim).

## Outputs

- `output_dir`: directory for pipeline outputs. By default the tar, manifest, and metrics files below are written here; override `workflow.0.output_audio_tar_path`, `workflow.0.output_manifest_path`, or `workflow.0.metrics_path` to move them.
- `snippets.tar`: tar archive of snippet audio files, one member per snippet, named `<snippet_id>.<ext>` (WebDataset/Energon compatible, sorted lexicographically). Each manifest row's `audio_filepath` refers to this tar-internal basename.
- `snippets.jsonl`: JSONL where each row is the original metadata of the source audio (with `alignment` removed) plus the new fields `snippet_id`, updated `audio_filepath` (tar-internal basename), updated `duration`, and a snippet-relative `segments` list. The original `id` is preserved unchanged so snippets are joinable back to their source.
- `metrics_summary.json`: JSON summary with input/output counts, total durations, dropped-snippet/segment breakdowns (empty / overlap / too-long / too-short / no-text / repetition), a 30-second-bin histogram of snippet durations, a per-original breakdown, and up to 1000 example texts of snippets dropped by the repetition filter.

## Cutting algorithm

1. **Drop empty segments** — a segment with no text and no words is dropped.
2. **Drop overlapping segments** — two segments overlap (and both are discarded) iff their intersection is at least `min_overlap_sec` seconds **or** one fully contains the other. Smaller incidental overlaps are kept.
3. **Greedy contiguous packing** — surviving segments are walked in start-time order. The current snippet grows while (a) `last.end - first.start <= max_duration_sec` AND (b) the gap from the current snippet's last accepted segment's `end` to the next segment's `start` is at most `max_segment_gap_in_snippet`. Either constraint failing closes the snippet and opens a new one starting from the current segment. Segments are never split.
4. **Drop snippets that don't fit** — a snippet whose span exceeds `max_duration_sec` (which only happens when a single segment alone is too long), is shorter than `min_duration_sec`, or has empty concatenated text is dropped and counted under `too_long` / `too_short` / `no_text`.
5. **Drop snippets with repetitive text** — for each candidate snippet the joined text is tokenized with the HuggingFace fast tokenizer at `tokenizer_path` and any snippet whose token-id n-gram histogram has an entry above the configured threshold is dropped (default `ngram_n: 10`, `ngram_max_count: 3`). This catches Whisper-style decoding loops without paying audio I/O cost on filtered snippets.
6. **Audio extraction** — for each surviving snippet the source audio is sliced, channel-averaged to mono if needed, resampled to `target_sample_rate` (default 16000), and written as a member of the tar archive named `<snippet_id>.<output_format>`. Snippet `segments` (and word timestamps) are shifted so the snippet starts at `0.0`.

## Example

```bash
python nemo_curator/config/run.py \
    --config-path ../../tutorials/audio/audio_pretrain \
    --config-name pipeline \
    input_manifest=/path/to/long_form.jsonl \
    audio_dir=/path/to/audios \
    output_dir=/path/to/output \
    tokenizer_path=/path/to/hf_tokenizer_dir \
    max_duration_sec=30
```

`--config-path` is relative to `nemo_curator/config/run.py`. The YAML runs
`AudioPretrainWorkflow`, which removes stale output shards before the run and
merges the per-worker shards into the final tar, manifest, and metrics files
afterwards (also when the run fails, so partial output is kept).

For large datasets on Xenna, limit concurrent Ray worker start-up and raise the
state-API list limit before running:

```bash
export RAY_worker_maximum_startup_concurrency=4
export RAY_MAX_LIMIT_FROM_API_SERVER=50000
```

`tokenizer_path` is required and accepts either a local directory loadable by `AutoTokenizer.from_pretrained` or a HuggingFace Hub repository id (e.g. `openai/whisper-large-v3`); when it's a repo id, the tokenizer is fetched once per node in `setup_on_node` so workers only ever load from the local cache. Use `workflow.0.tokenizer_cache_dir` and `workflow.0.hf_token` to override the cache location and provide a token for gated repos.

By default the pipeline runs on the Xenna streaming executor; pass `backend=ray_data` to switch. The other cutting parameters (`min_duration_sec`, `min_overlap_sec`, `max_segment_gap_in_snippet`, `ngram_n`, `ngram_max_count`, `target_sample_rate`, `output_format`, ...) are set under `workflow.0` in `pipeline.yaml` and can be overridden the same way, for example `workflow.0.output_format=wav`.

## Dry-run mode

Pass `workflow.0.dry_run=true` to skip all audio I/O — no snippet audio files are written — while still producing the output manifest and metrics summary. This is the right way to size up a real dataset before committing to a full run: you can read the metrics JSON to see how many snippets would be produced, the dropped-segment breakdown, and the duration histogram, in seconds rather than minutes.

```bash
python nemo_curator/config/run.py \
    --config-path ../../tutorials/audio/audio_pretrain \
    --config-name pipeline \
    input_manifest=/path/to/long_form.jsonl \
    audio_dir=/path/to/audios \
    output_dir=/tmp/snippets_dryrun \
    tokenizer_path=/path/to/hf_tokenizer_dir \
    max_duration_sec=30 \
    workflow.0.dry_run=true
```

In dry-run the manifest's `audio_filepath` still points at the tar-internal basename the snippet *would* have had, but no tar file is written. The snippet `duration` is the planned `end - start` rather than the resampled-frame-count duration of a real run (the difference is at most one frame at `target_sample_rate`, ≈62 µs at 16 kHz).
