# Payload-neutral global FastConformer bucketing

This pipeline keeps scheduling metadata separate from audio bytes:

1. `GlobalAudioManifestPlannerStage` reads the complete JSONL manifest, segments rows by duration, and globally
   packs file-range descriptors with `BatchPolicy`. It stores each original parent row once in a job-scoped metadata
   repository and does not decode audio. Every segment also exposes `audio_filepath`, segment `duration`,
   `segment_start_s`, and `segment_duration_s` for generic materializers.
2. `DispatchBatchTask` is the atomic handoff. Executors schedule each envelope as one row and do not inspect its
   child tasks.
3. `FastConformerDispatchStage` validates envelope ownership and policy constraints, decodes only the bounded
   segments in that envelope on CPU, and immediately makes exactly one FastConformer call. If an upstream stage
   supplies `waveform` and `sample_rate`, it consumes those values directly and skips descriptor decoding.
4. `DispatchBatchUnpackStage` explicitly returns processed segments to ordinary task rows.
5. `GlobalAudioParentAssemblerStage` restores each original manifest row and joins segment transcripts in source
   order. Its acknowledged operation cache safely replays unacknowledged retries while suppressing already
   acknowledged and same-call duplicates.

The standard segment fields plus optional `waveform`/`sample_rate` form the integration seam for a separate generic
materializer. `AudioSegmentDecoder` remains the standalone fallback; neither side imports the other.

Run:

```bash
python tutorials/audio/global_fastconformer/main.py \
  --input-manifest input.jsonl \
  --output-manifest output.jsonl
```
