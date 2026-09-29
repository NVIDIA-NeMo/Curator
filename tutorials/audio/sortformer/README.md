# Whole-Recording Sortformer Diarization

Run NVIDIA Streaming Sortformer v2.1 over a NeMo-style JSONL manifest through
NeMo Curator's task-facing diarization stage and NeMo model adapter.

From the repository root:

```bash
uv sync --extra audio_cuda12

python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/sortformer \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl
```

Each input row must contain `audio_filepath`. The output preserves the row and
adds:

- `diar_segments`: ordered `{start, end, speaker}` dictionaries;
- `num_speakers`: number of distinct speaker labels;
- `additional_notes`: task-local diagnostic notes only when preparation fails.

The stage owns file-versus-waveform selection, in-memory mono conversion and
16 kHz resampling, task fields, errors, resume checks, and output ordering.
The default `NeMoSortformerAdapter` owns checkpoint resolution, worker-local
model state, provider-native file decoding, streaming parameters, precision,
and inference. Adapter items contain exactly one of `audio_filepath` or
`waveform` plus `sample_rate`. Change provider options under `adapter_kwargs`;
change GPU allocation and the backend candidate window with `resources` and
`batch_size`.

`batch_size` groups independent recordings. Do not pre-split a recording to
make a larger batch: Sortformer speaker clustering and speaker identity depend
on whole-recording context. The stage may duration-sort complete recordings
for inference and scatters results back to manifest order.

The default checkpoint is
`nvidia/diar_streaming_sortformer_4spk-v2.1`. It supports up to four speakers
and is downloaded on first use.
