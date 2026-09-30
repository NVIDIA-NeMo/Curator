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

For a converted TensorRT deployment, replace both `adapter_target` and the
entire native `adapter_kwargs` mapping. Native options such as `precision` and
`chunk_len` are not TensorRT adapter arguments:

```yaml
adapter_target: nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt.TensorRTSortformerAdapter
adapter_kwargs:
  engine_path: /models/sortformer.plan
  config_path: /models/sortformer.json
  runtime_module_path: /models/sortformer.sortformer_modules.py
  inference_batch_size: 1
```

The task and output schema stay the same.

Install the TensorRT-specific dependencies before building or running that
adapter:

```bash
uv sync --extra audio_tensorrt
```

Build the target-GPU bundle from a local `.nemo` checkpoint and the matching
trusted Riva runtime module:

```bash
python -m nemo_curator.models.audio.speaker_diarization.build_sortformer_tensorrt_engine \
  --nemo-model /models/sortformer.nemo \
  --runtime-module /opt/riva/backends/sortformer_modules.py \
  --output /models/sortformer.plan
```

The command validates a staged bundle before publishing the JSON completeness
marker last. It writes `/models/sortformer.plan`, `/models/sortformer.json`,
`/models/sortformer.mel_basis.npy`,
`/models/sortformer.sortformer_modules.py`, and an optional
`/models/sortformer.learnable_sil_emb.npy`. Point the three adapter settings at
the engine, JSON, and stem-scoped copied Python file. Build on the same GPU
architecture used for deployment; TensorRT engines are target-specific. The
runtime module is executable Python, so only use one from a trusted deployment
environment. Long same-rate file inputs use bounded waveform reads and STFT
calls; full-recording normalized features and output probabilities still grow
with recording duration. Long files that need resampling are rejected rather
than silently materialized as a second full waveform.
The schema-v2 JSON records the checkpoint's frontend normalization contract;
the adapter rejects older or ambiguous configs instead of sending
off-distribution features to the engine, so rebuild pre-schema-v2 bundles.

`batch_size` groups independent recordings. Do not pre-split a recording to
make a larger batch: Sortformer speaker clustering and speaker identity depend
on whole-recording context. The stage may duration-sort complete recordings
for inference and scatters results back to manifest order.

The default checkpoint is
`nvidia/diar_streaming_sortformer_4spk-v2.1`. It supports up to four speakers
and is downloaded on first use.
