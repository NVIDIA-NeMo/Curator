# Review lenses - recurring themes in Curator audio PRs

Apply each lens to the diff and cite exact `path:line` evidence. These extend
the always-on rules in `.cursor/rules/` and the audio guide at
`nemo_curator/stages/audio/README.md`; when a change violates a rule, link it by
name.

## Stage contracts and reuse

- An audio stage subclasses `ProcessingStage[InputTask, OutputTask]` with
  explicit `inputs()` / `outputs()`, a meaningful `name`, and appropriate
  `resources` / `batch_size` (set as class attributes / dataclass fields, never
  as the read-only `_name` / `_resources` / `_batch_size` properties).
- GPU/model work (ASR, VAD, diarization, tagging inference) belongs in
  `process_batch()` with a real `batch_size`; `__init__` stays lightweight (it
  runs on the driver).
- Reuse existing audio readers, manifest writers, and shared model abstractions
  instead of bespoke orchestration. Model a tarred/sharded reader as a
  `CompositeStage` = a discovery stage plus a shard-reader fan-out.
- If `inputs()` does not declare a required key (e.g. `waveform_key`,
  `sample_rate_key`), runtime validation will not check it, and a direct
  `task.data[key]` access then raises an opaque `KeyError` instead of a clear
  validation error. Declare required keys.
- Fan-out shard readers need deterministic child IDs and must propagate
  `_metadata` / `_stage_perf`. First-stage shard discovery from an empty input
  should constrain placement (e.g. one worker per node) so discovery is not
  duplicated across every worker.

## Setup / teardown lifecycle and audio dependencies

- `setup_on_node()` = once-per-node prep (model download / shared cache).
  `setup()` = per-worker init (model load, GPU allocation). `teardown()` frees
  GPU memory. Don't conflate them.
- Import heavy/optional audio deps (vLLM, NeMo, model-specific utils,
  soundfile/lhotse, fasttext) lazily inside `setup()` or behind a guarded
  import, never unconditionally at module top level, when they ship only with an
  optional extra. A top-level `import <optional_pkg>` breaks
  `import nemo_curator.stages.audio...` on any install without that extra
  (including CPU-only / Mac/ARM).
- Declare the full audio runtime dependency set in the appropriate
  `pyproject.toml` optional extra (e.g. an audio CUDA extra); add an import
  smoke test under that extra. Don't rely on an external Docker image to supply
  dependencies a reusable audio stage needs.
- Avoid hard version pins (`pkg==x.y.z`) and global overrides for libraries
  shared across modalities (`transformers`, `huggingface-hub`, `accelerate`);
  prefer compatible ranges so other extras don't break.

## Secret-safe logging

- Never log a full resolved config; audio tutorials commonly accept an HF token
  / cloud credentials. Redact secret-like keys before logging (token, password,
  secret, credentials, `*_token`/`*_secret`), or log only a non-secret subset,
  and add a test asserting the secret value never appears in logged output.

## Waveform/tensor memory and manifest serialization

- Drop waveform arrays / audio tensors from `task.data` once a stage no longer
  needs them. A leaked numpy `ndarray` or torch tensor that reaches a JSON
  manifest writer raises `TypeError: Object of type ndarray is not JSON
  serializable` and can fail the entire shard/batch, or cause OOM at scale.
  Strip by waveform key plus a duck-typed guard (`.shape`/`.dtype`), and wrap
  `json.dumps` so a stray non-serializable value reports the offending key.
- Guard audio I/O edge cases: `tarfile.extractfile()` returns `None` for
  non-regular members even after `isfile()` - check before `.read()`. Handle
  resampling / channel-count / dtype mismatches explicitly.

## Tarred / sharded audio I/O

- Keep infrastructure assumptions (specific object-store URLs, auth env vars,
  cluster-local paths) out of the generic reader/writer. Make remote access
  pluggable: accept local paths, ordinary cloud URIs, or a user-provided
  opener/resolver, and isolate any environment-specific support behind a small,
  documented, optional component. For piped/streamed downloads use
  failure-aware flags + retries so truncated streams fail loudly.
- Manifest/shard writers that loop per task inside `process_batch()` reopen the
  output file repeatedly; group by shard and open each output once per batch.
  Preserve `.done`/checkpoint markers so runs are resumable.

## Streaming, throughput, benchmarking

- Streaming execution with per-stage resource configuration is the default
  model for scalable audio pipelines (see `executors.mdc`).
- Emit stage metrics through the supported metrics path (`metrics/`), JSON-
  shaped. For multi-node audio runs include worker/node/GPU identity and latency
  distributions (p50/p95/p99) plus audio-seconds-per-wall-second, not just
  totals/averages.
- Standalone audio benchmark scripts belong in the `benchmarking/` flow
  (`AUDIO_PROFILING.md`, `ALM_BENCHMARK.md`) with config entries and comparable
  parameters - not freestanding.

## Tutorials and docs

- Audio tutorials (`tutorials/audio/`) must be runnable, minimal, and use public
  APIs only - no private APIs, no machine-local absolute paths, no committed
  noisy notebook output, no executor mismatch between a notebook and its CLI.
  Prefer `uv` for environment setup if the tutorial standard does.
- Don't duplicate pipeline-construction logic between a tutorial entrypoint and
  an `examples/` script; extract one shared builder both call.
- Every config key documented for users must actually reach a stage; validate
  config-to-stage mapping and fail (or warn) on unused keys.
- Don't commit generated analysis/scratch docs into the source tree.

## Tests and standards (CONTRIBUTING + coding-standards)

- Source changes need accompanying tests; `tests/stages/audio/` mirrors
  `nemo_curator/stages/audio/`. The project enforces 80% coverage under
  `nemo_curator/`. GPU-only audio tests use `@pytest.mark.gpu`; reuse fixtures
  under `tests/fixtures/audio/` and sample data under `tests/data/audio/`.
- Every non-empty Python file carries the NVIDIA Apache-2.0 copyright header.
- Ruff clean at 119-char lines; type annotations on functions; loguru for logs.
- Every commit is DCO signed-off.

## PR size and reviewability

- Large multi-stage audio PRs are hard to review and defend. Prefer splitting by
  stage / module / tutorial / benchmark (e.g. reader+writer plumbing, then
  inference stages, then postprocessing, then tutorial/config). If a single PR
  must remain, organize it as if sliced: no duplicated builder logic, clear
  module ownership, a reviewable commit sequence.

## Smaller recurring nits

- Missing copyright headers on new files.
- Top-level imports that should be lazy/guarded (very common on audio model
  modules).
- Long inline prompt/config strings that should be module constants.
- Dataclass fields validated in `__post_init__` that could just be typed.
- Dead conditionals / redundant truthiness checks.
- Pytests added for tutorials that don't warrant them.
- Debug-only knobs (e.g. `max_utterances_per_shard`) presented as features -
  document them as debug throttles.
- `trust_remote_code=True` hardcoded on a model load - expose as a parameter so
  security-conscious users can opt out.

## Severity mapping

| Severity | Meaning |
|----------|---------|
| P0 | Merge blocker: crash/data-loss on a valid audio config, secret leaked to logs, import breaks a supported install |
| P1 | Fix before merge: undeclared audio deps, missing required tests/coverage, env-specific I/O in reusable reader, PR too large |
| P2 | Should fix: duplicated pipeline construction, inefficient writer batching, incomplete throughput metrics, documented-but-unwired config |
| P3 | Nice to have: temporary model/compat workaround needing test+comment, debug knob docs |
