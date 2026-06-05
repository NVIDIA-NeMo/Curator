# Review lenses - recurring themes in Curator PRs

Apply each lens to the diff and cite exact `path:line` evidence. These extend
the always-on rules in `.cursor/rules/`; when a change violates a rule, link the
rule by name. Most points are modality-agnostic; audio examples are
illustrative.

## Stage contracts and reuse

- A stage subclasses `ProcessingStage[InputTask, OutputTask]` with explicit
  `inputs()` / `outputs()`, a meaningful `name`, and appropriate `resources` /
  `batch_size` (set as class attributes / dataclass fields, never as the
  read-only `_name` / `_resources` / `_batch_size` properties).
- GPU or vectorizable work belongs in `process_batch()` with a real
  `batch_size`; `__init__` stays lightweight (it runs on the driver).
- Reuse existing readers, writers, and shared model abstractions instead of
  bespoke orchestration. Use `CompositeStage` to present several internal stages
  as one user-facing stage.
- If `inputs()` does not declare a required key, runtime validation will not
  check it, and a direct `task.data[key]` access then raises an opaque
  `KeyError` instead of a clear validation error. Declare required keys.
- Fan-out stages (one task -> many) need deterministic child IDs and propagated
  `_metadata` / `_stage_perf`. First-stage discovery from an empty input should
  constrain placement so work is not duplicated across every worker.

## Setup / teardown lifecycle and dependencies

- `setup_on_node()` = once-per-node prep (e.g. model downloads / shared cache).
  `setup()` = per-worker init (model load, GPU allocation). `teardown()` frees
  resources. Don't conflate them.
- Import heavy or optional dependencies lazily inside `setup()` (or behind a
  guarded import), never unconditionally at module top level, when the dependency
  ships only with an optional extra. A top-level `import <optional_pkg>` breaks
  `import nemo_curator...` on any install without that extra.
- Declare the full runtime dependency set in the appropriate `pyproject.toml`
  optional extra; add an import smoke test under that extra. Don't rely on an
  external Docker image to supply dependencies a reusable stage needs.
- Avoid hard version pins (`pkg==x.y.z`) and global overrides for libraries
  shared across modalities (e.g. `transformers`, `huggingface-hub`,
  `accelerate`); prefer compatible ranges so other extras don't break.

## Secret-safe logging

- Never log a full resolved config object; it can contain tokens / credentials
  (e.g. an HF token, access keys). Redact secret-like keys before logging, or
  log only a non-secret subset, and add a test asserting the secret value never
  appears in logged output.

## Memory and serialization of large payloads

- Drop large in-memory payloads (waveform arrays, image/video tensors) from
  `task.data` once a stage no longer needs them. Leaked arrays reach downstream
  JSON writers and raise `TypeError: Object of type ndarray is not JSON
  serializable` (which can fail an entire batch) or cause memory blowups. Strip
  by key plus a duck-typed guard, and wrap `json.dumps` so a stray
  non-serializable value reports the offending key instead of crashing.
- Guard external I/O edge cases (e.g. `tarfile.extractfile()` returns `None` for
  non-regular members even after `isfile()`; check before `.read()`).

## Streaming, performance, benchmarking

- Streaming execution with per-stage resource configuration is the default model
  for scalable pipelines (see `executors.mdc`).
- Emit stage metrics through the supported metrics path; keep records
  JSON-shaped. For multi-node analysis, include worker/node/GPU identity and
  latency distributions (p50/p95/p99), not just totals/averages.
- Writers that loop per task inside `process_batch()` reopen output files
  repeatedly; group by output target and open once per batch.
- Standalone benchmark scripts belong in the `benchmarking/` flow with config
  entries and comparable parameters - not freestanding.

## Environment-specific code in reusable paths

- Keep infrastructure assumptions (specific object-store URLs, auth env vars,
  cluster-specific paths) out of generic reader/writer code. Make remote access
  pluggable: accept local paths, ordinary cloud URIs, or a user-provided
  opener/resolver, and isolate any environment-specific support behind a small,
  documented, optional component.

## Tutorials and docs

- Tutorials must be runnable, minimal, and use public APIs only - no private
  APIs, no machine-local absolute paths, no committed noisy notebook output, no
  executor mismatch between a notebook and its CLI. Prefer `uv` for environment
  setup if the tutorial standard does.
- Don't duplicate pipeline-construction logic between a tutorial entrypoint and
  an `examples/` script; extract one shared builder both call.
- Every config key documented for users must actually reach a stage; validate
  config-to-stage mapping and fail (or warn) on unused keys.
- Don't commit generated analysis/scratch docs into the source tree.

## Tests and standards (CONTRIBUTING + coding-standards)

- Source changes need accompanying tests; `tests/` mirrors `nemo_curator/`. The
  project enforces 80% coverage under `nemo_curator/`. GPU-only tests use
  `@pytest.mark.gpu`.
- Every non-empty Python file carries the NVIDIA Apache-2.0 copyright header.
- Ruff clean at 119-char lines; type annotations on functions; loguru for logs.
- Every commit is DCO signed-off.

## PR size and reviewability

- Large multi-purpose PRs are hard to review and to defend. Prefer splitting by
  stage / module / tutorial / benchmark. If a single PR must remain, organize it
  as if sliced: no duplicated builder logic, clear module ownership, and a
  reviewable commit sequence.

## Smaller recurring nits

- Missing copyright headers on new files.
- Top-level imports that should be lazy/guarded.
- Long inline strings (prompts, configs) that should be module constants.
- Dataclass fields validated in `__post_init__` that could just be typed.
- Dead conditionals / redundant truthiness checks.
- Tests added for tutorials that don't warrant them.
- Debug-only knobs presented as features - document them as debug throttles.
- `trust_remote_code=True` (or similar) hardcoded - expose as a parameter so
  security-conscious users can opt out.

## Severity mapping

| Severity | Meaning |
|----------|---------|
| P0 | Merge blocker: crash/data-loss on a valid config, secret leaked to logs, import breaks a supported install |
| P1 | Fix before merge: undeclared deps, missing required tests/coverage, env-specific code in reusable path, PR too large |
| P2 | Should fix: duplicated logic, inefficient batching, incomplete metrics, documented-but-unwired config |
| P3 | Nice to have: temporary workaround needing test+comment, debug knob docs |
