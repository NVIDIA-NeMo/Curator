# AGENTS.md — NeMo Curator

NeMo Curator is a scalable library for preparing multimodal datasets. Pipelines
are composed of `ProcessingStage` objects executed by a backend (Ray Data, Xenna, Ray Actor Pool) over streams of `Task` objects.

## Core abstractions

| Abstraction | Location | Role |
|---|---|---|
| `ProcessingStage` | `nemo_curator/stages/base.py` | Unit of work: defines `process()` or `process_batch()`, `resources`, and optional `setup()` for stateful stages |
| `Resources` | `nemo_curator/stages/resources.py` | Per-stage CPU/GPU requirements (`cpus`, `gpus`, `gpu_memory_gb`) |
| `Task` | `nemo_curator/tasks/` | Data item flowing through the pipeline |
| `Pipeline` | `nemo_curator/pipeline/pipeline.py` | Ordered sequence of stages executed by a backend |
| `RayClient` / `SlurmRayClient` | `nemo_curator/core/client.py` | Cluster connection and Ray init |

## Key rules

- **Development environment**: run Python tools and tests in an existing virtual
  environment where NeMo Curator is installed, or in the project Docker image.
  Do not create or sync another environment unless explicitly requested.
- **Optional extras**: feature families are behind extras in `pyproject.toml`.
  Do not make heavyweight dependencies unconditional.
- **Fern docs**: user-facing documentation lives in `fern/`, not `docs/`. Edit
  MDX files there; do not add docs to the `docs/` directory.
- **Avoid local narration**: comments should explain only non-obvious, durable
  constraints—not narrate the current task, test setup, or implementation.
- **Reuse before adding**: search Curator for existing implementations,
  utilities, and patterns before writing new code; reuse or extend them when
  they fit.
- **Tests**: keep unit-test files in a one-to-one mapping with source files so
  coverage is easy to trace. Test user-visible behavior and real integration
  boundaries; avoid tests that merely enumerate defaults or mock away the
  dependency being validated. Prefer the narrowest scope that provides
  confidence: unit, then integration, then GPU. GPU tests must be registered
  separately.
- **Commits and PRs**: use Conventional Commits style for commit messages and
  PR titles, and run the repository's pre-commit hooks before submitting.

## Backend-scoped guidance

| Backend | Reference |
|---|---|
| Ray Data — scheduler internals, log events, tuning knobs | [`nemo_curator/backends/ray_data/AGENTS.md`](nemo_curator/backends/ray_data/AGENTS.md) |
