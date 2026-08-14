# AGENTS.md — NeMo Curator

NeMo Curator is a scalable library for preparing multimodal datasets. Pipelines
are composed of `ProcessingStage` objects executed by a backend (Ray Data, Xenna,
or Slurm) over streams of `Task` objects.

## Core abstractions

| Abstraction | Location | Role |
|---|---|---|
| `ProcessingStage` | `nemo_curator/stages/base.py` | Unit of work: defines `process_batch()`, `resources`, and optional `setup()` for stateful stages |
| `Resources` | `nemo_curator/stages/resources.py` | Per-stage CPU/GPU requirements (`cpus`, `gpus`, `gpu_memory_gb`) |
| `Task` | `nemo_curator/tasks/` | Data item flowing through the pipeline |
| `Pipeline` | `nemo_curator/pipeline/pipeline.py` | Ordered sequence of stages executed by a backend |
| `RayClient` / `SlurmRayClient` | `nemo_curator/core/client.py` | Cluster connection and Ray init |

## Key rules

- **Optional extras**: feature families are behind extras in `pyproject.toml`.
  Do not make heavyweight dependencies unconditional.
- **Fern docs**: user-facing documentation lives in `fern/`, not `docs/`. Edit
  MDX files there; do not add docs to the `docs/` directory.
- **Avoid local narration**: comments should explain only non-obvious, durable
  constraints—not narrate the current task, test setup, or implementation.
- **Reuse before adding**: search Curator for existing implementations,
  utilities, and patterns before writing new code; reuse or extend them when
  they fit.
- **Tests**: prefer narrowest-scope tests (unit > integration > GPU). GPU tests
  must be registered separately.

## Backend-scoped guidance

| Backend | Reference |
|---|---|
| Ray Data — scheduler internals, log events, tuning knobs | [`nemo_curator/backends/ray_data/AGENTS.md`](nemo_curator/backends/ray_data/AGENTS.md) |
