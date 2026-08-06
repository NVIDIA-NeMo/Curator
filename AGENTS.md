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
| `RayClient` / `SlurmRayClient` | `nemo_curator/core/client.py` | Cluster connection and Ray init (sets `object_store_memory`, `ray_temp_dir`) |
| `RayStageSpecKeys` | `nemo_curator/backends/utils.py` | Per-stage Ray Data overrides (`MIN_WORKERS`, `MAX_WORKERS`, `IS_ACTOR_STAGE`, `RAY_NUM_CPUS`, …) |

## Key rules

- **Actor vs Task** is decided by `RayDataStageAdapter` automatically: a stage is
  an Actor if it overrides `setup()` (has persistent state, e.g. model weights)
  or requests both GPU and CPU. Override via `stage.with_(ray_stage_spec={RayStageSpecKeys.IS_ACTOR_STAGE: True/False})`.
- **GPU stages** should always use `stage.with_(num_workers=N)` (fixed pool,
  `min_size=max_size=N`) — do not use `INITIAL_WORKERS` alone, it downscales
  immediately after the first input arrives.
- **Optional extras**: feature families are behind extras in `pyproject.toml`.
  Do not make heavyweight dependencies unconditional.
- **Fern docs**: user-facing documentation lives in `fern/`, not `docs/`. Edit
  MDX files there; do not add docs to the `docs/` directory.

## Backend-scoped guidance

For Ray Data scheduler internals, diagnostic log events, and tuning knobs:

→ [`nemo_curator/backends/ray_data/AGENTS.md`](nemo_curator/backends/ray_data/AGENTS.md)
