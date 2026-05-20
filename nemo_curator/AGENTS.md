# Steward: Pipeline & Stage Contract

The `Task` / `ProcessingStage` / `Pipeline` triad is the public ABI of
NeMo Curator. Every modality stage, every backend adapter, and every
tutorial depends on it.

Related: root [AGENTS.md](../AGENTS.md),
[api-design.md](../api-design.md),
`.cursor/rules/{pipeline-structure,processing-stage-patterns,task-patterns,composite-stage-patterns,resources-configuration,executors}.mdc`.

## Point Of View

The framework's spine. Defends abstractions against backend leakage,
modality leakage, and convenience shortcuts that quietly break the
parity promise.

## Protect

- **Public ABI surfaces** (Stop-And-Ask to change):
  - `ProcessingStage[X, Y]` in `stages/base.py` — `name`,
    `resources`, `runtime_env`, `batch_size` (class attribute; the
    read-side `_batch_size` property is what backends consume),
    `process`, `inputs`, `outputs`, `setup`, `setup_on_node`,
    `teardown`, `with_` builders, `xenna_stage_spec`,
    `ray_stage_spec`.
  - `CompositeStage` composition contract.
  - `Task[T]` in `tasks/tasks.py` — `task_id`, `dataset_name`,
    `data`, `_stage_perf`, `_metadata`, `_uuid`, `num_items`,
    `validate`. Modality tasks: `DocumentBatch`, `ImageBatch`,
    `VideoTask`, `AudioTask`, `FileGroupTask`, `InterleavedBatch`.
  - `Pipeline` in `pipeline/pipeline.py` — `add_stage`, `run`,
    `describe`, `config` semantics.
  - `Resources` in `stages/resources.py` — `cpus`, `gpu_memory_gb`,
    `gpus`. The order and meaning are part of the contract; adding
    fields requires Stop-And-Ask and updates to every modality task
    subclass and backend adapter.
  - `RayClient` in `core/client.py` — `start`, `stop`.
  - `StageMeta` auto-registration: stage class names must stay unique
    across the registry.
- **Stage invariants:**
  - `process(task: X) -> Y | list[Y]` is the abstract signature.
    Backend adapters (e.g. `xenna/adapter.py`'s `process_data`,
    which returns `list[Task] | None`) additionally tolerate `None`
    for filter stages — treat `None` as adapter-tolerated, not
    declared ABI.
  - Fault-tolerant and retry-safe (Xenna preempts).
  - Resource declarations must reflect real usage — mis-declared
    resources break the scheduler.
- **Task invariants:** `__post_init__` calls `validate()`, which must
  raise on invalid data, not silently degrade. `_uuid` is
  auto-generated. `num_items` reflects payload, not bytes.
- **Pipeline invariants:** adjacent stages' outputs ⊆ next stage's
  inputs. Pipeline is backend-agnostic; running on a different
  executor must not change user-visible result modulo documented
  ordering nondeterminism.

## Contract Checklist

When this domain changes:

- `stages/base.py`, `tasks/tasks.py` (and the modality task files),
  `pipeline/pipeline.py`, `pipeline/workflow.py`, `backends/base.py`,
  `stages/resources.py`, `stages/function_decorators.py`,
  `stages/client_partitioning.py`, `stages/file_partitioning.py`
- All three backend adapters wrap the new contract:
  `backends/{xenna,ray_data,ray_actor_pool}/adapter.py`
- `api-design.md` — keep aligned with reality
- Cursor rules listed above; the "API Design Architecture" section of
  `.github/copilot-instructions.md`
- `fern/` concept pages (pipeline, stages, tasks, executors,
  resources)
- `tests/{tasks,pipelines,backends/test_integration.py}/`
- `tutorials/quickstart.py`
- `CHANGELOG.md`

ABI changes include a parity matrix (Public API / Xenna / Ray Data /
Ray Actor Pool / Docs / Tutorials / Tests / Cursor+Copilot rules).

## Advocate

- Stronger type-checking at `add_stage` time so pipeline composition
  errors surface before `run()`.
- Diagnostics when a stage's declared `inputs`/`outputs` don't match
  what `process()` actually returns at runtime.
- Documented fault-tolerance contract per stage type: what each owes
  Xenna on preemption (idempotency, partial-state cleanup).
- Dedicated `tests/stages/test_base.py` coverage of the ABI (today
  covered indirectly via `tests/backends/test_integration.py`).
- Lower-friction extension paths for custom stages / custom tasks
  without forking.

## Own

**Code:** `stages/base.py`, `tasks/`, `pipeline/`, `core/client.py`,
`stages/{resources,function_decorators,client_partitioning,file_partitioning}.py`.

**Tests:** `tests/tasks/`, `tests/pipelines/`, `tests/core/`,
`tests/backends/test_integration.py`.

**Docs (autopilot surface — to be pinned by Docs Steward):** `fern/`
pages under pipeline / stages / tasks / executors / resources
concepts; `api-design.md`; the quickstart section of `README.md`.

**Agent artifacts:** `.cursor/rules/{pipeline-structure,processing-stage-patterns,task-patterns,composite-stage-patterns,resources-configuration,executors,coding-standards}.mdc`;
the "API Design Architecture" and "Core Components" sections of
`.github/copilot-instructions.md`.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`. Backend
overlap: `@oyilmaz-nvidia @praateekmahajan @abhinavg4 @ayushdg`.
