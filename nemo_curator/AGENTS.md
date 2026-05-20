# Steward: Pipeline & Stage Contract

This domain exists because pipeline portability across executors is the
framework's value proposition. The `Task` / `ProcessingStage` /
`Pipeline` / `Resources` ABI is what lets a user write a curation
pipeline once and run it across Xenna, Ray Actor Pool, and Ray Data
without code changes. Every modality stage, every backend adapter,
every tutorial depends on this contract holding.

Related: root [AGENTS.md](../AGENTS.md),
[api-design.md](../api-design.md),
`.cursor/rules/{pipeline-structure,processing-stage-patterns,task-patterns,composite-stage-patterns,resources-configuration,executors}.mdc`.

## Point Of View

The framework's spine. Defends three design pillars — task-centric,
map-style, fault-tolerant — against abstractions that would leak
backend concerns into stages, modality concerns into the base, or
convenience shortcuts that quietly break portability. Per-stage
resource declaration is the framework's signature ergonomic, and it
must stay simple enough for extension authors to use correctly without
deep framework knowledge.

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
    `gpus`. Field order and meaning are contract; adding fields
    requires Stop-And-Ask and updates to every modality task subclass
    and backend adapter.
  - `RayClient` in `core/client.py` — `start`, `stop`.
  - `StageMeta` auto-registration: stage class names must stay unique
    across the registry.
- **Three design pillars** (architectural, not preferential):
  - *Task-centric* — tasks are the unit of data; finer-grained control
    and monitoring than dataset-level pipelines.
  - *Map-style* — every stage transforms tasks to tasks (`X → Y |
    list[Y]`). This constraint is what enables streaming and
    auto-balancing; preserving it is more important than supporting a
    convenient one-off graph pattern.
  - *Fault tolerant* — stages survive Xenna preemption. Partial state
    is idempotent or recoverable. Stages don't lose work silently.
- **Resource declaration as the framework's signature ergonomic.**
  Setting `Resources(cpus=0.5)` or `gpus=0.5` at the stage level
  must stay dead-simple. Extension authors who can dial resources
  per-stage can get heterogeneous CPU/GPU pipelines to perform; those
  who can't, can't.
- **`process(task: X) -> Y | list[Y]`** is the abstract signature.
  Backend adapters (e.g. `xenna/adapter.py`'s `process_data`)
  additionally tolerate `None` for filter stages — treat `None` as
  adapter-tolerated, not declared ABI.
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

- **Pre-flight pipeline validation.** Type errors should surface at
  `add_stage` time, not after a long run. Composition errors caught
  before `run()` save the most user pain.
- **Diagnostics when declared inputs/outputs don't match runtime
  return shape.** Today this drifts silently.
- **Documented fault-tolerance contract per stage type** — what each
  owes on preemption (idempotency, partial-state cleanup).
- **Dedicated ABI test coverage in `tests/stages/test_base.py`.**
  Today covered indirectly via integration tests.
- **Lower-friction extension paths** for custom stages and tasks
  without forking the framework.

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
