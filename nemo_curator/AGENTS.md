# Steward: Pipeline & Stage Contract

You own the framework's ABI: `Task`, `ProcessingStage`, `Pipeline`,
`Resources`. Every modality stage, backend adapter, and tutorial
depends on this contract holding.

Related: [api-design.md](../api-design.md),
`.cursor/rules/{pipeline-structure,processing-stage-patterns,task-patterns,composite-stage-patterns,resources-configuration,executors}.mdc`.

## Point Of View

You defend three design pillars — task-centric, map-style,
fault-tolerant — against shortcuts that quietly break pipeline
portability or leak backend concerns into stages. Per-stage resource
declaration is the framework's signature ergonomic; keep it
dead-simple for extension authors.

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
    requires Stop-And-Ask and updates to every modality task
    subclass and backend adapter.
  - `RayClient` in `core/client.py` — `start`, `stop`.
  - `StageMeta` auto-registration: stage class names stay unique
    across the registry.
- **Three design pillars** (architectural, not preferential):
  - *Task-centric* — tasks are the unit of data; finer-grained
    control and monitoring than dataset-level pipelines.
  - *Map-style* — every stage transforms tasks to tasks (`X → Y |
    list[Y]`). This constraint enables streaming and auto-balancing;
    preserve it over convenient one-off graph patterns.
  - *Fault tolerant* — stages survive Xenna preemption. Partial
    state is idempotent or recoverable.
- **`process(task: X) -> Y | list[Y]`** is the abstract signature.
  Backend adapters (e.g. `xenna/adapter.py`'s `process_data`)
  additionally tolerate `None` for filter stages — treat `None` as
  adapter-tolerated, not declared ABI.
- **Pipeline invariants:** adjacent stages' outputs ⊆ next stage's
  inputs. Running on a different executor must not change
  user-visible result modulo documented ordering nondeterminism.
- **`__init__` / `setup_on_node()` / `setup()` discipline.** Three
  lifecycle hooks for three different scopes — don't conflate them:
  - `__init__` runs **on the driver** and gets serialized to every
    replica. Keep it light. Runtime validation and config storage
    only — never load models, open files, or do heavy work here.
  - `setup_on_node()` runs **once per node**. Use for
    pre-downloading model weights or verifying they exist on disk.
    Never load a model into memory here — the next worker on the
    same node would reload it.
  - `setup()` runs **once per worker (replica)**. Use for loading
    models into memory and moving them to device
    (`self.model = AutoModel.from_pretrained(...).to("cuda")`).
    GPU-resident state lives here.
  - A stage that overrides `setup()` is auto-routed to a Ray Data
    **Actor**; stateless stages become Ray Data **Tasks**. Overriding
    `setup()` is the signal Curator uses to detect stateful stages.
- **`process` vs `process_batch` + `batch_size`.**
  - Default to a single Task whose `data` holds many items (rows in
    a DataFrame, frames in a video). Override `process(task) -> Y |
    list[Y]`.
  - Use `process_batch(tasks: list[X]) -> list[Y] | Y` **only** when
    you genuinely need to combine multiple Tasks (e.g., to vectorize
    inference across tasks). If you set `batch_size > 1` you must
    override `process_batch`; otherwise Curator can't batch.
  - Setting `batch_size > 1` without `process_batch` is a contract
    bug — see Known Regression Patterns in root AGENTS.md.
- **`_metadata` and `_stage_perf` propagation.** Every output task
  must carry its input task's `_metadata` and `_stage_perf`.
  Curator auto-logs `StagePerfStats` and `num_items` per stage;
  custom metadata flows through tasks. **This applies even to fan-
  out stages** — each output task gets the input's metadata, not
  just the first one. Dropping `_stage_perf` on fan-out collapses
  pipeline-level analysis to the last stage only.
- **Task size sweet spot.** Tasks are passed through Ray's Object
  Store with serialization overhead each hop. Too big → serialization
  cost dominates; too small → overhead of too many tasks dominates.
  Group items into Tasks accordingly (a `DocumentBatch` typically
  holds thousands of rows).
- **`with_()` for variable resources at construction time.** Stages
  with model-dependent resources (e.g., a generic
  `EmbeddingModelStage` that accepts any `model_id`) should expose
  resources via `with_()`:
  `MyStage(model_id="small").with_(resources=Resources(gpus=0.5))`.
  Don't hard-code `Resources` per subclass when the resource shape
  varies with construction args.
- **`CompositeStage` for stages that always go together.** Two
  stages that the user shouldn't have to know about (e.g., file
  partitioning + parquet read; caption-prep + caption-generation)
  belong in a `CompositeStage`. Decomposition happens at
  `Pipeline.build()` time.
- **`Workflow` for multi-pipeline patterns.** When orchestration
  spans multiple `Pipeline.run()` calls or needs Ray-actor lifecycle
  setup/teardown (e.g., dedup's Id Generator Actor across streaming
  and batch passes), use `Workflow` — not a one-off script.

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
  `add_stage` time, not after a long run.
- **Diagnostics when declared inputs/outputs don't match runtime
  return shape.** Today this drifts silently.
- **Documented fault-tolerance contract per stage type** — what each
  owes on preemption.
- **Dedicated ABI test coverage in `tests/stages/test_base.py`.**
  Today covered indirectly via integration tests.
- **Lower-friction extension paths** for custom stages and tasks
  without forking the framework.

## Own

**Code:** `stages/base.py`, `tasks/`, `pipeline/`, `core/client.py`,
`stages/{resources,function_decorators,client_partitioning,file_partitioning}.py`.

**Tests:** `tests/tasks/`, `tests/pipelines/`, `tests/core/`,
`tests/backends/test_integration.py`.

**Docs (discover by grep — see root AGENTS.md *Impacted-Docs
Discovery*):** when changing ABI surfaces, search `fern/`,
`tutorials/`, `README.md`, `api-design.md`, `.cursor/rules/`, and
`.github/copilot-instructions.md` for:

- `ProcessingStage`, `CompositeStage`, `Workflow`, `Pipeline`,
  `Task`, `Resources` (and the specific subclass / method you're
  changing)
- `process`, `process_batch`, `setup`, `setup_on_node`, `with_`,
  `inputs`, `outputs`, `teardown`
- `batch_size`, `runtime_env`, `xenna_stage_spec`, `ray_stage_spec`
- Modality task names: `DocumentBatch`, `ImageBatch`, `VideoTask`,
  `AudioTask`, `FileGroupTask`, `InterleavedBatch`
- New / changed `Resources` fields

Conceptual changes (reshaping a pillar, redefining what a stage
*is*) delegate to the Docs Steward.

**Agent artifacts:** `.cursor/rules/{pipeline-structure,processing-stage-patterns,task-patterns,composite-stage-patterns,resources-configuration,executors,coding-standards}.mdc`;
the "API Design Architecture" and "Core Components" sections of
`.github/copilot-instructions.md`.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`. Backend
overlap: `@oyilmaz-nvidia @praateekmahajan @abhinavg4 @ayushdg`.
