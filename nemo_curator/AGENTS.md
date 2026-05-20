# Steward: Pipeline & Stage Contract

The `Task` / `ProcessingStage` / `Pipeline` triad is the public ABI of
NeMo Curator. Every modality stage, every backend adapter, and every
tutorial depends on it. Breaks here cascade to everything downstream.

Related docs:

- root [AGENTS.md](../AGENTS.md)
- [api-design.md](../api-design.md) — the canonical design statement
- `.cursor/rules/pipeline-structure.mdc`,
  `processing-stage-patterns.mdc`, `task-patterns.mdc`,
  `composite-stage-patterns.mdc`, `resources-configuration.mdc`
- `fern/` pipeline/stage/task concept pages

## Point Of View

The framework's spine — the contract that lets users define a pipeline
once and run it across Xenna, Ray Data, and Ray Actor Pool. Defends
abstractions against backend leakage, modality leakage, and convenience
shortcuts that quietly break the parity promise.

## Protect

- **Public ABI surfaces** (changes here are Stop-And-Ask):
  - `ProcessingStage[X, Y]` in `stages/base.py`: `name`, `resources`,
    `batch_size` (class attribute; the read-side `_batch_size` property
    is what backends consume), `runtime_env`, `process`, `inputs`,
    `outputs`, `setup`, `setup_on_node`, `teardown`, `with_` builders,
    `xenna_stage_spec`, `ray_stage_spec`.
  - `CompositeStage` composition contract.
  - `Task[T]` in `tasks/tasks.py`: `task_id`, `dataset_name`, `data`,
    `_stage_perf`, `_metadata`, `_uuid`, `num_items`, `validate`.
  - Modality tasks: `DocumentBatch`, `ImageBatch`, `VideoTask`,
    `AudioTask`, `FileGroupTask`, `InterleavedBatch`.
  - `Pipeline` in `pipeline/pipeline.py`: `add_stage`, `run`,
    `describe`; pipeline-level `config` semantics.
  - `Resources` in `stages/resources.py`: `cpus`, `gpu_memory_gb`,
    `gpus`. The order and meaning of these fields is part of the
    contract; adding fields requires Stop-And-Ask and updates to every
    modality task subclass and backend adapter.
  - `RayClient` in `core/client.py`: `start`, `stop`, lifecycle
    expectations.
  - `StageMeta` auto-registration: stage class names must remain unique
    across the registry.
- **Stage invariants**:
  - `process(task: X) -> Y | list[Y]` is the abstract signature in
    `stages/base.py`. Backend adapters and runtime helpers in
    `backends/utils.py` additionally tolerate `None` returns (used by
    filter stages); treat `None` as adapter-tolerated, not part of the
    declared ABI. If a future change widens the abstract signature to
    include `None`, propagate it here and in
    `.cursor/rules/processing-stage-patterns.mdc`.
  - All stages are fault-tolerant and retry-safe (Xenna preemption).
  - Stage code must not import RAPIDS/CUDA libs at module top level;
    lazy-import inside GPU methods.
  - Resource declarations must reflect actual usage; mis-declared
    resources break the scheduler.
- **Task invariants**:
  - `__post_init__` calls `validate()`; invalid tasks must raise, not
    silently degrade.
  - `_uuid` is auto-generated; do not set externally.
  - `num_items` reflects payload, not bytes.
- **Pipeline invariants**:
  - Adjacent stages' `outputs` ⊆ next stage's `inputs` (type-check
    happens at `add_stage` time or `run` time depending on stage
    declaration).
  - Pipeline is backend-agnostic; running it on a different executor
    must not change the user-visible result modulo documented
    nondeterminism (ordering of unordered outputs).

## Contract Checklist

When this domain changes, check:

- `stages/base.py`, `tasks/tasks.py`, `pipeline/pipeline.py`,
  `pipeline/workflow.py`, `backends/base.py`, `stages/resources.py`
- All three backend adapters import / wrap `ProcessingStage` correctly:
  `backends/xenna/adapter.py`, `backends/ray_data/adapter.py`,
  `backends/ray_actor_pool/adapter.py`
- `api-design.md` — keep it aligned with reality
- `.cursor/rules/pipeline-structure.mdc`,
  `processing-stage-patterns.mdc`, `task-patterns.mdc`,
  `composite-stage-patterns.mdc`, `resources-configuration.mdc`
- `.github/copilot-instructions.md` — the API Design Architecture
  section
- `fern/` concept pages (pipeline, stages, tasks, executors,
  resources) — list under **Own**
- `tests/stages/test_base.py`, `tests/tasks/`, `tests/pipelines/`,
  `tests/backends/` parity tests
- `tutorials/quickstart.py`
- `CHANGELOG.md`

For ABI changes include a parity matrix (Public API / Xenna / Ray Data
/ Ray Actor Pool / Docs / Tutorials / Tests / Cursor+Copilot rules).

## Advocate

- Stronger type-checking at `add_stage` time so pipeline composition
  errors surface before `run()`.
- Better diagnostics when a stage's declared `inputs`/`outputs` do not
  match what `process()` actually returns at runtime.
- Clear fault-tolerance contract docs: what each stage owes Xenna on
  preemption (idempotency, partial-state cleanup).
- Lower-friction extension paths for users who want custom stages or
  custom tasks without forking the framework.
- Performance counters on every stage by default
  (`StagePerfStats` is the hook).

## Serve Peers

- **To modality stewards** (text, image, audio, video, synthetic): give
  them a stable, well-documented base class with examples of correct
  resource declaration, fault-tolerance patterns, and composite stage
  composition.
- **To backends steward**: keep `ProcessingStage` and `Task` shaped so
  adapters can wrap them without reaching into private attributes.
  Surface any backend-relevant hook (`xenna_stage_spec`,
  `ray_stage_spec`, `runtime_env`) explicitly.
- **To tests steward**: provide pipeline test fixtures and a stable
  contract for `validate()` so test stages don't drift from real ones.
- **To docs steward**: keep `api-design.md` in sync with code so the
  Fern pipeline/stage/task pages have a single source of truth.
- **To extension authors**: keep `StageMeta` registry semantics
  documented.

## Do Not

- Add backend-specific code paths inside `ProcessingStage` subclasses.
  Differences belong in `nemo_curator/backends/<name>/`.
- Import `cudf`, `cupy`, `cuml`, or RAPIDS libs at module top level
  anywhere in `nemo_curator/`.
- Mutate `_uuid` or set `_stage_perf` externally.
- Add fields to `Task` or `Resources` without updating all modality
  task subclasses, all backend adapters, and the relevant Fern concept
  pages.
- Silently change `process()` return semantics (list vs single vs
  None) — that is a public-ABI change.
- Wrap `Pipeline.run()` in a way that bypasses executor selection
  logic.
- Add a config knob to `Pipeline.config` without documenting which
  executors honor it.

## Own

**Code surfaces**:

- `nemo_curator/stages/base.py`
- `nemo_curator/tasks/` (all task types)
- `nemo_curator/pipeline/`
- `nemo_curator/core/client.py`
- `nemo_curator/stages/resources.py`
- `nemo_curator/stages/function_decorators.py`,
  `client_partitioning.py`, `file_partitioning.py`

**Tests**:

- `tests/tasks/`
- `tests/pipelines/`
- `tests/backends/test_integration.py` (exercises the ABI today;
  consider adding dedicated `tests/stages/test_base.py` coverage —
  open advocacy)

**Docs (autopilot audit surface — keep this list current)**:

- Any `fern/` page under the pipeline / stages / tasks / executors /
  resources concept sections (canonical paths to be confirmed in the
  next docs autopilot pass and pinned here)
- `api-design.md`
- Root `README.md` quickstart section

**Agent-facing artifacts**:

- `.cursor/rules/pipeline-structure.mdc`
- `.cursor/rules/processing-stage-patterns.mdc`
- `.cursor/rules/task-patterns.mdc`
- `.cursor/rules/composite-stage-patterns.mdc`
- `.cursor/rules/resources-configuration.mdc`
- `.cursor/rules/executors.mdc`
- `.cursor/rules/coding-standards.mdc`
- The "API Design Architecture" / "Core Components" sections of
  `.github/copilot-instructions.md`

**CODEOWNERS routing**: default `@NVIDIA-NeMo/curator_reviewers`. Backends
overlap: `@oyilmaz-nvidia @praateekmahajan @abhinavg4 @ayushdg`.
