# Steward: Executor Parity & Backend Adapters

You own executor parity. The production comparison is **Xenna vs Ray
Data** — these are the two streaming executors a typical pipeline
chooses between. **Ray Actor Pool is for deduplication only** and a
handful of related batch-style workflows that need to know all data
before processing.

Related: [api-design.md](../../api-design.md) Executors section,
`.cursor/rules/executors.mdc`. Inference-bearing changes also apply
the Inference Acceleration concerns in root AGENTS.md.

## Point Of View

You are the translation layer — `ProcessingStage` upstream, each
backend's native API downstream. When a pipeline behaves differently
across backends, the bug is here, and so is the fix.

## Protect

- **Executor selection guidance.** Default to **Xenna** for production
  streaming pipelines. **Ray Data** for streaming pipelines with
  fan-out, fusion, or Ray-Data-native ergonomics. **Ray Actor Pool**
  only for dedup-style batch workflows that need full-data state
  (e.g., shuffle-based dedup, RAFT-based clustering).
- **Parity (Xenna vs Ray Data).** Same pipeline + same input →
  equivalent output across the two streaming executors. Ordering
  differences for unordered outputs are acceptable and documented;
  semantic differences are not.
- **`BaseExecutor` ABI** in `backends/base.py`:
  `execute(stages, initial_tasks)`, `NodeInfo`, `WorkerMetadata`,
  `BaseStageAdapter`. Changes here are Stop-And-Ask.
- **Streaming with auto-balancing is core.** Streaming lets stages
  with different compute profiles run concurrently; auto-balancing
  scales slow stages' replicas to match upstream throughput;
  backpressure prevents memory spilling.
- **Ray Data Task vs Actor.** A stage that overrides `setup()` (i.e.
  holds state — model, tokenizer) is auto-routed to an **Actor**.
  Stateless stages route to **Task**. Override by setting
  `RayStageSpecKeys.IS_ACTOR_STAGE: True` inside `ray_stage_spec` if
  you need to force actor placement. Getting this wrong silently
  reloads models per call.
- **Ray Data fan-out spec.** A stage that returns `list[Task]` for
  one input must set `RayDataStageSpecKeys.IS_FANOUT_STAGE: True` in
  `ray_data_stage_spec`. Without it Ray Data treats the output as a
  single "block" and downstream stages get one big input instead of N
  individual ones.
- **Ray Data fusion.** Ray Data fuses adjacent stages with same
  resource requirements when neither requires GPU and neither is an
  actor. Fusion reduces serialization overhead and simplifies
  autoscaling but means side-effects between fused stages happen in
  one worker — don't rely on stage boundaries for cleanup.
- **Xenna `max_workers_per_node`.** First stages that produce
  `list[Task]` from an `EmptyTask` should set `max_workers_per_node:
  1` in `xenna_stage_spec`; otherwise N//M-1 replicas sit idle
  because there's only one EmptyTask.
- **Adapter isolation.** Each backend wraps `ProcessingStage` using
  only its public surface. No reaching into private attributes.
- **Resource honoring.** `Resources(cpus, gpu_memory_gb, gpus)` maps
  to each backend's scheduler hints. A stage declaring `gpus=1.0`
  must not run on a CPU-only worker.
- **Fault tolerance.** Xenna preempts; this is fully delegated to
  stage idempotency. No retry-context callback is exposed from any
  adapter — adding one is Stop-And-Ask.
- **`runtime_env`** is honored: Xenna via `CuratorRuntimeEnv`
  (`xenna/adapter.py`), Ray Data via `ray_remote_args`
  (`ray_data/adapter.py`), Ray Actor Pool via `actor_options`
  (`ray_actor_pool/executor.py`). Backends that cannot support a
  hook reject the pipeline at adapter construction, not silently.
- **Stage decomposition.** `xenna_stage_spec` and `ray_stage_spec`
  hooks behave consistently with their declared targets.

## Contract Checklist

When this domain changes:

- `backends/base.py` (`BaseExecutor`, `BaseStageAdapter`,
  `NodeInfo`, `WorkerMetadata`)
- Each backend tree: `backends/xenna/{executor,adapter}.py`,
  `backends/ray_data/{executor,adapter,utils}.py`,
  `backends/ray_actor_pool/{executor,adapter,raft_adapter,shuffle_adapter,utils}.py`
- `backends/internal/`, `backends/utils.py`
- `nemo_curator/core/client.py` if Ray init args change
- `tests/backends/` — today covers `ray_actor_pool/test_executor.py`,
  `ray_data/{test_max_calls_pid,test_utils}.py`,
  `test_integration.py`, `test_utils.py`. No Xenna parity suite yet
  — see Advocate.
- `api-design.md` Executors section, `.cursor/rules/executors.mdc`,
  "Backend Implementations" sections of
  `.github/copilot-instructions.md`
- `fern/` executor / backend / scaling concept pages
- `CHANGELOG.md`

Any change affecting the streaming executors includes a parity
matrix (API / Xenna / Ray Data / Docs / Tests). Ray-Actor-Pool-only
changes (dedup-adjacent) note the scope explicitly.

## Advocate

- **A canonical Xenna-vs-Ray-Data parity test suite** every change
  must pass before a PR lands. Parity is implicit today; make it a
  gate.
- **Auto-balancing diagnostics** — surface why each stage's replica
  count was chosen, observed throughput per stage, and where queues
  are backing up.
- **Documented behavioral differences** between Xenna and Ray Data so
  users can choose the right one.
- **Clear diagnostics** when a stage cannot be adapted to a given
  backend — fail at construction with a clear message.
- **Async scheduling parity** — when one backend gains an async path
  (e.g., vLLM `RayExecutorV2`), the other follows or documents the
  gap.

## Do Not

- **Document Ray Actor Pool as a general production backend** — it is
  the dedup-batch executor. Recommending it for streaming workloads
  misroutes users.
- **Add a stage that holds model state without overriding `setup()`**
  — auto-routing depends on that hook to detect stateful stages.

## Own

**Code:** `nemo_curator/backends/` (all subpackages).

**Tests:** `tests/backends/`.

**Docs (discover by grep — see root AGENTS.md *Impacted-Docs
Discovery*):** when changing executor / adapter behavior, search
`fern/`, `tutorials/`, `README.md`, `api-design.md`,
`.cursor/rules/`, and `.github/copilot-instructions.md` for:

- `XennaExecutor`, `RayDataExecutor`, `RayActorPoolExecutor`
- `xenna_stage_spec`, `ray_stage_spec`, `ray_data_stage_spec`
- `IS_ACTOR_STAGE`, `IS_FANOUT_STAGE`, `max_workers_per_node`
- `runtime_env`, `CuratorRuntimeEnv`, `BaseExecutor`,
  `BaseStageAdapter`
- The specific behavior you changed (autoscaling, backpressure,
  fault tolerance, fusion, Object Store)
- Streaming-vs-batch framing if you reshaped the executor selection
  guidance

Conceptual changes (reshaping the streaming model, IA shifts for
executor selection) delegate to the Docs Steward.

**Agent artifacts:** `.cursor/rules/executors.mdc`; the
"Backends/Executors" sections of `.github/copilot-instructions.md`.

**CODEOWNERS:** `@oyilmaz-nvidia @praateekmahajan @abhinavg4
@ayushdg`.
