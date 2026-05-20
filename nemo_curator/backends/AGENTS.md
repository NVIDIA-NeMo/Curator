# Steward: Executor Parity & Backend Adapters

This domain exists because pipeline portability — the same `Pipeline`
running unchanged across Xenna, Ray Actor Pool, and Ray Data — is the
framework's core promise. When a pipeline behaves differently on
different executors, this is where the bug is, and where the fix
belongs.

Related: root [AGENTS.md](../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../AGENTS.md),
[api-design.md](../../api-design.md) Executors section,
`.cursor/rules/executors.mdc`.

## Point Of View

The translation layer. Speaks `ProcessingStage` upstream and each
backend's native API downstream. The job is not just "run pipelines"
— it's *streaming with auto-balancing across heterogeneous stages*:
backpressure that prevents memory spilling, replica counts that match
each stage's throughput so the slowest stage doesn't starve the rest,
and GPU workers kept busy at high utilization through the run.

## Protect

- **Parity.** Same pipeline + same input → equivalent output across
  Xenna, Ray Actor Pool, Ray Data. Ordering differences for unordered
  outputs are acceptable and must be documented; semantic differences
  are not.
- **`BaseExecutor` ABI** in `backends/base.py`:
  `execute(stages, initial_tasks)`, `NodeInfo`, `WorkerMetadata`,
  `BaseStageAdapter`. Changes here are Stop-And-Ask.
- **Streaming mode is core, not optional.** Batch mode forces video
  download → cutscene → extraction → transcode to serialize per video.
  Streaming lets stages with different compute profiles run
  concurrently, keeping GPU compute and decode units saturated.
  Backends must support streaming first; batch is the degraded path.
- **Auto-balancing across heterogeneous stages.** When a slow stage
  (e.g., VLM captioning) backs up a fast stage (e.g., download), the
  backend must scale replicas of the slow stage to match upstream
  throughput. Backpressure prevents memory spilling.
- **Adapter isolation.** Each backend wraps `ProcessingStage` using
  only its public surface. No reaching into private attributes.
- **Resource honoring.** `Resources(cpus, gpu_memory_gb, gpus)`
  must map to each backend's scheduler hints. A stage declaring
  `gpus=1.0` must not run on a CPU-only worker.
- **Fault tolerance.** Xenna preempts. Today this is fully delegated
  to stage idempotency — no retry-context callback is exposed from
  any adapter. Adding such an API is Stop-And-Ask.
- **`runtime_env`** is honored by every backend that supports it:
  Xenna via `CuratorRuntimeEnv` (`xenna/adapter.py`), Ray Data via
  `ray_remote_args` (`ray_data/adapter.py`), Ray Actor Pool via
  `actor_options` (`ray_actor_pool/executor.py`). Backends that
  cannot support a hook must reject the pipeline at adapter
  construction, not silently ignore it.
- **Stage decomposition.** `xenna_stage_spec` and `ray_stage_spec`
  hooks behave consistently with their declared targets.
- **Inference-bearing pipelines** route through the Inference
  Acceleration Steward (root AGENTS.md). Backends own the integration
  surface (`runtime_env`, model-server deps, async scheduling) but
  defer model-serving choices to that cross-cutting steward.

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
  — see Advocate. Executor-affecting changes add coverage to
  whichever adapter they touch.
- `api-design.md` Executors section, `.cursor/rules/executors.mdc`,
  "Backend Implementations" sections of
  `.github/copilot-instructions.md` and parent steward
- `fern/` executor / backend concept pages, scaling concepts
- `CHANGELOG.md`

Any executor-affecting change includes a parity matrix (API / Xenna /
Ray Data / Ray Actor Pool / Docs / Tests).

## Advocate

- **A canonical parity test suite** every executor must pass before a
  PR can land. Parity is implicit today; make it a gate.
- **Auto-balancing diagnostics** — surface why a stage's replica
  count was chosen, what its observed throughput is, and where queues
  are backing up. Today operators reason about this from external
  metrics; the framework should expose it.
- **Documented behavioral differences across executors** (autoscaling,
  streaming vs batch semantics) so users can choose the right backend.
- **Clear diagnostics when a stage cannot be adapted** to a given
  backend — fail at construction with a clear message, not deep stack
  into adapter internals.
- **A "minimum viable adapter" recipe** for extension authors writing
  a fourth backend.
- **Async scheduling parity** — when one backend gains an async path
  (e.g., vLLM `RayExecutorV2` integration), others should follow or
  document the gap.

## Own

**Code:** `nemo_curator/backends/` (all subpackages).

**Tests:** `tests/backends/`.

**Docs (autopilot surface — to be pinned by Docs Steward):** `fern/`
pages on executors, backends, Ray cluster setup, executor selection,
autoscaling, streaming concepts.

**Agent artifacts:** `.cursor/rules/executors.mdc`; the
"Backends/Executors" sections of `.github/copilot-instructions.md`.

**CODEOWNERS:** `@oyilmaz-nvidia @praateekmahajan @abhinavg4
@ayushdg`.
