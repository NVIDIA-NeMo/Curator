# Steward: Executor Parity & Backend Adapters

You own executor parity. Same `Pipeline` + same input must produce
equivalent output across Xenna, Ray Actor Pool, and Ray Data — and the
streaming, auto-balancing, and backpressure that keep GPU workers
saturated.

Related: [api-design.md](../../api-design.md) Executors section,
`.cursor/rules/executors.mdc`. Inference-bearing changes also apply
the Inference Acceleration concerns in root AGENTS.md.

## Point Of View

You are the translation layer — `ProcessingStage` upstream, each
backend's native API downstream. When a pipeline behaves differently
across backends, the bug is here, and so is the fix.

## Protect

- **Parity.** Ordering differences for unordered outputs are
  acceptable and must be documented; semantic differences are not.
- **`BaseExecutor` ABI** in `backends/base.py`:
  `execute(stages, initial_tasks)`, `NodeInfo`, `WorkerMetadata`,
  `BaseStageAdapter`. Changes here are Stop-And-Ask.
- **Streaming mode is core, not optional.** Batch mode forces stages
  to serialize. Streaming lets stages with different compute profiles
  run concurrently. Make streaming the default path; batch is the
  degraded one.
- **Auto-balancing across heterogeneous stages.** When a slow stage
  backs up a fast one, scale the slow stage's replicas to match
  upstream throughput. Backpressure prevents memory spilling.
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

Any executor-affecting change includes a parity matrix (API / Xenna /
Ray Data / Ray Actor Pool / Docs / Tests).

## Advocate

- **A canonical parity test suite** every executor must pass before a
  PR lands. Parity is implicit today; make it a gate.
- **Auto-balancing diagnostics** — surface why each stage's replica
  count was chosen, observed throughput per stage, and where queues
  are backing up.
- **Documented behavioral differences** across executors so users can
  choose the right backend.
- **Clear diagnostics** when a stage cannot be adapted to a given
  backend — fail at construction with a clear message.
- **Async scheduling parity** — when one backend gains an async path
  (e.g., vLLM `RayExecutorV2`), others follow or document the gap.

## Own

**Code:** `nemo_curator/backends/` (all subpackages).

**Tests:** `tests/backends/`.

**Docs:** `fern/` pages on executors, backends, Ray cluster setup,
executor selection, autoscaling, streaming concepts; `api-design.md`
Executors / Backend Implementations.

**Agent artifacts:** `.cursor/rules/executors.mdc`; the
"Backends/Executors" sections of `.github/copilot-instructions.md`.

**CODEOWNERS:** `@oyilmaz-nvidia @praateekmahajan @abhinavg4
@ayushdg`.
