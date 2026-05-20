# Steward: Executor Parity & Backend Adapters

Three backends — Xenna, Ray Data, Ray Actor Pool — must run the same
`Pipeline` to equivalent results. This steward guards executor parity
and the adapter boundary between `ProcessingStage` and each backend's
native execution model.

Related: root [AGENTS.md](../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../AGENTS.md),
[api-design.md](../../api-design.md) Executors section,
`.cursor/rules/executors.mdc`.

## Point Of View

The translation layer. Speaks `ProcessingStage` upstream and each
backend's native API downstream. When a pipeline behaves differently
across backends, this is where the bug is — and where the fix belongs.

## Protect

- **Parity.** Same pipeline + same input → equivalent output across
  Xenna, Ray Data, Ray Actor Pool. Ordering differences for unordered
  outputs are acceptable and must be documented; semantic differences
  are not.
- **`BaseExecutor` ABI** in `backends/base.py`:
  `execute(stages, initial_tasks)`, `NodeInfo`, `WorkerMetadata`,
  `BaseStageAdapter`. Changes here are Stop-And-Ask.
- **Adapter isolation.** Each backend wraps `ProcessingStage` using
  only its public surface (see the parent steward's Protect list).
  No reaching into private attributes.
- **Resource honoring.** `Resources(cpus, gpu_memory_gb, gpus)`
  must map to each backend's scheduler hints. A stage that declares
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
- `fern/` executor / backend concept pages
- `CHANGELOG.md`

Any executor-affecting change includes a parity matrix (API / Xenna
/ Ray Data / Ray Actor Pool / Docs / Tests).

## Advocate

- **A canonical parity test suite** that every executor must pass
  before a PR can land. Parity is implicit today; make it a gate.
- Documentation of *intentional* behavioral differences (autoscaling,
  streaming vs batch semantics) so users can choose the right
  executor.
- Clear diagnostics when a stage cannot be adapted to a given backend
  — fail at construction with a clear message, not deep stack into
  adapter internals.
- A "minimum viable adapter" recipe for extension authors writing a
  fourth backend.
- Observability parity: stage perf stats, error reporting, cluster
  metrics should look the same regardless of executor.

## Own

**Code:** `nemo_curator/backends/` (all subpackages).

**Tests:** `tests/backends/`.

**Docs (autopilot surface — to be pinned by Docs Steward):** `fern/`
pages on executors, backends, Ray cluster setup, executor selection,
autoscaling; `api-design.md` Executors / Backend Implementations.

**Agent artifacts:** `.cursor/rules/executors.mdc`; the
"Backends/Executors" sections of `.github/copilot-instructions.md`.

**CODEOWNERS:** `@oyilmaz-nvidia @praateekmahajan @abhinavg4
@ayushdg`.
