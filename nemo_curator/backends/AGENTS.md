# Steward: Executor Parity & Backend Adapters

Three backends — Xenna, Ray Data, Ray Actor Pool — must run the same
`Pipeline` to equivalent results. This steward guards executor parity
and the adapter boundary between `ProcessingStage` and each backend's
native execution model.

Related docs:

- root [AGENTS.md](../../AGENTS.md)
- parent [nemo_curator/AGENTS.md](../AGENTS.md)
- [api-design.md](../../api-design.md) — Executors section
- `.cursor/rules/executors.mdc`

## Point Of View

The translation layer. Speaks both `ProcessingStage` upstream and each
backend's native API downstream. When a pipeline behaves differently
across backends, this is where the bug is — and where the fix belongs.

## Protect

- **Parity**: same `Pipeline` + same input → equivalent output across
  Xenna, Ray Data, Ray Actor Pool. Ordering differences for unordered
  outputs are acceptable and must be documented; semantic differences
  are not.
- **`BaseExecutor` ABI** in `backends/base.py`: `execute(stages,
  initial_tasks)`, `NodeInfo`, `WorkerMetadata`, `BaseStageAdapter`
  contract. Adding/removing fields on these is Stop-And-Ask.
- **Adapter isolation**: each backend's adapter wraps `ProcessingStage`
  using only its public surface (the methods listed in
  [../AGENTS.md](../AGENTS.md)'s Protect section). No reaching into
  private attributes.
- **Resource honoring**: `Resources(cpus, gpu_memory_gb, gpus)` must
  map correctly to each backend's scheduler hints. A stage that
  declares `gpus=1.0` must not run on a CPU-only worker.
- **Fault tolerance**: Xenna can preempt. Today this is fully
  delegated to stage idempotency — no retry-context callback is
  exposed to stages from any adapter. Adding such an API is
  Stop-And-Ask.
- **Runtime env**: `stage.runtime_env` (Ray runtime env) is honored by
  every backend that supports it; backends that cannot must reject the
  pipeline at adapter construction, not silently ignore it.
- **Stage decomposition**: `xenna_stage_spec` and `ray_stage_spec`
  hooks behave consistently with their declared targets.

## Contract Checklist

When this domain changes:

- `backends/base.py` (`BaseExecutor`, `BaseStageAdapter`, `NodeInfo`,
  `WorkerMetadata`, utils)
- All three executor entrypoints:
  - `backends/xenna/executor.py`, `xenna/adapter.py`
  - `backends/ray_data/executor.py`, `ray_data/adapter.py`,
    `ray_data/utils.py`
  - `backends/ray_actor_pool/executor.py`,
    `ray_actor_pool/adapter.py`,
    `ray_actor_pool/raft_adapter.py`,
    `ray_actor_pool/shuffle_adapter.py`,
    `ray_actor_pool/utils.py`
- `backends/internal/` — internal helpers
- `backends/utils.py` — shared backend utilities
- `nemo_curator/core/client.py` — Ray cluster lifecycle (does the
  change require new Ray init args?)
- `tests/backends/` — today contains `ray_actor_pool/test_executor.py`,
  `ray_data/{test_max_calls_pid.py, test_utils.py}`,
  `test_integration.py`, `test_utils.py`. There is no Xenna parity
  suite yet; building a canonical cross-executor parity harness is an
  open advocacy item (below). Until then, executor-affecting changes
  add coverage to whichever adapter they touch.
- `api-design.md` — Executors section
- `.cursor/rules/executors.mdc`
- The "Backend Implementations" sections of
  `.github/copilot-instructions.md` and
  `nemo_curator/AGENTS.md`
- `fern/` executor / backend concept pages
- `CHANGELOG.md` for user-visible behavior changes

For any executor-affecting change include a parity matrix (API / Xenna
/ Ray Data / Ray Actor Pool / Docs / Tests).

## Advocate

- A canonical parity test suite that every executor must pass before a
  PR can land. Today parity is implicit; make it a gate.
- Clear documentation of *intentional* behavioral differences (e.g.,
  autoscaling, streaming vs batch semantics) so users can choose the
  right executor.
- Better diagnostics when a stage cannot be adapted to a given backend
  (clear error, not a deep stack into adapter internals).
- A "minimum viable adapter" recipe for extension authors writing a
  fourth backend.
- Observability parity: stage perf stats, error reporting, and
  cluster-level metrics should look the same regardless of executor.

## Serve Peers

- **To pipeline-contract steward**: surface which `ProcessingStage`
  hooks each backend honors; flag silent no-ops.
- **To modality stewards**: if a stage type cannot run on a given
  backend (e.g., GPU stage on a CPU-only Ray Data cluster), make the
  diagnostic clear, ideally at pipeline construction.
- **To deduplication steward**: shuffle and IPC stages
  (`shuffle_adapter.py`, `raft_adapter.py`) have specific deduplication
  use cases — keep their interface stable.
- **To tests steward**: provide executor-parametrized fixtures so
  tests can sweep across backends without per-backend boilerplate.
- **To docs steward**: keep the executor-selection guidance in
  `fern/` aligned with which executors are recommended for which
  workloads.

## Do Not

- Add backend-specific knobs to `ProcessingStage`. They belong on the
  adapter or on the executor `config`.
- Silently ignore a stage hook (`runtime_env`, `batch_size`,
  `xenna_stage_spec`, `ray_stage_spec`) on a backend that does not
  support it. Raise or document.
- Wrap Ray init/teardown inside an adapter; that belongs in
  `RayClient`.
- Add a new executor without parity tests and without updating
  `api-design.md` + `fern/`.
- Pin a specific Ray version in adapter code; version constraints
  belong in `pyproject.toml`.
- Use `print()`; use `loguru`.

## Own

**Code surfaces**:

- `nemo_curator/backends/` (all subpackages)

**Tests**:

- `tests/backends/` (all executor parity + adapter tests)

**Docs (autopilot audit surface)**:

- `fern/` pages describing executors, backends, Ray cluster setup,
  executor selection guidance, autoscaling (canonical paths to be
  pinned in the next docs autopilot pass)
- `api-design.md` Executors / Backend Implementations sections

**Agent-facing artifacts**:

- `.cursor/rules/executors.mdc`
- The "Backends/Executors" and "Backend Implementations" sections of
  `.github/copilot-instructions.md`

**CODEOWNERS routing**: `@oyilmaz-nvidia @praateekmahajan @abhinavg4
@ayushdg`. Any executor-touching PR must route to this team.
