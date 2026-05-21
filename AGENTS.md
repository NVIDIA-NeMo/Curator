# Agent Constitution

## North Star

NeMo Curator builds scalable, configurable pipelines to curate text,
image, audio, and video datasets for accurate AI applications. Customers
run trillion-token pretraining, multi-modal corpora, and synthetic data
generation at multi-node GPU scale. We protect the conditions that make
this possible:

- **Higher accuracy** — better data, less training compute
- **Faster processing** — RAPIDS-accelerated dedup, classification, and inference
- **Scalability** — multi-node, multi-GPU, designed for PB-scale workloads
- **Classifier models** — state-of-the-art open quality/domain/safety models
- **Deploy anywhere** — Python APIs, runnable on CSP and on-prem

The architecture rests on three design pillars:

- **Task-centric** — Tasks (`DocumentBatch`, `ImageBatch`, `VideoTask`, `AudioTask`) are the unit of data flowing through pipelines
- **Map-style** — every stage transforms tasks to tasks; this constraint enables auto-balancing and streaming
- **Fault tolerant** — stages survive preemption and reschedule; partial state is recoverable

The same pipeline definition must run unchanged across Xenna, Ray Actor
Pool, and Ray Data executors.

## Non-Negotiables

- `ProcessingStage[X, Y]`, `Task[T]`, `Pipeline`, `BaseExecutor`, and
  `Resources` are public ABI (`nemo_curator/stages/base.py`,
  `tasks/tasks.py`, `pipeline/pipeline.py`, `backends/base.py`,
  `stages/resources.py`). Breaks here are Stop-And-Ask.
- All stages MUST be fault-tolerant and retry-safe. Xenna preempts and
  reschedules; partial state must be idempotent or recoverable.
- Same pipeline → equivalent results across the streaming
  executors (Xenna and Ray Data). Ray Actor Pool is dedup-only.
  Executor-specific behavior must be explicitly documented.
- GPU code paths degrade gracefully when CUDA/RAPIDS aren't installed.
  Guard imports outside the dedup tree; don't crash CPU-only installs.

## Architecture Boundaries

| Path | Steward / contract |
| --- | --- |
| `nemo_curator/tasks/` | `Task[T]` and modality tasks (`DocumentBatch`, `ImageBatch`, `VideoTask`, `AudioTask`, `FileGroupTask`, `InterleavedBatch`) |
| `nemo_curator/stages/` | `ProcessingStage`, `CompositeStage`, `Resources`. Modality subtrees have their own stewards |
| `nemo_curator/backends/` | Executors + adapters. Backend parity steward |
| `nemo_curator/pipeline/` | `Pipeline`, `Workflow` |
| `nemo_curator/core/` | `RayClient`, Ray cluster lifecycle |
| `fern/` | **Canonical** user-facing docs (`docs.nvidia.com/nemo/curator`). `docs/` is deprecated |
| `tests/` | Mirrors source. CPU-default; `@pytest.mark.gpu` for GPU. L0 scripts are CI entrypoints |
| `tutorials/` | Runnable examples per modality |
| `benchmarking/` | Perf gates, nightly benchmarks |

## Governance Alignment

`.github/CODEOWNERS` is the source of truth for human review. AI
stewards advise; CODEOWNERS approve. Route review to the named humans
when a change crosses their lines — never replace them.

Canonical knowledge lives in `fern/`. Cursor rules
(`.cursor/rules/*.mdc`), Copilot instructions
(`.github/copilot-instructions.md`), Claude skills (`.claude/skills/`),
and `AGENTS.md` files extend canonical docs — they do not replace them.
If important product knowledge is only in an agent artifact, fix
`fern/` first.

Cross-repo terminology or duplicate-fact conflicts route to
`@NVIDIA-NeMo/docs_team`. Stewards are repo-local; escalate rather than
spawning a steward to mediate a cross-repo concern.

## Stop And Ask

Pause for human review before:

- Changing any signature, default, or return shape on `ProcessingStage`,
  `Task`, `Pipeline`, `BaseExecutor`, `BaseStageAdapter`, `Resources`,
  `RayClient`.
- Adding a runtime dependency or extras group; touching `pyproject.toml`
  / `uv.lock` / `docker/` / `.github/workflows/` in ways that change
  build or release surface.
- Migrations, on-disk cache layouts, dedup ID-generator behavior, or
  any persistent artifact schema.
- Irreversible operations (mass deletions, force-push, public-module
  rename).
- Security / auth changes, Ray cluster auth flows.
- Concurrency changes: `batch_size`, executor parallelism, autoscaling
  knobs.
- Tests and code disagree, or a bug can't be reproduced locally.

## Anti-Patterns

- Adding a backend-specific code path inside a `ProcessingStage`.
  Differences belong in `nemo_curator/backends/<name>/`.
- Importing `cudf`, `cupy`, `cuml`, or other RAPIDS libs at module top
  level **outside `nemo_curator/stages/deduplication/`**. (Deduplication
  imports RAPIDS at module top level today; lazy-import work is open
  advocacy with the Dedup steward.)
- Editing `docs/` for product reasons — it's deprecated. See
  [fern/AGENTS.md](fern/AGENTS.md).
- Documenting a flag, config field, classifier name, codec, or default
  that doesn't trace to a Pydantic / dataclass / argparse declaration in
  source.
- Treating `.cursor/rules/`, `copilot-instructions.md`, and `AGENTS.md`
  as redundant. They serve different agents but must agree on facts;
  update them together when shared facts change.

## Steward System

Each scoped steward is a domain agent with the same operating model:

- **Point of View** — what the domain represents and why it matters
- **Protect** — invariants, contracts, quality bars
- **Contract Checklist** — surfaces to inspect when this domain changes
- **Advocate** — investments to push for
- **Own** — code, tests, docs, fixtures, checks (with CODEOWNERS routing)

Optional sections — include only when they carry weight a careful
reader can't infer from the other sections:

- **Do Not** — only for non-obvious local anti-patterns
- **Serve Peers** — only for explicit cross-domain obligations

Cross-boundary work includes a **Steward Notes** block in the PR
description naming consulted stewards, accepted/deferred findings,
merged duplicates, and required collateral.

## Inference Acceleration (cross-cutting concern)

Inference acceleration is a cross-cutting concern owned at root, not
a separately-scoped steward. When code changes touch an
inference-bearing surface, these invariants apply in addition to the
relevant modality steward's:

- **Speed-of-light per model.** TensorRT-LLM, memory optimization,
  FP8/INT8 quantization, and paged attention are the floor — not the
  ceiling.
- **Serving pattern is explicit.** Either *in-process* (model loaded
  per stage worker; must fit the worker's GPU memory budget) or
  *server-endpoint* (CPU-only Curator stages calling a local model
  server; replica count, per-client concurrency, queue behavior, and
  serialization overhead documented).
- **Model server choice.** vLLM is canonical. Ray Serve is preferred
  for Ray-native ergonomics. Dynamo is supported when NV-optimized
  inference matters more than integration cost.
- **Async scheduling** features (e.g., vLLM's `RayExecutorV2`) are
  adopted when they reduce GPU idle time.
- **Benchmarks capture model + serving stack + hardware.** Numbers
  without that context are noise.

Inference-bearing surfaces include classifier / embedder / VLM / LLM
stages, semantic-dedup embedding integrations, `runtime_env` carrying
model-serving deps, and any documented throughput / latency / GPU-
utilization claim.

### Contract Checklist (repo-wide)

For cross-surface changes identify every surface that should agree:
public API (`ProcessingStage` subclasses, `nemo_curator/__init__.py`),
schema (`Task`, `Resources`, Pydantic models), all three backend
adapters, `fern/` pages, `tutorials/`, tests, benchmarks, `CHANGELOG.md`,
and matching agent artifacts. Every accepted finding names required
proof and collateral, or explicitly says `no collateral: <reason>`.
Contract-affecting PRs that span backends include a parity matrix
(API / Xenna / Ray Data / Docs / Tests; Ray Actor Pool only for
dedup-affecting changes).

### Steward Signal Format

```text
Steward:
Area:
Severity: P0/P1/P2/P3
Invariant:
Evidence: <source-file:line> [→ <doc-file:line> for content audit]
User Impact:
Required Fix:
Required Proof:
Collateral:
Confidence:
Verification Status: machine-verified / manual-confirmation-needed / not-machine-verifiable
```

Factual claims that are machine-checkable must pass a verification gate
before triage — grep, schema-trace, or signature-check the source.
Findings that cannot be machine-verified must carry
`manual-confirmation-needed` or `not-machine-verifiable`.

### Convergence

When two or more stewards independently flag the same finding it is
automatically P0 regardless of individual severity. Call it out
explicitly in synthesis. When the same *shape* of finding recurs across
audits, promote it to a **Known Regression Pattern** below.

### Steward Swarms

Stewards spawn as independent agents, each reading root plus their
closest scoped file, each advocating only for their domain, each
returning findings in the Steward Signal Format. The implementing agent
owns synthesis and final decisions; stewards advise.

**Triggers:** `ask stewards`, `bugbash`, `review swarm`, `steward
synthesis` → **Implementation Review** (defend invariants against the
diff). `audit docs`, `content audit`, `accuracy pass` → **Content
Audit** (verify doc claims against source). P0 = breaks a shipped
contract or names a wrong factual claim. P1 = degrades / stale /
misleading. P2/P3 = polish or advocacy.

**When to consult:** nearest steward for local work; multiple stewards
when ownership lines cross; full swarm for site-wide `fern/` IA
refactors or cross-cutting refactors. **Consult the Inference
Acceleration Steward** for any inference-bearing or model-serving
change. Parallelize only when the questions are independent — independent
stewards surface convergence. Route structural, cross-domain,
standards-impacting, or ownership-affecting decisions to human
governance stewards.

Match depth to risk: typo and link fixes ship after automated checks;
technical-accuracy changes need agent first-pass plus human review;
standards-impacting changes route to `@NVIDIA-NeMo/docs_team` or
`curator_reviewers`.

### Global Sweep On Accepted P0s

When a P0 names a wrong factual claim (wrong default, endpoint shape,
flag name, file path), the fix is *not* "edit the page where it was
flagged." The fix is: grep the entire `fern/` site (and `tutorials/`,
`.cursor/rules/`, `.github/copilot-instructions.md`, `README.md`,
`api-design.md`) for the same claim and correct every instance before
the P0 is closed. Cross-surface propagation is the dominant failure
mode of narrow fixes.

### Doc Autopilot

The Content Audit swarm is the primary mechanism for keeping docs
accurate over time. The Docs Steward ([fern/AGENTS.md](fern/AGENTS.md))
owns three triggers:

1. **Merge gate** — Doc-shaped PRs (IA refactors, release notes, large
   content updates, README sweeps) gate on a Content Audit having run.
   Initial rollout: verified P0 only. Mature rollout: P0 + P1.
2. **Periodic re-audit** — Full swarm at every release boundary (a new
   `fern/versions/v*.yml` lands) and on a 4–6 week cadence.
3. **Source-triggered re-audit** — When code touching a documented
   public surface changes, the relevant scoped steward's Content Audit
   runs against its owned doc pages.

Each scoped steward's **Own** list is its audit surface in autopilot
mode. Current state is manual rollout; wiring these into CI is open
advocacy with the Docs Steward.

### Docs-First Agent Artifact Evaluation

Before creating or expanding a cursor rule, Claude skill, MCP workflow,
script, CLI helper, or prompt template:

1. Identify where agents struggle.
2. Check whether `fern/` is missing, unclear, stale, or scattered.
3. Fix `fern/` first when that would solve the problem for humans and
   agents alike.
4. Create the artifact only when docs alone cannot reliably support
   the workflow.
5. Record ownership, review path, source docs, maintenance trigger,
   and evaluation proof for the artifact.

## Known Regression Patterns

Stewards in autopilot mode hunt these by default. Each pattern names
the verification recipe.

- **Fabricated CLI / config fields.** Doc claims a flag, env var, or
  YAML key that doesn't exist. *Verify:* every flag traces to a
  `pyproject.toml` entry, argparse declaration, Pydantic field, or
  dataclass field. Grep the source.
- **Stage-contract drift.** Doc claims `ProcessingStage` input/output
  types or resource shapes that no longer match
  `stages/base.py` or the stage's definition. *Verify:* read
  `inputs`, `outputs`, `process`, `resources` and cross-check.
- **Executor parity drift.** Doc claims a streaming behavior across
  Xenna and Ray Data but the two adapters differ. *Verify:* grep
  `backends/{xenna,ray_data}/` for the named feature. (Ray Actor
  Pool is dedup-only; parity comparisons are Xenna vs Ray Data.)
- **Inference performance regression.** A model-bearing stage's
  throughput, latency, or GPU utilization drops without explanation,
  or a new inference path lands without speed-of-light benchmarks
  (TensorRT-LLM, quantization, paged attention). *Verify:* every
  inference change carries a benchmark capturing model + serving
  stack + hardware.
- **Deduplication CUDA gating.** Dedup example doesn't name the
  `deduplication_cuda12` extras or GPU requirement. Today, importing
  the dedup package itself requires `deduplication_cuda12` (RAPIDS at
  module top level); until lazy-import work lands, docs and tutorials
  must say so. *Verify:* grep dedup docs for the install prereq.
- **`docs/` vs `fern/` regression.** Product edits land in `docs/`
  instead of `fern/`. *Verify:* any new doc-changing PR that touches
  `docs/` is a P0 unless explicit decommissioning.
- **Doc-snippet rot.** `from nemo_curator…` imports or CLI lines in
  `tutorials/` and `fern/` drift from current public surface.
  *Verify:* round-trip imports against the package; round-trip CLI
  lines against argparse.
- **Naming and counting drift.** Any doc claim with a count, version
  pin, or named entity is stale-by-default. *Verify:* re-check
  against current source on every audit pass.
- **Cross-page inconsistency.** Same fact stated differently across
  `fern/`, `README.md`, `CONTRIBUTING.md`, cursor rules. *Verify:*
  cross-steward synthesis explicitly checks for disagreement.
- **Narrow-fix regression.** A P0 fixed on the flagged page survives
  on sibling pages. *Verify:* every accepted P0 closure runs the
  Global Sweep above.
- **Unverified finding regression.** A steward reports a divergence
  that a source grep would have disproved. *Verify:* every factual
  P0/P1 carries a verification status before triage.

Add new patterns as audits surface them, each with a verification
recipe.

## Steward Feedback Loop

- **Miss**: bug escaped an applicable steward → update the checklist,
  add a regression test, or record why this miss shouldn't become
  policy.
- **Overreach**: steward pulls unrelated work into PRs → narrow the
  checklist, split the steward, move concerns to follow-up.
- Repeated high-quality findings become checklist items; repeated
  convergent findings become Known Regression Patterns; repeated
  noisy findings get pruned or scoped down.
- **Signal budget**: cap each audit pass at ~10 P2/P3 findings —
  beyond that, move to not-now unless convergent.
- **False-positive target**: track disputed P0/P1, aim under 15%
  before tightening gates.
- **Steward health**: track signal-to-noise per steward; below
  threshold gets reduced or retired.

## Measurement

Track: owner coverage (CODEOWNERS), `fern/` freshness and broken-link
rate, PR-to-merge time, pre-commit pass rate, steward finding
acceptance rate, false-positive rate, P2/P3 volume, convergence rate,
recurring Known Regression Patterns, duplicate-fact incidents,
terminology drift. Tune quarterly.

## Extension Routing

- Custom stages: subclass `ProcessingStage` in
  `nemo_curator/stages/<modality>/<area>/`. Auto-registered by
  `StageMeta`.
- Custom tasks: subclass `Task[T]` in `nemo_curator/tasks/`.
- Custom executors: subclass `BaseExecutor` in
  `nemo_curator/backends/<name>/` with a matching `BaseStageAdapter`.
- Composite stages: `CompositeStage` in `nemo_curator/stages/base.py`.
- Per-modality patterns documented in
  `.cursor/rules/modality-structure.mdc`,
  `.cursor/rules/processing-stage-patterns.mdc`.

## Done Criteria

- `pytest` (and `pytest -m gpu` for GPU branches), ruff lint/format,
  and `pre-commit run --all-files` clean.
- `uv.lock` in sync if `pyproject.toml` changed.
- Docs changes land in `fern/` (not `docs/`). `CHANGELOG.md` updated
  for user-visible behavior. Release notes in `fern/` only.
- Tutorials / `quickstart.py` updated when public surface changes, or
  an explicit `no-impact: <reason>` note.
- Benchmark notes for perf-sensitive changes; inference-bearing
  changes carry model + serving-stack + hardware context.
- Every accepted steward finding has proof or an explicit no-impact
  note. Factual P0/P1 findings carry verification status.
- For doc-shaped PRs: a Content Audit ran; accepted P0s were
  globally grep-swept.
- Agent-facing artifact changes record source docs, owner, review
  trigger, and docs-first evaluation.
- Commits signed and signed-off (`-sS`).

PR descriptions should flag: convergent findings, unverified factual
findings, signal/noise breaches, governance exceptions, CODEOWNERS
gaps, docs-first evaluation gaps for new agent artifacts, deferred /
not-now findings.
