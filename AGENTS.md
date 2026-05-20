# Agent Constitution

> Read this file first, then the closest scoped `AGENTS.md` to your change.
> Scoped stewards live under `nemo_curator/`, `nemo_curator/backends/`,
> `nemo_curator/stages/{text,video,synthetic,deduplication}/`, `tests/`,
> `fern/`, `benchmarking/`, and `tutorials/`.

## North Star

NeMo Curator is a Ray-based, backend-agnostic framework for building
distributed data-curation pipelines across text, image, audio, and video
modalities. The same pipeline definition must run unchanged on three
executors (Xenna, Ray Actor Pool, Ray Data). What we protect is the
**single pipeline / many backends** contract and the user-facing surfaces
that depend on it.

## Non-Negotiables

- The `ProcessingStage[X, Y]` / `Task[T]` / `Pipeline` / `BaseExecutor`
  contracts in `nemo_curator/stages/base.py`, `nemo_curator/tasks/tasks.py`,
  `nemo_curator/pipeline/pipeline.py`, and `nemo_curator/backends/base.py`
  are the public ABI. Breaking them requires Stop-And-Ask.
- **All stages MUST be fault-tolerant and retry-safe.** Xenna can preempt
  and reschedule running tasks. Partial state (half-written files, mutated
  caches) must be either idempotent on retry or detected and recovered.
- Same pipeline must produce equivalent results across all three backends.
  Executor-specific behavior is a leak unless explicitly documented.
- GPU code paths must degrade gracefully when CUDA/RAPIDS are unavailable.
  Guard imports; do not crash CPU-only installs.
- Ruff is the formatter and linter (line length 119). Run `pre-commit run
  --all-files` before pushing.
- Commits are signed and signed-off (`git commit -sS`). The DCO and signing
  steps in `CONTRIBUTING.md` are enforced.
- `uv.lock` and `pyproject.toml` move together. The `uv-lock` pre-commit
  hook enforces this.

## Architecture Boundaries

- `nemo_curator/tasks/` — `Task[T]` and modality task types
  (`DocumentBatch`, `ImageBatch`, `VideoTask`, `AudioTask`,
  `FileGroupTask`, `InterleavedBatch`). Owner: pipeline-contract steward.
- `nemo_curator/stages/` — `ProcessingStage[X, Y]`, `CompositeStage`,
  resources, decorators. Modality subtrees own their own stewards.
- `nemo_curator/backends/` — Executors and stage adapters. Backend parity
  steward owns cross-executor equivalence.
- `nemo_curator/pipeline/` — `Pipeline` and `Workflow` orchestration.
- `nemo_curator/core/` — `RayClient`, constants, serve. Touches Ray
  cluster lifecycle.
- `fern/` — canonical user-facing documentation (Fern site published at
  `docs.nvidia.com/nemo/curator`). **`docs/` is deprecated**; do not add
  pages there.
- `tests/` — mirrors source layout. CPU-default; GPU tests via
  `@pytest.mark.gpu`. `tests/L0_Unit_Test_CPU.sh` and
  `tests/L0_Unit_Test_GPU.sh` are the CI entrypoints.
- `tutorials/` — runnable user-facing examples per modality.
- `benchmarking/` — perf gates, ALM/audio profiling, nightly benchmarks.

## Governance Alignment

- **Domain/page owners**: `.github/CODEOWNERS` is the source of truth for
  human review routing. AI stewards advise; CODEOWNERS approve. Stewards
  must route review to the listed humans when a change crosses their lines
  (deduplication, text embedders/classifiers, backends, synthetic, video,
  docs, automation, benchmarking).
- **Human governance stewards**: `@NVIDIA-NeMo/curator_reviewers` (default),
  `@NVIDIA-NeMo/docs_team` (docs), `@NVIDIA-NeMo/automation` (CI/build).
- **Canonical knowledge**: product documentation in `fern/` is the
  authoritative user-facing layer. Cursor rules (`.cursor/rules/*.mdc`),
  Copilot instructions (`.github/copilot-instructions.md`), Claude skills
  (`.claude/skills/`), and these `AGENTS.md` files are agent-facing
  artifacts that **extend** canonical docs. If important product knowledge
  is only in an agent artifact, that is a docs bug — fix `fern/` first,
  then point the artifact at it.
- **Portfolio coordination**: this repo's docs publish into the NVIDIA
  NeMo docs portfolio. Cross-repo terminology and duplicate-fact conflicts
  route to `@NVIDIA-NeMo/docs_team`. Stewards are repo-local; do not add a
  steward to mediate a cross-repo concern — escalate instead.

## Stakes

- **Users**: copy-paste of stage imports, CLI invocations, or YAML config
  must work as documented. Fabricated flags or wrong defaults break the
  first thing a new user tries.
- **Operators**: Ray cluster lifecycle, GPU memory limits, and autoscaling
  behavior must match what the docs claim. A wrong retry-policy claim
  costs node-hours.
- **Contributors**: the `ProcessingStage` contract is the extension point.
  Silent breaks in `process()` signature, resource shape, or
  `inputs/outputs` validation cascade to every downstream stage.
- **Extension authors** (custom stages, custom executors): the public
  abstract methods on `ProcessingStage`, `Task`, and `BaseExecutor` are
  the contract.
- **Agents** (Claude / Cursor / Copilot): if `AGENTS.md`, cursor rules,
  copilot instructions, and `fern/` disagree on the same fact, agents
  produce wrong code. Convergence across these surfaces is non-optional.

## Stop And Ask

Pause and request human review before:

- Changing any signature, default, or return shape on `ProcessingStage`,
  `Task`, `Pipeline`, `BaseExecutor`, `BaseStageAdapter`, `Resources`, or
  `RayClient`.
- Adding a new runtime dependency or a new optional-extras group in
  `pyproject.toml`.
- Touching `pyproject.toml`, `uv.lock`, `docker/`, or `.github/workflows/`
  in ways that change build/release surface.
- Changing migration paths, on-disk cache layouts, ID-generator behavior
  in deduplication, or any persistent artifact schema.
- Anything irreversible: deleting cached artifacts, force-pushing, mass
  doc deletions, renaming public modules.
- Security/auth changes, including credentials flows in Ray cluster setup.
- Changing concurrency model: stage `batch_size`, executor parallelism
  defaults, autoscaling knobs.
- Tests and code disagree about behavior, or a bug cannot be reproduced
  locally.

## Anti-Patterns

- Adding a backend-specific code path inside a `ProcessingStage`. Backend
  differences belong in adapters under `nemo_curator/backends/<name>/`.
- Importing `cudf`, `cupy`, `cuml`, or other RAPIDS/CUDA libs at module
  top level **outside `nemo_curator/stages/deduplication/`**. Use lazy
  imports inside GPU-only methods so CPU installs still load the module.
  (Deduplication today imports RAPIDS at module top level and therefore
  requires `deduplication_cuda12` to import at all — see the
  Deduplication steward; making those imports lazy is tracked there as
  active advocacy.)
- Editing `docs/` to fix a documentation issue. `docs/` is deprecated —
  fix `fern/` instead. See [fern/AGENTS.md](fern/AGENTS.md).
- Adding a YAML config field or CLI flag without a corresponding Pydantic
  / dataclass / argparse declaration. Documented surfaces must trace to
  a typed source of truth.
- Skipping `git commit -sS`. Unsigned/un-DCO commits get rejected.
- Bypassing `uv-lock` pre-commit when `pyproject.toml` changes.
- Treating `.cursor/rules/`, `.github/copilot-instructions.md`, and these
  `AGENTS.md` files as redundant. They serve different agents but must
  agree on facts. Update them together when shared facts change.

## Steward System

Agents read this file plus the closest scoped `AGENTS.md` for the path
they are touching. Root is the constitution and routing guide. Scoped
files are domain stewards that own local invariants, refusal patterns,
docs, tests, examples, and checks.

Each scoped steward follows the same operating model:

- **Point of View** — who/what the domain represents
- **Protect** — invariants, contracts, quality bars, failure modes
- **Contract Checklist** — concrete surfaces to inspect when this domain
  changes
- **Advocate** — features, fixes, investments to push for
- **Serve Peers** — upstream/downstream domains needing better contracts,
  diagnostics, docs, tests
- **Do Not** — local anti-patterns
- **Own** — tests, docs, examples, fixtures, maintenance checks

Cross-boundary work includes a **Steward Notes** block in the PR
description naming which stewards were consulted, which findings were
accepted, deferred, or merged-as-duplicate, and any required collateral
(docs/tests/examples) updated in the same PR.

Stewards respect `.github/CODEOWNERS`. When a change affects a path with
named human owners, the steward's job is to surface the relevant
invariants and route review to those humans — not to replace them.

### Contract Checklist (repo-wide)

For any cross-surface change, identify every surface that should agree:

- Public API: `ProcessingStage` subclasses, public functions in
  `nemo_curator/__init__.py`, factory exports
- Programmatic surface: imports used in tutorials and quickstarts
- Schema / types: `Task` dataclass fields, `Resources` fields, Pydantic
  config models
- Backend adapters: Xenna, Ray Data, Ray Actor Pool — does the change
  hold across all three?
- Docs: `fern/` pages that describe the changed surface (find via grep)
- Examples: `tutorials/<modality>/`, `tutorials/quickstart.py`
- Tests: unit tests, GPU tests, backend-parity tests
- Benchmarks: `benchmarking/` configs and runners
- Changelog: `CHANGELOG.md` for user-visible behavior changes
- Cursor rules / Copilot instructions: if they describe the changed
  surface, update them in the same PR

Every accepted finding names required proof and collateral, or
explicitly says `no collateral: <reason>`. Docs and examples move in the
same PR as user-facing behavior changes unless synthesis records why
they are unaffected. Contract-affecting PRs that span backends include a
parity matrix (API / Xenna / Ray Data / Ray Actor Pool / Docs / Tests).

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

Before synthesis, factual findings must pass a verification gate when
machine-checkable: grep the source for the flag, trace the schema, run
the snippet. Findings that cannot be machine-verified must carry
`manual-confirmation-needed` or `not-machine-verifiable`.

### Convergence

When two or more stewards independently flag the same finding, it is
automatically P0 regardless of individual severity. Call it out
explicitly in synthesis: "X stewards independently flagged Y." When the
same *shape* of finding recurs across audits (fabricated flag,
miscounted entity, stale version pin, cross-page disagreement), escalate
to a **Known Regression Pattern** below.

### Steward Swarms

Stewards spawn as independent agents, each reading root plus its closest
scoped `AGENTS.md`, each advocating only for its domain, each returning
findings in the Steward Signal Format. The implementing agent owns
synthesis and final decisions. Keep PR scope bounded to accepted
findings; defer unrelated suggestions to not-now.

**Triggers**: `ask stewards`, `bugbash`, `review swarm`, `steward
synthesis` → Implementation Review. `audit docs`, `content audit`,
`accuracy pass` → Content Audit.

**Implementation Review** (default) — Stewards defend invariants
against the diff. Severity: P0 breaks a shipped contract, P1 degrades
without breaking, P2/P3 polish or advocacy. Match depth to risk: typo
and link fixes move after automated checks; technical-accuracy changes
need agent first-pass plus human review; cross-domain or
standards-impacting changes route to human governance stewards.

**Content Audit** — Stewards verify doc claims against source code.
Findings cite `<source-file:line> → <doc-file:line>` divergence and the
corrected text. P0: factually wrong (wrong default, missing flag, named
entity does not exist, copy-paste would break). P1: incomplete, stale,
or misleading but not wrong. P2/P3: voice and polish. Initial rollout
gates merge on verified P0; mature rollout gates on P0+P1.

### Global Sweep On Accepted P0s

When a P0 names a wrong factual claim (wrong default, endpoint shape,
flag name, file path, retry behavior), the fix is not "edit the page
where it was flagged." The fix is: grep the entire `fern/` site (and
`tutorials/`, cursor rules, copilot instructions, root `README.md`,
`api-design.md`) for the same claim and correct every instance before
the P0 is closed. Stewards check their primary owned surfaces;
cross-surface propagation is the dominant failure mode of narrow fixes.

### Doc Autopilot

The Content Audit mode is the primary mechanism for keeping docs
accurate over time. The Docs Steward ([fern/AGENTS.md](fern/AGENTS.md))
owns the recurring ritual:

1. **Merge gate** — Doc-shaped PRs (IA refactors, release notes, large
   content updates, README sweeps) gate on a Content Audit having run.
   Initial rollout: verified P0 only. Mature rollout: P0 and P1
   actioned or explicitly deferred.
2. **Periodic re-audit** — Spawn the full swarm at every release
   boundary (when `fern/versions/v*.yml` lands) and on a recurring
   cadence (every 4–6 weeks).
3. **Source-triggered targeted re-audit** — When code touching a
   documented public surface changes, the relevant scoped steward's
   Content Audit runs against its owned doc pages (`Own:` list in that
   steward).

### Docs-First Agent Artifact Evaluation

Before creating or expanding a `.cursor/rules/*.mdc` rule, a
`.claude/skills/<skill>`, an MCP workflow, a script, a CLI helper, or a
prompt template:

1. Identify where agents struggle.
2. Check whether `fern/` is missing, unclear, stale, or scattered.
3. Fix or restructure `fern/` first when that would solve the problem
   for both humans and agents.
4. Create or expand the agent-facing artifact only when docs alone
   cannot reliably support the workflow.
5. Record ownership, review path, source docs, maintenance trigger, and
   evaluation proof for the artifact.

Important product knowledge belongs in `fern/`; agent artifacts point
back to it.

## Backlog / Roadmap / Prioritization

When asked for prioritization, consult all scoped stewards and produce:
raw steward signals, confidence, dependencies, risks, convergence,
minority reports, ranked backlog, not-now items.

## Known Regression Patterns

Seeded from this repo's actual structure and recurring failure modes.
Stewards in autopilot mode hunt these by default. Each pattern names
the verification recipe.

- **Fabricated CLI / config fields.** Doc claims a flag, env var, or
  YAML key that does not exist in source. *Verify*: every flag or config
  field documented anywhere must trace to a `pyproject.toml` script
  entry, an argparse declaration, a Pydantic class field, or a
  dataclass field. Grep the source file.
- **Stage-contract drift.** Doc claims a `ProcessingStage` accepts /
  produces a task type or has a resource shape that no longer matches
  `nemo_curator/stages/base.py` or the stage's own definition. *Verify*:
  read the stage's `inputs`, `outputs`, `process`, and `resources` and
  cross-check the doc.
- **Executor parity drift.** Doc claims a behavior holds for all
  executors but the implementation differs across
  `nemo_curator/backends/{xenna,ray_data,ray_actor_pool}/`. *Verify*:
  grep all three adapters/executors for the named feature.
- **Deduplication CUDA gating.** Doc or example uses a deduplication
  stage without naming the `deduplication_cuda12` extras requirement
  or the GPU prerequisite. *Verify*: every dedup example must state
  the install extras and minimum GPU. Current reality: top-level RAPIDS
  imports in `nemo_curator/stages/deduplication/{io_utils.py,
  exact/identification.py, fuzzy/*.py, semantic/*.py,
  shuffle_utils/rapidsmpf_shuffler.py}` mean *importing* the dedup
  package requires `deduplication_cuda12`. Making these lazy is an
  open advocacy item with the Deduplication steward; until then, dedup
  docs and tutorials must say so explicitly.
- **`docs/` vs `fern/` regression.** Edits land in `docs/` instead of
  `fern/`. *Verify*: `docs/` is deprecated. Any new doc-changing PR
  that touches `docs/` is a P0 unless it is an explicit
  decommissioning change.
- **Doc-snippet rot.** Tutorial or quickstart imports / CLI lines drift
  from current public surface. *Verify*: every `from nemo_curator...`
  import in `tutorials/` and `fern/` round-trips against current
  `nemo_curator/` modules; every CLI invocation matches an actual
  argparse entrypoint.
- **Naming and counting drift.** Any doc claim with a count (number of
  classifiers, supported modalities, backends), a version pin, or an
  entity name is stale-by-default. *Verify*: re-count or re-verify
  against current source on every audit pass.
- **Cross-page inconsistency.** Same fact stated differently across
  `fern/` pages and `README.md` / `CONTRIBUTING.md` / cursor rules.
  *Verify*: cross-steward synthesis checks for two surfaces disagreeing
  on the same fact.
- **Narrow-fix regression.** A P0 corrected on one page survives on
  sibling pages and resurfaces in the next audit. *Verify*: every
  accepted P0 closure runs a global grep for the corrected claim across
  the entire `fern/` site, plus `tutorials/`, `README.md`,
  `api-design.md`, `.cursor/rules/`, `.github/copilot-instructions.md`.
- **Unverified finding regression.** A steward reports a divergence
  that a source grep would have disproved. *Verify*: every factual P0/P1
  carries machine-verified / manual-confirmation-needed /
  not-machine-verifiable before triage.

Add new patterns as audits surface them, each with a verification
recipe.

## Steward Feedback Loop

- **Steward miss**: when a bug escapes an applicable steward, update
  the checklist, add a regression test, add a docs/snippet check, or
  record why the miss should not become policy.
- **Steward overreach**: when a steward repeatedly pulls unrelated work
  into PRs, narrow the checklist, split the steward, or move the
  concern to follow-up.
- Repeated high-quality findings become checklist items. Repeated
  convergent findings become Known Regression Patterns.
- Repeated noisy findings get pruned, clarified, auto-suppressed, or
  scoped down before reviewers learn to ignore the swarm.
- **Signal budget**: cap each audit pass at ~10 P2/P3 findings. Beyond
  that, move to not-now unless convergent.
- **False-positive target**: track disputed P0/P1; aim under 15% before
  tightening gates.
- **Steward health**: track signal-to-noise per steward; below
  threshold gets reviewed for reduction or retirement.

## Measurement

- **Content health**: owner coverage (CODEOWNERS), `fern/` freshness,
  broken-link rate (`fern/_fix_broken_links.py` output).
- **Process health**: PR-to-merge time, pre-commit pass rate, review
  SLA, steward finding acceptance rate.
- **Agent readiness**: cursor rules / copilot instructions / `fern/`
  parity, agent task completion rates.
- **Steward health**: false-positive rate, P2/P3 volume, convergence
  rate, recurring Known Regression Patterns.
- **Cross-domain health**: duplicate-fact incidents, terminology drift,
  unresolved ownership conflicts.

Tune quarterly: tighten gates that catch real issues, relax gates that
create friction without value, split or retire noisy stewards, promote
repeated high-quality findings to checklists or automation.

## When To Consult

- **Proactively consult** stewards for cross-boundary, public-facing,
  hard-to-reverse, performance-sensitive, concurrency-sensitive,
  security-sensitive, or contract-affecting work.
- Use the **nearest** steward for local work.
- Use **multiple** stewards when ownership lines cross.
- Parallelize steward consultation only when the questions are
  independent — independent stewards surface convergence.
- Keep final synthesis and implementation accountability with the
  implementing agent.
- Route structural, cross-domain, standards-impacting,
  ownership-affecting, or exception/waiver decisions to human governance
  stewards (CODEOWNERS teams).

### Ask Stewards

For implementation work: consult affected stewards, return synthesis
before or during the change, include accepted/deferred findings, merged
duplicates, minority reports, required proof, collateral updates, and
not-now items.

For content audits: spawn stewards across all domains the docs touch;
default to all scoped stewards on a site-wide `fern/` IA refactor; run
the swarm in parallel; synthesize into a triaged P0/P1/P2 punch list
cited at `file:line`; machine-verify factual claims before triage; for
every accepted P0 that names a wrong factual claim, grep the whole
docs site and fix every instance; gate merge on verified P0 during
initial rollout; keep P2/P3 inside the signal budget; route cross-domain
conflicts to human governance stewards.

For multi-surface work, include a parity matrix:

| Contract | Public API | Xenna | Ray Data | Ray Actor Pool | Docs (fern) | Tutorials | Tests |
| -------- | ---------- | ----- | -------- | -------------- | ----------- | --------- | ----- |

## Extension Routing

- Custom stages: subclass `ProcessingStage` in
  `nemo_curator/stages/<modality>/<area>/`. Register is automatic via
  `StageMeta`.
- Custom tasks: subclass `Task[T]` in `nemo_curator/tasks/`.
- Custom executors: subclass `BaseExecutor` in
  `nemo_curator/backends/<name>/` with a matching `BaseStageAdapter`.
- Composite stages: `CompositeStage` in `nemo_curator/stages/base.py`.
- Per-modality patterns are documented in
  `.cursor/rules/modality-structure.mdc` and
  `.cursor/rules/processing-stage-patterns.mdc`.

## Done Criteria

- `pytest` (and `pytest -m gpu` on GPU branches) passes; ruff lint/format
  passes; `pre-commit run --all-files` is clean
- `uv.lock` in sync if `pyproject.toml` changed
- Docs changes land in `fern/` (not `docs/`); changelog updated for
  user-visible behavior changes; release notes go in `fern/` only
- Tutorials / `quickstart.py` updated when public surface changes; or an
  explicit `no-impact: <reason>` note
- Benchmark notes updated for perf-sensitive changes
- Every accepted steward finding has test/docs/example/benchmark proof
  or an explicit no-impact note
- Factual P0/P1 findings carry verification status
- For doc-shaped PRs: a Content Audit has run; accepted P0s have been
  globally grep-swept across `fern/`, `tutorials/`, root docs, cursor
  rules, and copilot instructions
- Agent-facing artifact changes (`.cursor/rules/`, `.claude/skills/`,
  `.github/copilot-instructions.md`) record source docs, owner, review
  trigger, and docs-first evaluation
- Commits signed (`-sS`); DCO satisfied

## Review Notes

PR descriptions should flag:

- Weird tests, unused public names, suppressions, dead code
- Benchmark gaps for perf-sensitive changes
- Steward disagreement; convergent findings (multiple stewards flagging
  the same thing)
- Unverified factual findings
- Signal/noise threshold breaches
- Governance exceptions, waivers, timeboxed deferrals
- Ownership or CODEOWNERS gaps
- Docs-first evaluation gaps for new agent-facing artifacts
- Deferred / not-now findings
