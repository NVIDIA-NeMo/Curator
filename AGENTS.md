<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Agent Constitution — NeMo Curator

Ordinary work: use this root map plus only scoped maps on the target path.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Pillars

- NeMo Curator is a scalable, accelerator-aware library for turning raw multimodal data into traceable, reusable curated datasets.
- Processing stages, tasks, workflows, and backend adapters are public contracts; changes preserve lifecycle, lineage, resource, and resumability semantics.
- Optional feature families stay import-safe behind their extras, and focused CPU tests remain runnable without unrelated heavyweight dependencies.
- Code, tests, Fern documentation, tutorials, and benchmark evidence move together when a user-visible contract changes.
- The steward network is advisory and evidence-driven; CODEOWNERS and human reviewers remain the approval authority.

## Search Discipline

- At task start, read the root map before repository discovery; do not inventory every AGENTS.md in the repository.
- Before reading or searching beneath a path, open the nearest scoped map on that path; add another map only when the investigation crosses its scope.
- If the request names an exact file, symbol, workflow, or failing test, inspect that target before searching elsewhere.
- Search progressively: likely files, then filename or import discovery, then scoped content search; expand repository-wide only when evidence proves a cross-cutting dependency.
- For pipeline bugs, trace the bounded path from task and stage contracts through the selected backend adapter and executor before changing shared abstractions.
- For optional features, verify both the intended extra-enabled path and import behavior when unrelated extras are absent.

## Operating Rules

- Use the root map plus only maps governing changed paths during ordinary work; open the protocol and manifest only for explicit review, audit, or steward maintenance.
- Generated AGENTS.md files are projections; edit `.stewards/manifest.toml`, run the projector, and commit both source and generated maps.
- Public stage, task, workflow, backend, configuration, or output-layout changes include focused regression proof and user-facing collateral or an explicit no-impact rationale.
- Keep CPU-only tests isolated from optional GPU, service, and cloud dependencies; use registered markers and shared fixtures where applicable.
- Do not broaden an invariant beyond what its proof command or evidence anchor demonstrates.
- Review triggers such as `ask stewards`, `bugbash`, `review swarm`, and `steward synthesis` open the protocol and solicit independent affected viewpoints.
- Cross-surface work records Steward Notes naming consulted maps, accepted or deferred findings, proof run, collateral, and unresolved dissent.
- Avoid unrelated refactors, speculative knobs, silent exception handling, and new suppressions unless they are part of the demonstrated fix.

## Protects (constitution)

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| The manifest, generated maps, proof anchors, coverage classifications, and active-context budgets remain internally consistent. | P1 | machine-backed | `python -m unittest discover -s tests/stewards -p 'test_*.py' -v` (`steward-tools`) |
| ProcessingStage is a public architectural boundary; incompatible changes require explicit API and downstream review. | P1 | manual | nemo_curator/stages/base.py · `class ProcessingStage` |
| Fern is the canonical source for the published NeMo Curator documentation. | P1 | manual | fern/README.md · `This directory holds the Fern MDX source` |
| Feature-family dependencies remain optional extras rather than unconditional base imports. | P1 | manual | pyproject.toml · `[project.optional-dependencies]` |

## Stop & Ask

- A change alters a public Python API, persisted task shape, stage lifecycle, backend semantics, output layout, configuration schema, or compatibility promise.
- A change adds a required dependency, changes optional-extra boundaries, or makes a CPU path require GPU or network access.
- Code and tests disagree, the reported behavior cannot be reproduced, or the intended cross-backend behavior is unresolved.
- A security, credential, destructive-data, or external-service boundary is involved without an explicit test or validation plan.
- Benchmark claims, scaling claims, or hardware requirements would be published without reproducible evidence and environment context.
- An irreversible operation, release action, external write, or coordinated downstream change is required.

## Done Criteria

- Run the narrowest affected tests first, followed by the repository-prescribed lint, type, and broader test targets in proportion to risk.
- Run `python .stewards/project.py --check` and `python .stewards/verify.py --coverage` when steward source or generated maps change.
- New GPU test files are registered in `.github/gpu_test_groups.json`; CPU-only coverage stays selectable with `-m "not gpu"`.
- User-visible behavior updates Fern docs, tutorials, examples, and release collateral where applicable, or records why each surface is unaffected.
- Performance-sensitive work names the benchmark, environment, baseline, result, and limitations; do not present an unmeasured expectation as fact.
- Cross-boundary changes include concise Steward Notes with proof and any remaining manual-verification debt.

---

Explicit review/audit only: [.stewards/PROTOCOL.md](.stewards/PROTOCOL.md). Steward maintenance only: [.stewards/manifest.toml](.stewards/manifest.toml), then `python .stewards/verify.py --coverage`.
