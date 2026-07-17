<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: deduplication

Protect stable identity, deterministic duplicate decisions, and the distinct exact, fuzzy, and semantic workflow contracts.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| Exact, fuzzy, and semantic deduplication remain distinct workflows with independently reviewed contracts. | P1 | manual | nemo_curator/stages/deduplication/exact/workflow.py · `class ExactDeduplicationWorkflow` |
| Duplicate decisions depend on stable ID generation; algorithm or input-normalization changes require migration review. | P1 | manual | nemo_curator/stages/deduplication/id_generator.py · `class IdGeneratorBase` |
| GPU deduplication dependencies remain isolated in the deduplication_cuda12 optional extra. | P1 | manual | pyproject.toml · `deduplication_cuda12 = [` |
| Ray Actor Pool workflow guidance remains scoped to deduplication; proposals to broaden it require backend and product review. | P2 | none | — |

## Guardrails

- Do not collapse workflow-specific prerequisites or output semantics into a generic dedup claim.

## Edges

- depends-on → **pipeline** (workflow and task contracts)
- depends-on → **backends** (distributed execution and resume behavior)

## Owns

- **code:** `nemo_curator/stages/deduplication`
- **tests:** `tests/stages/deduplication`
