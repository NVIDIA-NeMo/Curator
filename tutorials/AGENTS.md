<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: tutorials

Protect runnable, dependency-explicit examples that teach current public APIs without hidden environment assumptions.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| Tutorial dependency instructions use optional-extra names declared by the package. | P1 | manual | pyproject.toml · `text_cpu = [` |
| Tutorial examples use current public imports and state their data, network, credential, and hardware prerequisites. | P2 | none | — |

## Guardrails

- Tutorial setup names the relevant optional extra and does not rely on undeclared local state.
- Run Ruff on changed tutorial Python without implying that tutorial execution is covered by the unit-test matrix.

## Edges

- teaches → **pipeline** (public workflow construction)
- collaborates-with → **docs** (published conceptual guidance)

## Owns

- **docs:** `tutorials`
