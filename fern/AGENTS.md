<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: docs

Protect Fern as the canonical published documentation source and keep claims synchronized with tested behavior.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| Published NeMo Curator documentation changes are authored in the Fern MDX source tree. | P1 | manual | fern/README.md · `docs.nvidia.com/nemo/curator` |
| API, backend, hardware, and performance claims identify the implementation or reproducible evidence that supports them. | P2 | none | — |

## Guardrails

- Do not edit generated or legacy documentation as a substitute for the canonical Fern source.

## Edges

- documents → **pipeline** (public concepts and API behavior)
- collaborates-with → **tutorials** (runnable learning paths)

## Owns

- **docs:** `fern`

## Advocate

- Write from verified source behavior and include the setup needed to reproduce it.

## Do Not

- Do not publish benchmark or hardware claims without linked evidence context.

## Serves

- New adopters, existing users, and maintainers reviewing public promises.
