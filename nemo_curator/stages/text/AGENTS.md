<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: text

Protect text batch shape, lazy optional imports, and the composability of classifiers, filters, embedders, I/O, and download stages.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| The text classifier namespace exports its lazy registry without eagerly initializing optional classifier implementations. | P1 | machine-backed | `uv run pytest tests/stages/text/classifiers/test_classifiers_init.py -q -m 'not gpu'` (`text-registry`) |
| Text pipeline stages preserve the DocumentBatch table/data-frame task boundary unless a public migration is approved. | P1 | manual | nemo_curator/tasks/document.py · `class DocumentBatch` |
| Classifier, embedder, download, I/O, filter, and modifier concerns remain independently composable where their public stages permit it. | P2 | none | — |

## Guardrails

- Keep classifier registration lazy so importing the namespace does not initialize optional frameworks.

## Edges

- depends-on → **pipeline** (DocumentBatch and processing-stage contracts)
- collaborates-with → **deduplication** (text duplicate-removal workflows)

## Owns

- **code:** `nemo_curator/stages/text`
- **tests:** `tests/stages/text`
