<!-- generated steward map; edit source layers, not this file -->
<!-- source layers: .stewards/layers/text.toml -->

> **Guidance owners:** @sarahyurick, @praateekmahajan, @VibhuJawa. Update `.stewards/layers/text.toml`.
> Regenerate with `python .stewards/project.py`, then run `python .stewards/verify.py --coverage`.

# Steward: text

Protect text batch shape and the composability of classifiers, filters, embedders, I/O, and download stages.

Ordinary work: use this map directly with the root map and run only affected checks.
Open `.stewards/` only for explicit review, audit, or steward maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| Text pipeline stages preserve the DocumentBatch table/data-frame task boundary unless a public migration is approved. | P1 | manual | nemo_curator/tasks/document.py · `class DocumentBatch` |
| Classifier, embedder, download, I/O, filter, and modifier concerns remain independently composable where their public stages permit it. | P2 | none | — |

## Guardrails

- When tokenization and model inference have distinct CPU/GPU scaling or persistent model state, use separate stages composed together unless the implementation documents why coupling is required.

## Edges

- depends-on → **pipeline** (DocumentBatch and processing-stage contracts)
- collaborates-with → **deduplication** (text duplicate-removal workflows)

## Owns

- **code:** `nemo_curator/stages/text`
- **tests:** `tests/stages/text`
