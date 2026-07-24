<!-- generated steward map; edit source layers, not this file -->
<!-- source layers: .stewards/layers/backends.toml -->

> **Guidance owners:** @oyilmaz-nvidia, @praateekmahajan, @abhinavg4, @ayushdg. Update `.stewards/layers/backends.toml`.
> Regenerate with `python .stewards/project.py`, then run `python .stewards/verify.py --coverage`.

# Steward: backends

Protect backend-neutral results while preserving each executor's lifecycle, resource, failure, and resume semantics.

Ordinary work: use this map directly with the root map and run only affected checks.
Open `.stewards/` only for explicit review, audit, or steward maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| The canonical backend integration fixture produces equivalent results for the backend configurations it parametrizes. | P1 | machine-backed | `uv run pytest tests/backends/test_integration.py -q -m 'not gpu'` (`backend-sample`) |
| The functional resume loop continues from persisted task state for the scenarios exercised by its fixture. | P1 | machine-backed | `uv run pytest tests/backends/test_resumability_functional.py -q -m 'not gpu'` (`resume-sample`) |
| Ray adapters communicate actor and fanout stage traits through RayStageSpecKeys. | P1 | manual | nemo_curator/backends/utils.py · `class RayStageSpecKeys` |

## Guardrails

- A focused integration sample is evidence for that sample, not proof of universal backend parity.

## Edges

- depends-on → **pipeline** (stage descriptors and task lineage)
- serves → **performance** (executor behavior measured by benchmarks)

## Owns

- **code:** `nemo_curator/backends`
- **tests:** `tests/backends`

## Advocate

- Make retry, resume, failure, and task-lineage behavior observable in focused tests.

## Do Not

- Do not claim parity beyond the backends and cases exercised by evidence.

## Serves

- Pipeline authors running on Xenna or Ray Data. Ray Actor Pool guidance remains scoped to deduplication.
