<!-- generated steward map; edit source layers, not this file -->
<!-- source layers: .stewards/layers/performance.toml -->

> **Guidance owners:** @rlratzel, @praateekmahajan, @sarahyurick, @ayushdg. Update `.stewards/layers/performance.toml`.
> Regenerate with `python .stewards/project.py`, then run `python .stewards/verify.py --coverage`.

# Steward: performance

Protect reproducible performance evidence with explicit workload, environment, baseline, timeout, and result context.

Ordinary work: use this map directly with the root map and run only affected checks.
Open `.stewards/` only for explicit review, audit, or steward maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| Benchmark sessions preserve the tested default maximum timeout behavior. | P2 | machine-backed | `uv run pytest tests/benchmarking/runner/test_session.py -q -m 'not gpu'` (`benchmark-session`) |
| Reported benchmark results include workload, environment, hardware, software versions, baseline, and limitations. | P1 | none | — |

## Guardrails

- Benchmark expectations are proposals until a reproducible run records evidence.

## Edges

- measures → **backends** (executor scalability and resource behavior)
- measures → **pipeline** (stage and workflow performance)

## Owns

- **code:** `benchmarking`
- **tests:** `tests/benchmarking`
