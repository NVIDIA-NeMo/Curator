<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: performance

Protect reproducible performance evidence with explicit workload, environment, baseline, timeout, and result context.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

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
