<!-- generated steward map; edit source layers, not this file -->
<!-- source layers: .stewards/layers/tests.toml -->

> **Guidance owners:** @NVIDIA-NeMo/curator_reviewers. Update `.stewards/layers/tests.toml`.
> Regenerate with `python .stewards/project.py`, then run `python .stewards/verify.py --coverage`.

# Steward: tests

Protect deterministic, correctly marked, dependency-isolated tests and shared infrastructure that reproduces user contracts.

Ordinary work: use this map directly with the root map and run only affected checks.
Open `.stewards/` only for explicit review, audit, or steward maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| Every GPU test file is assigned to a registered GPU test group. | P1 | machine-backed | `python .github/scripts/check_gpu_test_coverage.py` (`gpu-test-coverage`) |
| The gpu marker remains registered so CPU-only selection can deselect GPU tests. | P1 | manual | pyproject.toml · `gpu: marks tests as GPU tests` |
| Tests use the shared Ray cluster fixture by default; isolated cluster ownership is reserved for lifecycle behavior under test. | P2 | manual | tests/conftest.py · `def shared_ray_cluster` |
| Tests requiring credentials, network services, or special hardware declare and isolate those requirements. | P1 | none | — |

## Guardrails

- Prefer shared Ray fixtures by default; isolated lifecycle tests may own a cluster when that ownership is the behavior under test.
- Lightweight Hugging Face downloads are acceptable; mock or skip heavier remote API calls, and coordinate token-gated CI with the automation team.

## Edges

- verifies → **pipeline** (stage and task contracts)
- verifies → **backends** (executor and resume behavior)

## Owns

- **tests:** `tests`, `.github/gpu_test_groups.json`, `.github/scripts/check_gpu_test_coverage.py`
