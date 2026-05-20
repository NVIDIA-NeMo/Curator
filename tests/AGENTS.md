# Steward: Tests

The test suite. CPU-default with explicit GPU markers. The conftest
sets up a smart Ray cluster that adapts to available GPUs. CI runs via
`L0_Unit_Test_CPU.sh` and `L0_Unit_Test_GPU.sh`. Pre-commit checks
plus 80% coverage on changes are gate conditions.

Related docs:

- root [AGENTS.md](../AGENTS.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md) — unit test / coverage
  policy
- `.cursor/rules/coding-standards.mdc`

## Point Of View

The safety net. Defends the parity, fault-tolerance, and ABI claims
the rest of the repo makes. Without consistent test conventions, every
modality drifts independently and parity testing across backends
collapses.

## Protect

- **`@pytest.mark.gpu`**: every test that requires CUDA carries this
  marker. CPU-only runs (`pytest -m "not gpu"`) must pass on a clean
  install without GPU libs.
- **`conftest.py` cluster lifecycle**: tests share a single Ray
  cluster configured by `tests/conftest.py` (session-scoped
  `shared_ray_cluster` fixture). Do not start ad-hoc Ray clusters
  inside tests.
- **Modality selection rules**: `pytest_ignore_collect` in
  `tests/conftest.py` enforces one modality per `-m` flag, forbids
  negation, and rejects multi-modality selection. Don't try to work
  around this; if you need cross-modality runs, do separate
  invocations.
- **GPU detection**: GPU presence is detected via pynvml /
  `nvidia-smi`. Tests gracefully skip when no GPU; they must not
  hard-fail.
- **Test-source layout mirrors `nemo_curator/`**. New code under
  `nemo_curator/<area>/` gets tests under `tests/<area>/`.
- **CI scripts** (`L0_Unit_Test_CPU.sh`, `L0_Unit_Test_GPU.sh`) and
  `gpu_test_groups.json` are the canonical CI entrypoints; keep
  them in sync with the test tree.
- **80% coverage on changed lines** (per `CONTRIBUTING.md`).
  `codecov.yml` enforces this in CI.
- **Determinism**: avoid wall-clock and randomness in tests unless
  the random source is explicitly seeded.
- **Fixtures and data** (`tests/fixtures/`, `tests/data/`): keep
  small, license-clean, and committed (no large binaries).

## Contract Checklist

When this domain changes:

- `tests/conftest.py` — Ray cluster lifecycle, GPU detection,
  marker registration
- `tests/L0_Unit_Test_CPU.sh`, `tests/L0_Unit_Test_GPU.sh`,
  `tests/gpu_test_groups.json`
- `tests/fixtures/`, `tests/data/`
- `pyproject.toml` `[tool.pytest.ini_options]` — markers, addopts
- `codecov.yml` — coverage thresholds
- `.github/workflows/` — CI pipelines that invoke the L0 scripts
- `CONTRIBUTING.md` — unit test / coverage policy
- `.cursor/rules/coding-standards.mdc`
- The "Testing Guidelines" section of
  `.github/copilot-instructions.md`

## Advocate

- Backend-parity test fixtures so executor sweeps are first-class
  instead of hand-coded per test.
- A canonical set of small fixtures per modality so new tests do not
  reinvent fixture loading.
- Coverage reporting per-module so low-coverage hotspots are visible.
- Faster CPU CI by selective test selection on changed-paths (CODEOWNERS
  / pytest collect).
- Snapshot or determinism tests for deduplication and synthetic-data
  outputs.

## Serve Peers

- **To pipeline-contract steward**: provide minimal stage / task
  fixtures so contract tests do not depend on real modality stages.
- **To backends steward**: own the parity test harness so executor
  sweeps are trivial to add.
- **To modality stewards**: keep `tests/data/` and `tests/fixtures/`
  organized so each modality can pull representative inputs.
- **To docs steward**: tests are the verification surface for doc
  claims. When a content audit lands a P0 about a flag/default,
  there should be a test that pins it.

## Do Not

- Add a test that requires a GPU without `@pytest.mark.gpu`.
- Start ad-hoc Ray clusters; reuse the conftest's cluster.
- Commit large binaries or model artifacts.
- Use real network calls (HuggingFace download, remote APIs) inside
  CPU tests; mock or skip.
- Disable coverage with `# pragma: no cover` for non-trivial logic.
- Use `print()`; tests should be quiet on success.
- Set `RAY_memory_usage_threshold` in test code; conftest already
  sets a reasonable default.

## Own

**Code surfaces**:

- `tests/` (entire tree)
- `tests/conftest.py`, `tests/L0_Unit_Test_*.sh`,
  `tests/gpu_test_groups.json`

**Tests**: self.

**Docs (autopilot audit surface)**:

- The "Unit tests" and "Coverage" sections of `CONTRIBUTING.md`
- The "Testing Guidelines" section of
  `.github/copilot-instructions.md`
- `fern/` developer / contributor pages about testing

**Agent-facing artifacts**:

- The "Testing" portion of `.cursor/rules/coding-standards.mdc`

**CODEOWNERS routing**:

- Default: `@NVIDIA-NeMo/curator_reviewers`
- Per-domain subtrees route to their modality CODEOWNERS where
  defined: `tests/stages/deduplication/`,
  `tests/stages/text/embedders/`, `tests/stages/text/classifiers/`,
  `tests/stages/synthetic/`, `tests/stages/video/`, `tests/backends/`.
  Other subtrees (`tests/stages/{audio,image,interleaved,math_stages,common}/`,
  `tests/{tasks,pipelines,core,metrics,models,config,utils}/`) fall
  through to the default reviewers team today; add explicit
  CODEOWNERS entries when those modalities grow dedicated owners.
