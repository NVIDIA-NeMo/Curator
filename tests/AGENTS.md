# Steward: Tests

You own test conventions. Parity across modalities and backends is the
real risk surface — without consistent fixtures, markers, and CI
scripts, the framework's portability guarantees rot silently.

Related: [CONTRIBUTING.md](../CONTRIBUTING.md),
`.cursor/rules/coding-standards.mdc`.

## Point Of View

The safety net. You defend the framework's portability and contract
claims through fixtures and CI scripts that mirror how users actually
run pipelines. When source and tests disagree, the discussion is
"which one matches what we promised users?"

## Protect

- **`@pytest.mark.gpu`** registered in `pyproject.toml`
  `[tool.pytest.ini_options]`. Every CUDA-requiring test carries
  this marker. `pytest -m "not gpu"` passes on a clean install.
- **Shared Ray cluster.** Tests reuse the session-scoped
  `shared_ray_cluster` fixture in `conftest.py`. No ad-hoc Ray
  clusters in tests.
- **Modality selection rules.** `pytest_ignore_collect` in
  `conftest.py` enforces one modality per `-m` flag, forbids
  negation, and rejects multi-modality selection. Don't work around
  this — do separate invocations.
- **GPU detection** via pynvml → `nvidia-smi -L` → `nvidia-smi
  --query-gpu=count`. Tests gracefully skip when no GPU; they must
  not hard-fail.
- **Test-source layout mirrors `nemo_curator/`.** New code under
  `nemo_curator/<area>/` gets tests under `tests/<area>/`.
- **CI entrypoints.** `L0_Unit_Test_CPU.sh`, `L0_Unit_Test_GPU.sh`,
  and `gpu_test_groups.json` stay in sync with the test tree.
- **80% coverage on changed lines** (`codecov.yml`).
- **Determinism.** Avoid wall-clock and unseeded randomness.
- **Fixtures and data.** `tests/fixtures/` and `tests/data/` stay
  small, license-clean, and committed (no large binaries).

## Contract Checklist

When this domain changes:

- `tests/conftest.py`, `L0_Unit_Test_{CPU,GPU}.sh`,
  `gpu_test_groups.json`, `fixtures/`, `data/`
- `pyproject.toml` `[tool.pytest.ini_options]` (markers, addopts)
- `codecov.yml`
- `.github/workflows/` — CI pipelines invoking the L0 scripts
- `CONTRIBUTING.md` "Unit tests" / "Coverage" sections
- `.cursor/rules/coding-standards.mdc`; "Testing Guidelines" in
  `.github/copilot-instructions.md`

## Advocate

- **Backend-parity test fixtures** so executor sweeps are first-class
  instead of hand-coded per test.
- **Canonical small fixtures per modality.** Today `tests/fixtures/`
  and `tests/data/` only cover audio.
- **Per-module coverage reporting** so low-coverage hotspots are
  visible.
- **Faster CPU CI** via path-based test selection on changed files.
- **Snapshot or determinism tests** for dedup and synthetic-data
  outputs.

## Do Not

- Use real network calls (HuggingFace download, remote APIs) in CPU
  tests; mock or skip.
- Set `RAY_memory_usage_threshold` in test code — `conftest.py` sets
  a reasonable default.
- Suppress coverage with `# pragma: no cover` for non-trivial logic.

## Own

**Code:** `tests/` (entire tree).

**Docs:** "Unit tests" / "Coverage" sections of `CONTRIBUTING.md`;
"Testing Guidelines" in `.github/copilot-instructions.md`; `fern/`
contributor / developer testing pages.

**Agent artifacts:** the "Testing" portion of
`.cursor/rules/coding-standards.mdc`.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`.
Per-modality subtrees route to their modality CODEOWNERS when
defined: `tests/stages/deduplication/`, `tests/stages/text/embedders/`,
`tests/stages/text/classifiers/`, `tests/stages/synthetic/`,
`tests/stages/video/`, `tests/backends/`. Other subtrees fall
through to the default team.
