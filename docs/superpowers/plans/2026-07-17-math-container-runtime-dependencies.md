# Math Container Runtime Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the math runtime's Python, native-library, container, and GPU dependencies explicit and continuously verified.

**Architecture:** The Docker image owns the native `libmagic` runtime, while the `math_cpu` extra owns the Python wrapper. Existing source-mirrored math test files gain non-mocked runtime coverage, and the PR-built image runs the extractor's real libmagic test directly.

**Tech Stack:** Docker, Ubuntu apt, uv, pytest, PyTorch CUDA, GitHub Actions

---

### Task 1: Add the native runtime contract to the extractor tests

**Files:**
- Modify: `tests/stages/math_stages/download/test_math_content_extractor.py`
- Modify: `.github/workflows/cicd-main.yml`

- [x] **Step 1: Write the failing runtime contract**

Add `test_extract_with_real_magic` to `TestMathContentExtractor`. Construct a
real plain-text record, call `MathContentExtractor.extract` without patching
`magic.Magic`, and assert the returned MIME type, text, and document type.

- [x] **Step 2: Verify the contract fails without native libmagic**

Mount the worktree at `/opt/Curator` in the rc4 container and collect the
source-mirrored test after syncing `math_cpu`. Observed: collection fails with
`ImportError: failed to find libmagic`.

- [x] **Step 3: Run the contract against the PR-built image in CI**

Add a post-build `docker run` step to `cicd-container-build` that executes
`python -m pytest` for `test_extract_with_real_magic` inside
`${{ env.container-registry }}/curator:${{ github.sha }}`.

### Task 2: Own the native and Python dependencies explicitly

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] **Step 1: Install the Ubuntu native library**

Add `libmagic1t64` to the base apt package list in `docker/Dockerfile`.

- [x] **Step 2: Verify the runtime contract passes with the native package**

Install `libmagic1t64` in the rc4 container and rerun the real extractor test.
Observed: `libmagic1t64` installs `libmagic-mgc`, and the complete math CPU
suite passes.

- [x] **Step 3: Declare the Python wrapper**

Add `python-magic==0.4.24` to `math_cpu` in `pyproject.toml`, then run
`uv lock` and verify `uv lock --check` succeeds.

### Task 3: Add real math GPU coverage in source-mirrored tests

**Files:**
- Modify: `tests/stages/math_stages/classifiers/test_finemath_classifier.py`
- Modify: `tests/stages/math_stages/modifiers/test_llm_cleanup.py`
- Modify: `tests/gpu_test_groups.json`

- [x] **Step 1: Write the GPU runtime test**

Add a `@pytest.mark.gpu` FineMath test that uses the composite classifier as a
caller would: decompose it, set up each real stage, and pass a `DocumentBatch`
through the tokenizer and CUDA model. Add a `@pytest.mark.gpu` LLM-cleanup test
that opts out of the file's autouse mocks, sets up the stage with the
repository's established small vLLM integration model, and processes a real
math `DocumentBatch`. These tests exercise public stage behavior directly;
they do not add pipeline orchestration to a source-file unit test.

- [x] **Step 2: Verify the test is selected by the GPU marker**

Run `pytest --collect-only -m gpu` for the two existing test files. Expected:
the new FineMath and LLM cleanup tests are collected.

- [x] **Step 3: Add the math GPU group**

Add `math` to `tests/gpu_test_groups.json` with extra `math_cuda12` and the
source-mirrored math test directory. Run
`python .github/scripts/check_gpu_test_coverage.py`; expected: exit 0.

### Task 4: Verify and publish

**Files:**
- Verify all modified files

- [x] **Step 1: Run focused checks**

Run Ruff on the new tests, JSON validation on `gpu_test_groups.json`, YAML
parsing on `cicd-main.yml`, lockfile validation, CPU execution of the native
runtime contract, and GPU test collection.

- [ ] **Step 2: Review the complete diff**

Compare the branch to `upstream/main`, confirm no `AGENTS.md` changes, and
check that the modified paths exist on `upstream/r1.3.0` for cherry-picking.

- [ ] **Step 3: Commit and publish**

Commit the scoped files, push `agent/fix-math-container-dependencies` to the
fork, and open a draft pull request against `NVIDIA-NeMo/Curator:main` with
root cause, validation evidence, and cherry-pick guidance.
