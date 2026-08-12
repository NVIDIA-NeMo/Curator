# Contributing to NeMo Curator

We're glad you're contributing to NeMo Curator. This guide explains what kinds of contributions we accept, where to ask questions, how to find a good first issue, and how to set up your environment and open a PR.

All contributions are accepted under the project's [Apache License 2.0](LICENSE). All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Your First Contribution](#your-first-contribution)
- [Asking Questions and Discussing Changes](#asking-questions-and-discussing-changes)
- [Code of Conduct](#code-of-conduct)
- [General Principles](#general-principles)
- [Python Style](#python-style)
- [Setup and Dev](#setup-and-dev)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Unit Tests](#unit-tests)
- [Coverage](#coverage)
- [Pull Requests (PR) Guidelines](#pull-requests-pr-guidelines)
  - [Updating Package Dependencies](#updating-package-dependencies)

## Ways to Contribute

Contributions of any size are welcome. Common types:

- **Bug fixes** — reproducible issues with a failing test where practical.
- **Documentation** — fixes, clarifications, and new tutorials in `fern/` (the canonical docs source) and the in-repo `tutorials/` directory.
- **Tests** — extending unit, integration, or GPU test coverage; converting flaky tests into deterministic ones.
- **New pipeline stages** — modality-aware filters, classifiers, dedupers, loaders, or writers that fit the existing `Stage`/`Pipeline` abstractions.
- **Recipes and examples** — end-to-end curation workflows that exercise real datasets.
- **Performance work** — benchmarks, scaling improvements, or executor enhancements (please include before/after numbers).
- **Build, CI, packaging, and dependency hygiene.**

If you're planning a larger change (new modality, new executor, breaking API change), please open a [Discussion](https://github.com/NVIDIA-NeMo/Curator/discussions) or a draft issue first so we can align on direction before you invest in code.

## Your First Contribution

New to the project? Start here:

1. Browse issues labeled [`good first issue`](https://github.com/NVIDIA-NeMo/Curator/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — these are scoped to be approachable without deep familiarity with the codebase.
2. Comment on the issue to claim it before you start work, so we don't duplicate effort.
3. Read [General Principles](#general-principles) and follow the [PR Guidelines](#pull-requests-pr-guidelines) below.

Good starter areas if no labeled issue catches your eye: docs typos and clarifications, missing docstrings, additional unit-test coverage, and small isolated Python-stage fixes.

## Asking Questions and Discussing Changes

| You want to… | Where to go |
|--------------|-------------|
| Ask a "how do I…" or "is this supported?" question | [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions) |
| Propose a non-trivial feature or design change | Open a Discussion or a draft Issue describing the design first |
| Report a reproducible bug | [GitHub Issues](https://github.com/NVIDIA-NeMo/Curator/issues) using the bug template |
| Show off a recipe or pipeline | Discussions → "Show and tell" |

These are community channels staffed on a best-effort basis; there is no support SLA. Please avoid opening Issues for usage questions — keep Issues focused on bugs and concrete proposals so the backlog stays tractable.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating — in issues, discussions, PRs, or any other project space — you agree to abide by its terms. To report unacceptable behavior, follow the reporting instructions in `CODE_OF_CONDUCT.md`.

## General Principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo Curator supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Python Style
We use ``ruff`` as our style guide. To fix your format run `pre-commit install && pre-commit run --all`.

**Setting up pre-commit hooks locally (optional)** (one-time setup):
```bash
pip install pre-commit
pre-commit install --install-hooks
```

To manually run all checks: `pre-commit run --all-files`

1. Include docstrings for every class and method exposed to the user.
1. Loggers are preferred to print.

## Setup and Dev

### Prerequisites

- Python >=3.11, < 3.14
- OS: Ubuntu 22.04/20.04
- NVIDIA GPU (optional)
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12.x
- uv

```
# We use `uv` for package management and environment isolation.
pip3 install uv

# If you cannot install at the system level, you can install for your user with
pip3 install --user uv
```

### Installation

NeMo Curator uses [uv](https://docs.astral.sh/uv/) for package management.

You can configure uv with the following commands:

```bash
uv sync
```

You can additionally sync optional dependency groups:

```bash
uv sync --extra text

# Sync multiple dependency groups
uv sync --extra text --extra video

# Sync all (includes audio_cuda12, deduplication_cuda12, image_cuda12, text_cuda12, video_cuda12)
uv sync --extra all
```

- If project dependencies are updated, the lock file needs to be regenerated. See [Updating Package Dependencies](#updating-package-dependencies) for the full workflow.

## Unit Tests
Unit tests should be simple and fast.
Developers should be able to run them frequently while developing without any slowdown.
```
pytest
# If you don't have NVIDIA GPU do:
# pytest -m 'not gpu'
```

## Coverage
Pull requests should cover at least 80% of its changes with tests. CI will reject PRs that do not fulfill this requirement. Please refer to the [Unit tests](#unit-tests) section for more about writing unit tests.

## Pull Requests (PR) Guidelines

**Send your PRs to the `main` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
2) Read General Principles and style guide above
3) Ensure that your environment is set up for signing commits. This [GitHub doc](https://docs.github.com/en/authentication/managing-commit-signature-verification) contains all the information about setting up commit signing.
    - [This doc](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits) has more details about how you can sign commits and has links with instructions to set up keys for commit signing.
4) Make sure you sign your commits. E.g. use ``git commit -sS`` when committing.
    1) If you forget to do this, please follow the steps below to undo the commits and reapply the changes under a new (signed and signed-off) commit. Note: This will preserve your changes, but delete the git history of commits.
    ```bash
    git reset --soft HEAD~N
    git add <insert all files you want to include>
    git commit -sS -m "My commit message"
    git push --force
    ```
    Replace `N` in the first line with the number of commits you want to undo. To undo the latest commit, do `git reset --soft HEAD~1`.
4) Make sure all unittests finish successfully before sending PR ``pytest`` or (if your dev box does not have GPU) ``pytest --cpu`` from the root folder
5) Send your PR and request a review

### Updating Package Dependencies

When you modify dependencies in `pyproject.toml`, you need to regenerate the lock file to keep it in sync:

```bash
# Regenerate uv.lock
uv lock

# Stage and commit dependency files
git add pyproject.toml uv.lock
git commit -s -m "Update dependencies"
```

**Pre-commit hooks**: This repository has a pre-commit hook (`uv-lock`) that checks if the lock file is in sync. If you have pre-commit installed locally and the lock file is out of sync, the hook will:
1. Generate the updated lock file
2. Block the commit (showing "files were modified by this hook")
3. You then need to stage the generated file and commit again

**Workflow**:
1. Modify dependencies in `pyproject.toml`
2. Either:
   - **Option A (Manual)**: Run `uv lock` before committing
   - **Option B (Let hooks do it)**: Just try to commit - the hook will generate the file and block, then stage and commit again
3. Stage files: `git add pyproject.toml uv.lock`
4. Commit with sign-off: `git commit -s -m "Your message"`

> **Note**: If you encounter issues with the pre-commit hook (e.g., `uv` not installed or platform-specific problems), you can bypass it with `git commit --no-verify`. The CI will still verify the lock file is in sync.

Unit tests are expected to pass before merging into `main`.
Every release a new branch will be cut from `main`.

Full text of the DCO:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
