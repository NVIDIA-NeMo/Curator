# Checklist

We are glad you are contributing to NeMo Curator! Before you make a PR, be sure to read over this guide in detail.
This checklist ensures that NeMo Curator stays easy-to-use by both users and developers.
Not all steps are necessary for some contributions, so read the linked sections for more information about each item.

- [Checklist](#checklist)
  - [General principles](#general-principles)
  - [Python style](#python-style)
  - [Setup and Dev](#setup-and-dev)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Unit tests](#unit-tests)
  - [Coverage](#coverage)
  - [Pull Requests (PR) Guidelines](#pull-requests-pr-guidelines)

## General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo Curator supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Python style
We use ``ruff`` as our style guide. To fix your format run `pre-commit install && pre-commit run --all`.

1. Include docstrings for every class and method exposed to the user.
1. Loggers are preferred to print.

## Setup and Dev

### Prerequisites

- Python >=3.10, < 3.13
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

- If project dependencies are updated a new uv lock file needs to be generated. Run `uv lock` and add the changes of the new uv.lock file.

## Unit tests
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
