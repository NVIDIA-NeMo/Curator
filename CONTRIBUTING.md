> [!note]
> This document is still a work in progress and may change frequently.

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

# Sync all (includes dali, deduplication_cuda12x, text, video, video_cuda)
uv sync --extra all
```

### Dev Pattern

- Sign and signoff commits with `git commit -s` (see DCO section below for details)
- If project dependencies are updated a new uv lock file needs to be generated. Run `uv lock` and add the changes of the new uv.lock file.

## Developer Certificate of Origin (DCO)

All contributions to this project must be accompanied by acknowledgment of, and agreement with, the Developer Certificate of Origin. Acknowledgment of and agreement with the DCO is done automatically by adding a `Signed-off-by` line to commit messages.

```bash
git commit -s -m "your commit message"
```

By making a contribution to this project, I certify that:

1. The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

2. The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

3. The contribution was provided directly to me by some other person who certified (1), (2) or (3) and I have not modified it.

4. I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

### How to sign your commits

When committing your changes, make sure to use the `-s` flag to automatically add your sign-off:

```bash
git commit -s -m "Add new feature"
```

This will add a `Signed-off-by` line to your commit message like:

```
Add new feature

Signed-off-by: Your Name <your.email@example.com>
```

If you forget to sign a commit, you can amend it:

```bash
git commit --amend -s
```

For multiple commits, you can use:

```bash
git rebase --signoff HEAD~N  # where N is the number of commits
```

### Testing

Work in Progress...
