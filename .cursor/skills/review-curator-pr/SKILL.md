---
name: review-curator-pr
description: >-
  Review NVIDIA-NeMo/Curator pull requests against the project's stage
  contracts and contribution standards. Pulls fresh PR data with the GitHub
  CLI, diffs the changed code, applies Curator's documented review lenses, and
  emits a structured review digest plus a paste-ready comment queue. Use when a
  contributor or maintainer asks to review, re-review, or triage a Curator PR
  (e.g. "review PR 1967", "what's still open on PR 1898", "triage this PR"), or
  to draft replies to reviewer comments.
---

# Review NeMo Curator PRs

Produce a defensible, reproducible review of an `NVIDIA-NeMo/Curator` PR from a
fresh GitHub pull plus the code diff. The audience is the PR author (who must
respond to reviewers and land the change) and maintainers triaging the queue.

Requirements: the GitHub CLI (`gh`) authenticated against `github.com`, and a
local checkout of this repository (the one containing this skill).

## Two deliverables (always)

Each review run writes two Markdown files into a scratch directory (default
`.curator-pr-review/`, override with `--outdir`):

1. `curator_pr<N>_fresh_review_<YYYY_MM_DD>.md` - full digest: PR state,
   commits, changed-file table, reviews, every inline comment grouped by file
   with OPEN / OUTDATED / RESOLVED status, and issue comments.
2. `curator_pr<N>_github_comment_queue_<YYYY_MM_DD>.md` - only the OPEN root
   threads that need an author reply or code change, paste-ready, with stale
   threads listed at the bottom for traceability.

See [templates.md](templates.md) for the exact section layout. Keeping the
structure stable makes diffs across review passes mechanical, so a re-review
only surfaces what changed.

## Workflow

```
- [ ] 1. Identify the PR number
- [ ] 2. Pull fresh GitHub data (gh)
- [ ] 3. Check out the PR head and diff against the base
- [ ] 4. Read the changed code through the review lenses
- [ ] 5. Generate the digest + comment queue
- [ ] 6. Classify findings P0-P3 and (optionally) draft replies
```

### Step 1 - Identify the PR

Confirm the target: `gh pr view <N> --repo NVIDIA-NeMo/Curator`. If a prior
review digest for this PR exists, read it first and record its head SHA as the
baseline so commits and comments can be marked NEW vs already-seen.

### Step 2 - Pull fresh GitHub data

Run the pull script (writes `pr<N>_*_latest.json` plus timestamped snapshots so
prior pulls are preserved for delta analysis):

```bash
.cursor/skills/review-curator-pr/scripts/pr_review_pull.sh <N>
```

It pulls six REST endpoints (`pr view`, `reviews`, inline `comments`, issue
`comments`, `files`, `commits`) and the GraphQL review threads, which carry the
`isResolved` / `isOutdated` flags the REST inline endpoint omits.

### Step 3 - Diff the code

```bash
gh pr checkout <N>                       # or: git fetch origin pull/<N>/head:pr-<N>
git diff origin/main...HEAD             # changed files on this PR
```

Always review the actual diff, never the comment text alone. Use `main` as the
baseline for "how the project already does this".

### Step 4 - Read through the review lenses

Apply the lenses in [recurring-themes.md](recurring-themes.md): stage
contracts, setup/teardown lifecycle, dependency and optional-extra hygiene,
secret-safe logging, memory/serialization of large payloads, streaming and
performance, tutorials/docs, tests/coverage, and PR size/reviewability.

These build on the project's own always-on rules in `.cursor/rules/`
(`processing-stage-patterns`, `composite-stage-patterns`, `task-patterns`,
`executors`, `resources-configuration`, `pipeline-structure`,
`modality-structure`, `coding-standards`) - treat those as the canonical
contract and cite them when a change deviates.

### Step 5 - Generate the two files

```bash
.cursor/skills/review-curator-pr/scripts/build_digest.py <N> --today <YYYY-MM-DD>
```

The builder joins the pulled JSON and classifies each comment by thread status:
**OPEN** (actionable), **OUTDATED** (pre-dates the current head), **RESOLVED**.
The comment queue lists only OPEN root threads.

### Step 6 - Findings and replies

In the digest's findings section, classify each issue by severity:

- **P0** - merge blocker: data loss / crash on a valid config, a secret leaked
  into logs, or an import that breaks a supported install.
- **P1** - fix before merge: undeclared runtime dependencies, missing required
  tests/coverage, environment-specific code in a reusable path, a PR too large
  to review safely.
- **P2** - should fix: duplicated logic, inefficient batch handling, incomplete
  metrics/observability, documented-but-unwired config.
- **P3** - nice to have: temporary workarounds that need a test + comment,
  debug-only knobs to document.

When drafting replies, cite the exact `path:line-range` on the current head,
state whether the point is already addressed (with the commit SHA) or will be,
and keep each reply self-contained. See [templates.md](templates.md) section C.

## Conventions to enforce (from CONTRIBUTING.md and rules)

- **DCO sign-off** is required on every commit (`git commit -s`).
- **Tests** must accompany source changes; GPU-only tests use
  `@pytest.mark.gpu`; the project enforces 80% coverage under `nemo_curator/`,
  and `tests/` mirrors the `nemo_curator/` layout.
- **NVIDIA Apache-2.0 copyright header** on every non-empty Python file.
- **Ruff** lint/format, 119-char lines; **loguru** for logging.
- Supported Python: 3.10-3.12.

## Knowledge sources

Full index in [reference.md](reference.md). Quick map:

| Need | Source |
|------|--------|
| Stage / pipeline contracts | `.cursor/rules/*.mdc`, `nemo_curator/stages/base.py` |
| Modality guides | `nemo_curator/stages/<modality>/README.md`, `tutorials/<modality>/README.md` |
| Backends / executors | `nemo_curator/backends/`, `nemo_curator/pipeline/pipeline.py` |
| Perf expectations | `benchmarking/` |
| Contribution rules | `CONTRIBUTING.md` |
| Live PR data | `gh` (see scripts) |
