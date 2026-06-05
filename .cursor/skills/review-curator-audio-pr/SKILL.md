---
name: review-curator-audio-pr
description: >-
  Review someone else's NVIDIA-NeMo/Curator audio-modality pull request. This is
  a reviewer's tool: given a PR number, it pulls the PR, diffs the changed audio
  code, applies Curator's audio stage contracts and contribution standards,
  shows what other reviewers have already raised (so you don't duplicate), and
  helps you produce a structured set of review findings (P0-P3) to post as
  review comments. Use when you are assigned or pick up an audio Curator PR and
  need to review it (e.g. "review audio PR 1967", "do a review of this Curator
  audio PR", "what should I flag on PR 1898"). It is NOT for a PR author
  responding to a review.
---

# Review NeMo Curator audio PRs

**Use this when you are the reviewer and need to review someone else's audio
Curator PR.** Given a PR number, it helps you understand the change, see what
other reviewers already said, and write your own review. It does not help a PR
author reply to reviewers - it helps you *produce* the review.

Scope: the **audio** modality - code under `nemo_curator/stages/audio/`,
`tutorials/audio/`, audio tests under `tests/stages/audio/`, and audio
benchmarks. For non-audio PRs the stage-contract lenses still apply, but the
audio-specific guidance below will not.

Requirements: the GitHub CLI (`gh`) authenticated against `github.com`, and a
local checkout of this repository (the one containing this skill).

## What you produce

Your review = a set of **findings** (P0-P3), each tied to a `path:line` on the
PR's current head, that you post as PR review comments. The skill writes two
helper files into a scratch directory (default `.curator-pr-review/`, override
with `--outdir`) to support that:

1. `curator_pr<N>_fresh_review_<YYYY_MM_DD>.md` - **your working digest**: PR
   state, commits, changed-file table, and every existing review/comment grouped
   by file with OPEN / OUTDATED / RESOLVED status. Read it to understand the PR
   and to see what other reviewers already raised so you don't repeat them; add
   your own findings at the bottom.
2. `curator_pr<N>_github_comment_queue_<YYYY_MM_DD>.md` - **prior open threads**:
   a condensed list of the threads other reviewers left that are still
   unresolved on the current head. Scan it before writing your own comments.

See [templates.md](templates.md) for the exact layout, including how to write up
your findings.

## Workflow

```
- [ ] 1. Identify the PR and confirm it touches audio paths
- [ ] 2. Pull fresh GitHub data (gh)
- [ ] 3. Check out the PR head and read the diff
- [ ] 4. Review the changed code through the audio review lenses
- [ ] 5. Generate the digest + prior-threads file for context
- [ ] 6. Write up your findings (P0-P3) and post them as review comments
```

### Step 1 - Identify the PR

`gh pr view <N> --repo NVIDIA-NeMo/Curator`. Confirm the diff touches audio
paths (`nemo_curator/stages/audio/`, `tutorials/audio/`, `tests/stages/audio/`,
`benchmarking/AUDIO_PROFILING.md` / `ALM_BENCHMARK.md`); if it does not, this
audio skill is the wrong lens.

### Step 2 - Pull fresh GitHub data

Run the pull script (writes `pr<N>_*_latest.json` plus timestamped snapshots):

```bash
.cursor/skills/review-curator-audio-pr/scripts/pr_review_pull.sh <N>
```

It pulls six REST endpoints (`pr view`, `reviews`, inline `comments`, issue
`comments`, `files`, `commits`) and the GraphQL review threads, which carry the
`isResolved` / `isOutdated` flags the REST inline endpoint omits. You pull this
so you can see prior review activity, not because you own the PR.

### Step 3 - Read the diff

```bash
gh pr checkout <N>                       # or: git fetch origin pull/<N>/head:pr-<N>
git diff origin/main...HEAD             # the change you are reviewing
```

Always review the actual diff, never the comment text alone. Use `main` as the
baseline for "how the audio stages already do this".

### Step 4 - Review through the audio lenses

Apply the lenses in [recurring-themes.md](recurring-themes.md): stage contracts,
setup/teardown lifecycle, audio optional-dependency hygiene (vLLM, NeMo, model
utils), secret-safe logging, waveform/tensor memory and manifest serialization,
tarred/sharded audio I/O, sample-rate and metadata propagation,
streaming/throughput, tutorials/docs, tests/coverage, and PR reviewability.

These build on the project's always-on rules in `.cursor/rules/`
(`processing-stage-patterns`, `composite-stage-patterns`, `task-patterns`,
`executors`, `resources-configuration`, `pipeline-structure`,
`modality-structure`, `coding-standards`) and the audio stage guide at
`nemo_curator/stages/audio/README.md`. Cite these when the PR deviates.

### Step 5 - Generate the context files

```bash
.cursor/skills/review-curator-audio-pr/scripts/build_digest.py <N> --today <YYYY-MM-DD>
```

The builder joins the pulled JSON and classifies each existing comment by thread
status: **OPEN** (still unresolved), **OUTDATED** (pre-dates the current head),
**RESOLVED**. Use the OPEN list to avoid re-raising points another reviewer
already made.

### Step 6 - Write up and post your findings

Record each issue you found as a finding, classified by severity, and post them
as a PR review (inline comments for specific lines, a top-level summary for
overall verdict). Each finding should cite the exact `path:line-range` on the
current head and propose a concrete fix.

- **P0** - merge blocker: data loss / crash on a valid audio config (e.g. a
  manifest writer that crashes on a kept waveform), a secret (HF token) leaked
  into logs, or an import that breaks a supported install.
- **P1** - should fix before merge: undeclared audio runtime dependencies,
  missing required tests/coverage, environment-specific I/O in a reusable
  reader, a PR too large to review safely.
- **P2** - worth fixing: duplicated pipeline construction, inefficient per-task
  writer batching, incomplete throughput metrics, documented-but-unwired config.
- **P3** - nice to have: temporary model/compat workarounds needing a test +
  comment, debug-only knobs (e.g. per-shard utterance caps) to document.

See [templates.md](templates.md) section C for how to phrase review comments.

## What to check the PR against (CONTRIBUTING.md and rules)

A compliant audio PR must satisfy these; flag any that are missing:

- **DCO sign-off** on every commit (`git commit -s`).
- **Tests** accompany source changes; GPU-only audio tests use
  `@pytest.mark.gpu`; the project enforces 80% coverage under `nemo_curator/`,
  and `tests/stages/audio/` mirrors `nemo_curator/stages/audio/`.
- **NVIDIA Apache-2.0 copyright header** on every non-empty Python file.
- **Ruff** lint/format, 119-char lines; **loguru** for logging.
- Supported Python: 3.10-3.12.

## Knowledge sources

Full index in [reference.md](reference.md). Quick map:

| Need | Source |
|------|--------|
| Audio stage guide | `nemo_curator/stages/audio/README.md` |
| Stage / pipeline contracts | `.cursor/rules/*.mdc`, `nemo_curator/stages/base.py` |
| Audio tutorials | `tutorials/audio/` |
| Backends / executors | `nemo_curator/backends/`, `nemo_curator/pipeline/pipeline.py` |
| Audio perf expectations | `benchmarking/AUDIO_PROFILING.md`, `benchmarking/ALM_BENCHMARK.md` |
| Contribution rules to check against | `CONTRIBUTING.md` |
| Live PR data | `gh` (see scripts) |
