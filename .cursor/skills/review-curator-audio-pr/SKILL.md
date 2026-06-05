---
name: review-curator-audio-pr
description: >-
  Review someone else's NVIDIA-NeMo/Curator audio-modality pull request. This is
  a reviewer's tool: given a PR number, it locates (or shallow-clones) the repo,
  pulls the PR, diffs the changed audio code, applies Curator's audio stage
  contracts and contribution standards, shows what other reviewers have already
  raised (so you don't duplicate), and helps you produce a structured set of
  review findings (P0-P3) to post as review comments. Use when you are assigned
  or pick up an audio Curator PR and need to review it (e.g. "review audio PR
  1967", "do a review of this Curator audio PR", "what should I flag on PR
  1898"). It is NOT for a PR author responding to a review.
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

Requirements: the GitHub CLI (`gh`) authenticated against `github.com`, and
`git`. You do **not** need to pre-clone the repo - step 0 reuses any checkout
you already have and only shallow-clones
[NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator) if none is found.

## The reviewer prompt

A human reviewer invokes this skill with a prompt like:

> **"Review audio Curator PR <N> with the review-curator-audio-pr skill. Find or
> shallow-clone the repo, pull the PR and prior review activity, apply the audio
> review lenses, and give me a review digest plus my own findings (P0-P3) with
> exact `path:line` evidence and concrete fixes, ready to post as PR comments.
> Skip anything other reviewers already raised."**

Variants: *"re-review PR <N> and show only what changed since the last review"*,
*"what should I flag on the diarization PR <N>?"*, *"triage open audio PRs and
review the smallest one first"*. First-time in a repo, you can prime context
with *"build the post-#1608 audio review corpus first, then review PR <N>"*
(see [knowledge-sources.md](knowledge-sources.md) section 4).

## What you produce

You produce two things, **in this order**: (1) a **detailed overview of what the
PR does and why** - so the reader understands the change before any critique -
then (2) your **findings** (P0-P3), each tied to a `path:line` on the PR's
current head, that you post as PR review comments. Never lead with review
comments; always explain the PR first.

The skill writes two helper files into a scratch directory (default
`.curator-pr-review/`, override with `--outdir`) to support that:

1. `curator_pr<N>_fresh_review_<YYYY_MM_DD>.md` - **your working digest**: leads
   with a **"What this PR does (overview)"** section (the author's description,
   an areas-touched table, and a plain-language summary you write), then PR
   state, commits, changed-file table, and every existing review/comment grouped
   by file with OPEN / OUTDATED / RESOLVED status, then a placeholder for your
   findings. Fill in the overview first, then record your findings at the bottom.
2. `curator_pr<N>_github_comment_queue_<YYYY_MM_DD>.md` - **prior open threads**:
   a condensed list of the threads other reviewers left that are still
   unresolved on the current head. Scan it before writing your own comments.

See [templates.md](templates.md) for the exact layout and how to phrase your
review comments. The review lenses and every doc/code reference live in
[knowledge-sources.md](knowledge-sources.md).

## Workflow

```
- [ ] 0. Locate or shallow-clone the Curator repo
- [ ] 1. Identify the PR and confirm it touches audio paths
- [ ] 2. Pull fresh GitHub data (gh)
- [ ] 3. Read the diff
- [ ] 4. Review the changed code through the audio review lenses
- [ ] 5. Generate the digest + prior-threads file for context
- [ ] 6. Write the detailed PR overview (present this first)
- [ ] 7. Write up your findings (P0-P3) and post them after the overview
```

### Step 0 - Locate or shallow-clone the repo

Don't clone unnecessarily. This helper checks whether you are already inside a
Curator checkout, searches the current directory tree for one, and only then
shallow-clones (`--depth 1`, no full history):

```bash
eval "$(scripts/ensure_repo.sh | tail -1)"   # sets CURATOR_REPO=<path>
cd "$CURATOR_REPO"
```

If you are already in the checkout that contains this skill, you can skip this.

### Step 1 - Identify the PR

`gh pr view <N> --repo NVIDIA-NeMo/Curator`. Confirm the diff touches audio
paths (`nemo_curator/stages/audio/`, `nemo_curator/tasks/audio_task.py`,
`tutorials/audio/`, `tests/stages/audio/`, audio benchmarks); if it does not,
this audio skill is the wrong lens.

### Step 2 - Pull fresh GitHub data

```bash
.cursor/skills/review-curator-audio-pr/scripts/pr_review_pull.sh <N>
```

Pulls six REST endpoints (`pr view`, `reviews`, inline `comments`, issue
`comments`, `files`, `commits`) and the GraphQL review threads, which carry the
`isResolved` / `isOutdated` flags the REST inline endpoint omits. You pull this
to see prior review activity, not because you own the PR.

### Step 3 - Read the diff

```bash
gh pr diff <N> --repo NVIDIA-NeMo/Curator        # works on a shallow clone
# optional, to run/inspect locally: gh pr checkout <N>
```

`gh pr diff` fetches the patch straight from GitHub, so it works even on the
shallow clone from step 0. Always review the actual diff, never the comment text
alone. Use `main` as the baseline for "how the audio stages already do this".

### Step 4 - Review through the audio lenses

Apply the lenses in [knowledge-sources.md](knowledge-sources.md) section 2 -
each lens links the audio code, README section, and `.cursor/rules` contract it
governs: stage contracts, setup/teardown lifecycle, audio optional-dependency
hygiene (vLLM, NeMo, model utils), secret-safe logging, waveform/tensor memory
and manifest serialization, tarred/sharded audio I/O, sample-rate and metadata
propagation, streaming/throughput, tutorials/docs, tests/coverage, and PR
reviewability. Cite the linked source by name whenever the PR deviates.

### Step 5 - Generate the context files

```bash
.cursor/skills/review-curator-audio-pr/scripts/build_digest.py <N> --today <YYYY-MM-DD>
```

The builder joins the pulled JSON and classifies each existing comment by thread
status: **OPEN** (still unresolved), **OUTDATED** (pre-dates the current head),
**RESOLVED**. Use the OPEN list to avoid re-raising points another reviewer
already made.

### Step 6 - Summarize the PR in detail (present this first)

Before any critique, write the **"What this PR does (overview)"** section of the
digest so a reader understands the change end to end: the problem it solves; the
main audio stages/files it adds or modifies (use the "Areas touched" table the
builder generates); key design decisions; new dependencies, config, or APIs; and
the blast radius (what could regress). Lead your review with this overview - the
author and other reviewers read it before any findings.

### Step 7 - Write up and post your findings

Record each issue as a finding, classified by severity, and post them as a PR
review (inline comments for specific lines, a top-level summary for the overall
verdict). Each finding cites the exact `path:line-range` on the current head and
proposes a concrete fix.

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

[knowledge-sources.md](knowledge-sources.md) is the single index: canonical docs
(audio README, developer-guide slides, `.cursor/rules`, tutorials, benchmarks),
the audio code map, the review lenses with per-concept code references, the
post-#1608 PR corpus workflow, and the GitHub/`gh` data reference.
