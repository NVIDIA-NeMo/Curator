# Output templates

You are the reviewer. `scripts/build_digest.py` generates files A and B as
context; you author the findings (section C) and post them as your review.
The digest **leads with a PR overview** that you must complete and present
before any review comments. Replace `<...>` placeholders. Keep the structure
stable across review passes so a re-review diffs cleanly and only surfaces what
changed.

## A. Working digest (generated, then you append findings)

File: `curator_pr<N>_fresh_review_<YYYY_MM_DD>.md`. It **leads with a PR
overview** (what the change does and why), then PR state, the diff, and all
existing review activity. Complete the overview's plain-language summary first
and present it before any findings; add your **Findings** section at the end.

````markdown
# Curator PR <N> Fresh Review - <YYYY-MM-DD>

Review target: https://github.com/NVIDIA-NeMo/Curator/pull/<N>

Current PR head reviewed: `<headRefOid>`

Base recorded by GitHub metadata: `<baseRefOid>`

## What this PR does (overview)

**<title>** - by @<author>  (<state>, +<a>/-<d> across <n> files, <k> commits)

### Author's description
<verbatim PR body>

### Areas touched
| Area | files | +/- |
|---|---|---|
| stages/audio/<area> | <n> | +<a> -<d> |

### Plain-language summary (write this BEFORE any findings)
<In your own words, after reading the knowledge sources: what the PR changes and
why, the main stages/files, key design decisions, new deps/config/APIs, and
blast radius.>

## PR state at review time

| Field | Value |
|---|---|
| state | <OPEN/CLOSED/MERGED> |
| reviewDecision | <APPROVED/CHANGES_REQUESTED/none> |
| changedFiles | <n> |
| additions / deletions | +<a> / -<d> |
| commits | <n> |
| inline review comments total | <n> |
| reviews submitted | <n> |
| updatedAt | <iso> |

## Commits (<k>)

```
<sha8>  <subject>  (<date>)
```

Files changed (`git diff <base>..<head>`):

| File | +/- | status |
|---|---|---|
| `<path>` | +<a> -<d> | <added/modified> |

## Existing reviews (by other reviewers)

### #<id> by @<login>  state=<STATE>  commit=<sha8>  submitted=<iso>
<body or empty>

## Existing inline comments (by other reviewers)

Total: <n> comments across <m> threads.

- OPEN: <n>     # still unresolved - do NOT duplicate these
- OUTDATED: <n>
- RESOLVED: <n>

### `<path>`

- **#<id>** @<login> <iso>  line=<L> status=**<OPEN/OUTDATED/RESOLVED>**  ...
  url: <html_url>
  > <body, truncated>

## My findings (your review)

### P0: <title>
<evidence: path:line on current head> - <why it's a blocker> - <concrete fix>

### P1 / P2 / P3: <title>
<...>

## Verdict
<APPROVE / COMMENT / REQUEST CHANGES, and the blockers if any>
````

## B. Prior open threads (generated context)

File: `curator_pr<N>_github_comment_queue_<YYYY_MM_DD>.md`. The still-unresolved
threads other reviewers already opened, so you can scan prior feedback before
adding your own. This is **context, not your output** - you are not replying to
these as the author; you read them to avoid duplicating and to check whether the
author addressed them.

````markdown
# Curator PR <N> Open Review Threads - <YYYY-MM-DD>

Current PR head: `<headRefOid>`

Threads other reviewers left that are still unresolved on the current head.

## Thread <i> - <path>:<line> by @<login>

Link: <html_url>

```text
<verbatim reviewer body>
```

## Stale (outdated/resolved) threads
- [<OUTDATED/RESOLVED>] `<path>:<line>` @<login> (<html_url>): <one-line body>
````

## C. Writing your review comments

Turn each finding into a review comment you post on the PR. For each:

- Anchor it to the exact `path:line-range` on the current head.
- Lead with severity (P0-P3) and a one-line statement of the problem.
- Explain *why* it matters (cite a `.cursor/rules/*.mdc` contract or
  `stages/audio/README.md` when the change deviates).
- Propose a concrete fix, ideally with a `suggestion` block.
- Before posting, check the "OPEN" list in file A - if another reviewer already
  raised it, don't repeat it (optionally +1 their thread instead).

Example review comment:

> **P0 (blocker).** `sharded_manifest_writer.py:101` calls `json.dumps(record)`
> while `record` can still hold the waveform `ndarray` when `keep_waveform=True`,
> which raises `TypeError: Object of type ndarray is not JSON serializable` and
> fails the whole shard. Drop the waveform key before serialising and guard
> `json.dumps` so a stray non-serialisable value names the offending key.
> Please also add a test under `tests/stages/audio/io/` covering
> `keep_waveform=True`.

When you have an exact fix, attach a GitHub `suggestion` block so the author can
apply it in one click (anchor the comment to the line(s) it replaces):

```suggestion
        record = {k: v for k, v in record.items() if k != "waveform"}
        line = json.dumps(record)
```

Then post the set as one PR review: inline comments for line-specific findings,
a top-level summary carrying the overall verdict (APPROVE / COMMENT / REQUEST
CHANGES).
