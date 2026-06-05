# Output templates

`scripts/build_digest.py` emits both files automatically; these templates show
the exact structure so hand edits and the added findings/verdict narrative stay
consistent. Replace `<...>` placeholders. Keep the structure stable across
review passes so re-reviews diff cleanly.

## A. Fresh review digest

File: `curator_pr<N>_fresh_review_<YYYY_MM_DD>.md`

```markdown
# Curator PR <N> Fresh Review - <YYYY-MM-DD>

Review target: https://github.com/NVIDIA-NeMo/Curator/pull/<N>

Current PR head reviewed: `<headRefOid>`

Base recorded by GitHub metadata: `<baseRefOid>`

Previous reviewed head: `<prior_head_sha>`   # if a prior review exists

## PR state at review time

| Field | Value |
|---|---|
| state | <OPEN/CLOSED/MERGED> |
| isDraft | <bool> |
| reviewDecision | <APPROVED/CHANGES_REQUESTED/none> |
| mergeable / mergeStateStatus | <...> / <...> |
| changedFiles | <n> |
| additions / deletions | +<a> / -<d> |
| commits | <n> |
| inline review comments total | <n> |
| reviews submitted | <n> |
| top-level issue comments | <n> |
| updatedAt | <iso> |

## Commits since prior reviewed head `<prior8>` (<k> new)

```
<sha8>  <subject>  (<date>)
```

Files changed in current PR head (`git diff <base>..<head>`):

| File | +/- | status |
|---|---|---|
| `<path>` | +<a> -<d> | <added/modified> |

## Reviews

### #<id> by @<login>  state=<STATE>  commit=<sha8>  submitted=<iso>
<body or empty>

## Inline review comments

Total: <n> comments across <m> threads.

- OPEN: <n>
- OUTDATED: <n>
- RESOLVED: <n>
- ORPHAN: <n>

### `<path>`

- **#<id>** @<login> <iso>  line=<L> commit=<sha8> status=**<OPEN/OUTDATED/RESOLVED>**  thread_comments=<k> (<root/reply>) review_state=<STATE>
  url: <html_url>
  > <body, truncated, each line prefixed with "> ">

## Top-level issue comments

### #<id> by @<login>  <iso>
<body, truncated>

## Findings (reviewer analysis)

### P0: <title>
<evidence: path:line on current head> - <why blocker> - <recommended fix + test>

### P1 / P2 / P3: <title>
<...>

## Verdict
<merge-ready or not, and the blockers>
```

## B. GitHub comment queue

File: `curator_pr<N>_github_comment_queue_<YYYY_MM_DD>.md`. Only OPEN root
threads needing an author response or code change; stale threads at the bottom.

```markdown
# Curator PR <N> GitHub Comment Queue - <YYYY-MM-DD>

Review target: https://github.com/NVIDIA-NeMo/Curator/pull/<N>

Current PR head: `<headRefOid>`

## Comment <i> - <path>:<line> by @<login>

Thread: <html_url>

File: `<path>`

Line: `<line>`

Reviewer text:

```text
<verbatim reviewer body>
```

## Stale (outdated/resolved) comments

- [<OUTDATED/RESOLVED>] `<path>:<line>` @<login> (<html_url>): <one-line body>
```

## C. Reply drafting style

For each OPEN comment the user wants a paste-ready reply for:

- Acknowledge briefly.
- State status: already addressed (cite the commit SHA) or will-fix.
- Cite the exact `path:line-range` on the current head as evidence.
- Describe the concrete mechanism, not just "fixed".
- Keep each reply self-contained; the reviewer reads it in isolation.

Example:

> Thanks for flagging. Addressed on the current head (`<sha8>`): the writer now
> drops the waveform key before serialisation and guards `json.dumps`, so a
> non-serialisable value reports the offending key instead of failing the batch.
> Citation: `nemo_curator/stages/audio/io/sharded_manifest_writer.py:96-111`.
