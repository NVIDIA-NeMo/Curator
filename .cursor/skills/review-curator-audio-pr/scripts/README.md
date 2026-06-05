# Scripts - review-curator-audio-pr

Helpers for a **reviewer** reviewing someone else's audio Curator PR. They
pull the PR and summarise existing review activity as context; you write the
actual review findings.

Both require the GitHub CLI (`gh`) authenticated against github.com. They write
to a scratch directory (default `.curator-pr-review/`) that is safe to delete;
do not commit its contents.

## pr_review_pull.sh

```bash
.cursor/skills/review-curator-audio-pr/scripts/pr_review_pull.sh <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
```

Pulls six REST endpoints (`pr view`, `reviews`, inline `comments`, issue
`comments`, `files`, `commits`) plus the GraphQL review threads into
`pr<N>_*_latest.json` (and timestamped snapshots for delta analysis).

## build_digest.py

```bash
.cursor/skills/review-curator-audio-pr/scripts/build_digest.py <PR_NUMBER> [--outdir DIR] [--today YYYY-MM-DD] [--prev-head SHA] [--baseline-ts TS]
```

Joins the `pr<N>_*_latest.json` files and writes two context files:
`curator_pr<N>_fresh_review_<date>.md` (working digest: PR state, diff, and
existing reviews/comments with OPEN/OUTDATED/RESOLVED status, plus a placeholder
for your findings) and `curator_pr<N>_github_comment_queue_<date>.md` (the
still-open threads other reviewers left, so you don't duplicate them).

- `--prev-head` sets the SHA in the "commits since prior reviewed head" header.
- `--baseline-ts <TS>` (a prior pull's timestamp suffix) marks
  comments/reviews/commits NEW vs already-seen.

The thread join uses comment `databaseId` when present, else a
(path, body-prefix) fallback, so both GraphQL thread-dump shapes classify
correctly. The scripts fetch any PR; the audio focus is applied in the review
lenses (see ../recurring-themes.md).
