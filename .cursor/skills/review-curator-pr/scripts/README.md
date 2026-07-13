# Scripts - review-curator-pr (generic)

Modality-agnostic helpers for reviewing NVIDIA-NeMo/Curator pull requests.
Modality skills (e.g. `review-curator-audio-pr`) wrap these with a path filter.

All scripts require the GitHub CLI (`gh`) authenticated against github.com.

## ensure_repo.sh

Reuse an existing Curator checkout or shallow-clone one. Prints `CURATOR_REPO=<path>`
on the last line.

## pr_review_pull.sh

```bash
.cursor/skills/review-curator-pr/scripts/pr_review_pull.sh <PR_NUMBER> \
  [--outdir DIR] [--repo OWNER/REPO] [--path-regex REGEX] [--modality-label LABEL]
```

Pulls PR metadata, files, reviews, inline comments, issue comments, commits, and
GraphQL review threads into `pr<N>_*_latest.json`. Paginated endpoints are merged
into valid JSON arrays (safe for PRs with >100 comments/files).

When `--path-regex` is set, aborts (exit 3) if the PR touches no matching path.

## build_digest.py

```bash
.cursor/skills/review-curator-pr/scripts/build_digest.py <PR_NUMBER> \
  [--outdir DIR] [--today YYYY-MM-DD] [--prev-head SHA] [--baseline-ts TS] \
  [--path-regex REGEX] [--modality-label LABEL]
```

Builds `curator_pr<N>_fresh_review_<date>.md` and
`curator_pr<N>_github_comment_queue_<date>.md`.

## build_corpus.py

```bash
.cursor/skills/review-curator-pr/scripts/build_corpus.py \
  [--outdir DIR] [--today YYYY-MM-DD] [--title TITLE] [--intro INTRO] \
  [--output-prefix PREFIX]
```

Consolidates per-PR JSON from a corpus pull into one markdown file with
recurring-theme tallies.
