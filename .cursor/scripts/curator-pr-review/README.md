# Shared Curator PR review scripts

Modality-agnostic helpers for reviewing NVIDIA-NeMo/Curator pull requests.
Modality skills call these through a modality-specific entry point that supplies
path filters, area rules, corpus labels, and defaults.

Requirements: Python 3.10+ and the GitHub CLI (`gh`) authenticated against
github.com.

## ensure_repo.sh

Reuse an existing Curator checkout or shallow-clone one. Prints
`CURATOR_REPO=<path>` on the last line.

## pr_review_pull.sh

```bash
.cursor/scripts/curator-pr-review/pr_review_pull.sh <PR_NUMBER> \
  [--outdir DIR] [--repo OWNER/REPO] [--path-regex REGEX] [--modality-label LABEL]
```

Pulls PR metadata, files, reviews, inline comments, issue comments, commits, and
GraphQL review threads into `pr<N>_*_latest.json`. REST endpoints, GraphQL review
threads, and comments within each thread are fully paginated.

When `--path-regex` is set, the command exits 3 if the PR touches no matching
path. `path_matches.py` is the shared implementation of that check.

## build_digest.py

```bash
.cursor/scripts/curator-pr-review/build_digest.py <PR_NUMBER> \
  [--outdir DIR] [--repo OWNER/REPO] [--today YYYY-MM-DD] \
  [--prev-head SHA] [--baseline-ts TS] \
  [--path-regex REGEX] [--modality-label LABEL] [--area-rules FILE]
```

Builds `curator_pr<N>_fresh_review_<date>.md` and
`curator_pr<N>_github_comment_queue_<date>.md`. Area rules are an ordered JSON
array of `{"pattern": REGEX, "label": FORMAT}` objects. Capturing groups are
available to labels as `{0}`, `{1}`, and so on. Without rules, paths are grouped
by their first component.

## build_corpus.py

```bash
.cursor/scripts/curator-pr-review/build_corpus.py \
  [--outdir DIR] [--numbers-file FILE] [--repo OWNER/REPO] \
  [--today YYYY-MM-DD] [--title TITLE] [--intro INTRO] \
  [--output-prefix PREFIX]
```

Consolidates per-PR JSON from a corpus pull into one markdown file with complete,
verbatim comment bodies grouped by pull request and file. Relative
`--numbers-file` values are resolved under `--outdir`; generic defaults are
`.curator-pr-review/corpus/` and `_pr_numbers.txt`.

## pull_review_threads.py

Internal helper used by `pr_review_pull.sh`. It follows GraphQL cursors for every
review-thread page and every comment page within each thread, and emits the
single JSON envelope consumed by `build_digest.py`.
