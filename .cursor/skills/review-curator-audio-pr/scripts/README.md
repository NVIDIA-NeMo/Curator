# Scripts - review-curator-audio-pr

Helpers for a **reviewer** reviewing someone else's audio Curator PR. They pull
the PR (and prior audio review history) and summarise it as context; you write
the actual review findings. All require the GitHub CLI (`gh`) authenticated
against github.com, and write to a scratch directory (default
`.curator-pr-review/`) that is safe to delete; do not commit its contents.

## ensure_repo.sh

```bash
eval "$(.cursor/skills/review-curator-audio-pr/scripts/ensure_repo.sh | tail -1)"  # CURATOR_REPO=<path>
```

Avoids cloning the whole repo unnecessarily: reuses an existing checkout if you
are already inside one or there is one under the current directory tree
(bounded by `MAXDEPTH`, default 4); only then `git clone --depth 1` (no full
history) from <https://github.com/NVIDIA-NeMo/Curator>. Prints
`CURATOR_REPO=<absolute path>` on the last line.

## pr_review_pull.sh

```bash
.cursor/skills/review-curator-audio-pr/scripts/pr_review_pull.sh <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
```

Pulls six REST endpoints (`pr view`, `reviews`, inline `comments`, issue
`comments`, `files`, `commits`) plus the GraphQL review threads into
`pr<N>_*_latest.json` (and timestamped snapshots for delta analysis).
**Audio-only:** it pulls metadata + files first and aborts (exit 3) if the PR
touches no audio path, so the skill never reviews a non-audio PR.

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
correctly. It also re-checks the audio-only guard and aborts on a non-audio PR.

## pull_audio_pr_corpus.sh + build_corpus.py (required pre-review corpus)

Run these **before every audio PR review** (SKILL.md step 3). Do not skip even
when `.curator-pr-review/audio-corpus/` already has data - the pull is
incremental and `build_corpus.py` must render today's consolidated file.

```bash
.cursor/skills/review-curator-audio-pr/scripts/pull_audio_pr_corpus.sh --since 1608
.cursor/skills/review-curator-audio-pr/scripts/build_corpus.py --today <YYYY-MM-DD>
```

`pull_audio_pr_corpus.sh` lists every PR with number > `--since` (PR numbers are
monotonic in time, so this is "opened after that PR"), keeps the ones touching
audio paths, and pulls each one's reviews + inline + issue comments into
`.curator-pr-review/audio-corpus/`. It is **incremental**: PRs already pulled are
skipped (and non-audio candidates are remembered in `_non_audio_prs.txt`), so a
rerun only fetches PRs new since last time. Pass `--refresh` to re-pull
everything (e.g. to pick up new comments on still-open PRs).
`build_corpus.py` consolidates them into
`audio_pr_corpus_<date>.md`: one section per audio PR with every reviewer comment
verbatim (anchored to path:line) plus a recurring-themes tally. Read-only
context; it posts nothing. See `../knowledge-sources.md` section 4.

The single-PR scripts are **audio-only**: both `pr_review_pull.sh` and
`build_digest.py` abort on a PR that touches no audio path. The audio review
lenses live in `../knowledge-sources.md`.
