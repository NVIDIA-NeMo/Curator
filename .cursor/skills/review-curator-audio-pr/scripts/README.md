# Scripts - review-curator-audio-pr

The audio skill has one entry point for shared PR-review operations and one
audio-specific corpus discovery script. Modality-agnostic implementations live
once under `.cursor/scripts/curator-pr-review/`.

## review_audio_pr.sh

```bash
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh ensure-repo [CLONE_DIR]
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh pull <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh digest <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO] [--today YYYY-MM-DD]
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh build-corpus [--cache-dir DIR] [--outdir DIR] [--repo OWNER/REPO] [--since N] [--today YYYY-MM-DD]
```

The entry point supplies the audio path filter from `../audio_paths.sh` and
rejects non-audio PRs. The filter includes explicitly audio-scoped Fern pages;
generic Fern navigation files qualify only when the same PR changes an
audio-scoped page. `pull` gathers PR metadata and review activity; `digest`
renders the working digest and open-thread queue; `build-corpus` renders the
audio corpus. The full digest and corpus preserve complete comment bodies; the
raw JSON remains the archival source.

## pull_audio_pr_corpus.sh

Run the corpus pull and build before every audio PR review:

```bash
export CURATOR_PR_REVIEW_CACHE_ROOT=/absolute/workspace/reviews/.cache/curator-pr-review
.cursor/skills/review-curator-audio-pr/scripts/pull_audio_pr_corpus.sh --since 1608
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh build-corpus \
  --since 1608 --outdir <CURRENT_REVIEW_DIR>/audio-corpus --today <YYYY-MM-DD>
```

The puller discovers audio PRs after #1608 and incrementally stores reviewer
comments once in a shared repository-keyed cache. Cached PRs are refreshed when
`updatedAt` changes; `--refresh` forces a complete re-pull. The `--outdir`
option on `build-corpus` writes only one rendered Markdown file into the current
review directory. The default cache uses XDG/`~/.cache`; set
`CURATOR_PR_REVIEW_CACHE_ROOT` for a stable workspace cache or pass the same
`--cache-dir` to both commands. Never copy or seed raw cache files into per-PR
review directories. Cache and output paths cannot overlap, and refreshes of one
cache are serialized. Cache payloads are atomically published and the selection
manifest is committed last, so a failed refresh retains the prior complete
corpus. Manifests are keyed by `--since` and the audio path filter; pass the same
`--since` to pull and build. The corpus puller intentionally rejects its old
ambiguous `--outdir` option; use `--cache-dir`. Fern-only PRs are in scope when
they change explicitly audio-scoped pages. The diff baseline for the target PR
remains `main`; the corpus is separate context from reviews of the post-#1608
architecture.
