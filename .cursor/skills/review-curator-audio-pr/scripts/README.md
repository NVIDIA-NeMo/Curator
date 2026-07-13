# Scripts - review-curator-audio-pr

The audio skill has one entry point for shared PR-review operations and one
audio-specific corpus discovery script. Modality-agnostic implementations live
once under `.cursor/scripts/curator-pr-review/`.

## review_audio_pr.sh

```bash
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh ensure-repo [CLONE_DIR]
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh pull <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh digest <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO] [--today YYYY-MM-DD]
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh build-corpus [--outdir DIR] [--repo OWNER/REPO] [--today YYYY-MM-DD]
```

The entry point supplies the audio path filter from `../audio_paths.sh` and
rejects non-audio PRs. `pull` gathers PR metadata and review activity; `digest`
renders the working digest and open-thread queue; `build-corpus` renders the
audio corpus. The full digest and corpus preserve complete comment bodies; the
raw JSON remains the archival source.

## pull_audio_pr_corpus.sh

Run the corpus pull and build before every audio PR review:

```bash
.cursor/skills/review-curator-audio-pr/scripts/pull_audio_pr_corpus.sh --since 1608
.cursor/skills/review-curator-audio-pr/scripts/review_audio_pr.sh build-corpus --today <YYYY-MM-DD>
```

The puller discovers audio PRs after #1608 and incrementally stores reviewer
comments in `.curator-pr-review/audio-corpus/`. Cached PRs are refreshed when
`updatedAt` changes; `--refresh` forces a complete re-pull. Fern-only PRs are out
of scope. The
diff baseline for the target PR remains `main`; the corpus is separate context
from reviews of the post-#1608 architecture.
