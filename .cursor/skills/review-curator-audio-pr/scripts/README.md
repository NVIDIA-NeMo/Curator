# Scripts - review-curator-audio-pr

Audio-specific wrappers and corpus tooling. Generic pull/digest/corpus logic
lives in `.cursor/skills/review-curator-pr/scripts/`; this folder forwards to
those helpers with the audio path filter from `../audio_paths.sh`.

## ensure_repo.sh / pr_review_pull.sh / build_digest.py

Thin wrappers that call the generic scripts with the audio path regex. **Audio-only:**
abort if the PR touches no audio path.

## pull_audio_pr_corpus.sh + build_corpus.py (required pre-review corpus)

Run before every audio PR review (SKILL.md step 3):

```bash
.cursor/skills/review-curator-audio-pr/scripts/pull_audio_pr_corpus.sh --since 1608
.cursor/skills/review-curator-audio-pr/scripts/build_corpus.py --today <YYYY-MM-DD>
```

`pull_audio_pr_corpus.sh` discovers audio PRs after #1608, pulls reviewer comments
(incremental by default; `--refresh` to re-pull), and writes into
`.curator-pr-review/audio-corpus/`. `build_corpus.py` renders
`audio_pr_corpus_<date>.md`.

The diff baseline for the PR under review is always `main`; the corpus is
separate read-only context from post-#1608 audio PR reviews. The cutoff excludes
feedback about the pre-`AudioTask` architecture that #1608 replaced. See
`../knowledge-sources.md` section 4.
