# Reference - NeMo Curator audio PR review

All paths are relative to the repository root. The skill is self-contained: it
relies only on this repository plus the GitHub CLI (`gh`). Scope: the **audio**
modality.

## Primary audio guide

- `nemo_curator/stages/audio/README.md` - audio stage contracts, `process` vs
  `process_batch`, lazy/optional imports, metrics. Read this first for any
  audio review.

## Canonical contracts (`.cursor/rules/`)

Always-on rules describing how stages and pipelines must be written. Cite them
when a PR deviates.

| Rule | Covers |
|------|--------|
| `processing-stage-patterns.mdc` | `ProcessingStage` subclassing, `inputs()`/`outputs()`, `process()`, lifecycle hooks |
| `composite-stage-patterns.mdc` | `CompositeStage` decomposition (e.g. a reader = discovery + shard fan-out) |
| `task-patterns.mdc` | `Task` / typed task subclasses, metadata flow |
| `executors.mdc` | Xenna / Ray Data backends and when to use each |
| `resources-configuration.mdc` | `Resources`, `batch_size`, GPU allocation |
| `pipeline-structure.mdc` | Pipeline assembly |
| `modality-structure.mdc` | Where modality code/tests/tutorials live |
| `coding-standards.mdc` | Ruff, copyright header, loguru, pytest, Python versions |

## Audio stage map (diff ground truth)

`nemo_curator/stages/audio/` is organized as:

| Area | Path |
|------|------|
| Base contract | `nemo_curator/stages/base.py` |
| I/O (tarred/sharded readers, manifest writers) | `nemo_curator/stages/audio/io/` |
| Preprocessing | `nemo_curator/stages/audio/preprocessing/` |
| Inference: ASR / VAD / speaker diarization | `nemo_curator/stages/audio/inference/{asr,vad,speaker_diarization}/` |
| Filtering (band, SigMOS, etc.) | `nemo_curator/stages/audio/filtering/` |
| Segmentation (speaker separation) | `nemo_curator/stages/audio/segmentation/` |
| Tagging (inference + text) | `nemo_curator/stages/audio/tagging/` |
| Postprocessing | `nemo_curator/stages/audio/postprocessing/` |
| Metrics / throughput | `nemo_curator/stages/audio/metrics/` |
| Datasets (fleurs, readspeech) | `nemo_curator/stages/audio/datasets/` |
| ALM (audio-language pretrain) | `nemo_curator/stages/audio/alm/` |
| Composite pipelines | `nemo_curator/stages/audio/advanced_pipelines/` |

Backends (how stages run): `nemo_curator/backends/` (`base.py`, `xenna/`,
`ray_data/`, `ray_actor_pool/`); pipeline assembly:
`nemo_curator/pipeline/pipeline.py`; task types: `nemo_curator/tasks/`.

## Audio tutorials

`tutorials/audio/` - `alm/`, `audio_pretrain/`, `callhome_diar/`, `fleurs/`,
`readspeech/`, `single_speaker_filter/`, `tagging/`. Tutorials must be runnable,
use public APIs only, and not duplicate pipeline-building logic that belongs in
a shared builder.

## Audio performance / benchmarking

- `benchmarking/README.md` - how benchmarks are wired and run.
- `benchmarking/AUDIO_PROFILING.md` - audio benchmark / perf expectations.
- `benchmarking/ALM_BENCHMARK.md` - ALM pipeline benchmark.

Standalone audio benchmark scripts should be integrated into this flow, not left
freestanding.

## Audio tests

`tests/stages/audio/` mirrors `nemo_curator/stages/audio/` (io, inference,
filtering, segmentation, tagging, postprocessing, preprocessing, metrics,
datasets, alm, advanced_pipelines). Audio fixtures live under
`tests/fixtures/audio/` and `tests/data/audio/`. GPU-only tests are marked
`@pytest.mark.gpu`.

## Contribution rules

- `CONTRIBUTING.md` - DCO sign-off (`git commit -s`), PR process, test
  requirements, coverage threshold, copyright header.

## GitHub data via `gh`

`scripts/pr_review_pull.sh <N>` pulls these into the scratch outdir:

| File | Endpoint |
|------|----------|
| `pr<N>_gh_latest.json` | `gh pr view <N> --json ...` |
| `pr<N>_reviews_latest.json` | `repos/:owner/:repo/pulls/<N>/reviews` |
| `pr<N>_review_comments_latest.json` | `repos/:owner/:repo/pulls/<N>/comments` (inline) |
| `pr<N>_issue_comments_latest.json` | `repos/:owner/:repo/issues/<N>/comments` |
| `pr<N>_files_latest.json` | `repos/:owner/:repo/pulls/<N>/files` |
| `pr<N>_commits_latest.json` | `repos/:owner/:repo/pulls/<N>/commits` |
| `pr<N>_review_threads_latest.json` | GraphQL `pullRequest.reviewThreads` (isResolved/isOutdated) |

The REST inline-comments endpoint does not expose resolve/outdate state; the
GraphQL thread payload does. The builder joins them by comment `databaseId`
(== REST `id`), falling back to a `(path, body-prefix)` match when an older
thread dump lacks `databaseId`.

## Scripts

- `scripts/pr_review_pull.sh <N> [--outdir DIR] [--repo OWNER/REPO]` - fetch.
- `scripts/build_digest.py <N> [--outdir DIR] [--today YYYY-MM-DD] [--prev-head SHA] [--baseline-ts TS]` - render the digest + comment queue.

Default outdir: `.curator-pr-review/` (scratch; safe to delete or gitignore).
