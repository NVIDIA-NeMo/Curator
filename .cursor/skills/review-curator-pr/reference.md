# Reference - NeMo Curator PR review

All paths are relative to the repository root. The skill is self-contained: it
relies only on this repository plus the GitHub CLI (`gh`).

## Canonical contracts (`.cursor/rules/`)

These always-on rules are the authoritative description of how stages and
pipelines must be written. Cite them when a PR deviates.

| Rule | Covers |
|------|--------|
| `processing-stage-patterns.mdc` | `ProcessingStage` subclassing, `inputs()`/`outputs()`, `process()`, lifecycle hooks |
| `composite-stage-patterns.mdc` | `CompositeStage` decomposition |
| `task-patterns.mdc` | `Task` / typed task subclasses, metadata flow |
| `executors.mdc` | Xenna / Ray Data backends and when to use each |
| `resources-configuration.mdc` | `Resources`, `batch_size`, GPU allocation |
| `pipeline-structure.mdc` | Pipeline assembly |
| `modality-structure.mdc` | Where modality code/tests/tutorials live |
| `coding-standards.mdc` | Ruff, copyright header, loguru, pytest, Python versions |

## Framework code (diff ground truth)

- `nemo_curator/stages/base.py` - `ProcessingStage` base class.
- `nemo_curator/backends/` - `base.py`, `xenna/`, `ray_data/`,
  `ray_actor_pool/`; adapters define how stages run on each backend.
- `nemo_curator/pipeline/pipeline.py` - pipeline construction and guards.
- `nemo_curator/tasks/` - task types.

## Modality guides

| Modality | Stage README | Tutorials |
|----------|--------------|-----------|
| audio | `nemo_curator/stages/audio/README.md` | `tutorials/audio/` |
| interleaved | `nemo_curator/stages/interleaved/README.md` | `tutorials/interleaved/` |
| text / image / video / math / synthetic | (stage code under `nemo_curator/stages/`) | `tutorials/{text,image,video,math,synthetic}/` |

## Performance / benchmarking

- `benchmarking/README.md` - how benchmarks are wired and run.
- `benchmarking/AUDIO_PROFILING.md`, `benchmarking/ALM_BENCHMARK.md` - modality
  perf expectations. Standalone benchmark scripts should be integrated here, not
  left freestanding.

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

GraphQL query used by the pull script:

```graphql
query($owner:String!,$repo:String!,$pr:Int!){
  repository(owner:$owner,name:$repo){
    pullRequest(number:$pr){
      reviewThreads(first:100){ nodes{
        id isResolved isOutdated isCollapsed line originalLine path
        comments(first:50){ nodes{ databaseId } }
      }}}}}
```

## Scripts

- `scripts/pr_review_pull.sh <N> [--outdir DIR] [--repo OWNER/REPO]` - fetch.
- `scripts/build_digest.py <N> [--outdir DIR] [--today YYYY-MM-DD] [--prev-head SHA] [--baseline-ts TS]` - render the digest + comment queue.

Default outdir: `.curator-pr-review/` (scratch; safe to delete or gitignore).
