# Audio PR review knowledge sources

Read this before reviewing. It is the single index for an audio Curator review
and has five parts:

0. **Canonical docs** - the authoritative references to ground every finding.
1. **Audio code map** - where each audio stage lives.
2. **Review lenses** - what to check, each linked to the audio code, README
   section, and `.cursor/rules` contract it governs.
3. **Severity mapping** - P0-P3.
4. **Pre-review corpus (required)** - pull every audio PR after #1608 (open +
   closed) with reviewer comments and read the consolidated file before you
   write findings.
5. **GitHub data + scripts reference** - the `gh` endpoints and helper scripts.

All paths are repo-relative. The skill is self-contained: this repository plus
the GitHub CLI (`gh`). Scope: the **audio** modality.

---

## 0. Canonical docs (start here)

| Source | What it gives you |
|--------|-------------------|
| `nemo_curator/stages/audio/README.md` | The audio stage **developer guide** (1000+ lines): CPU vs GPU stages, `process` / `process_batch`, `batch_size`, call chains, per-stage memory characteristics, end-to-end FLEURS & ALM traces, new-stage checklist. Read first. |
| Audio curation developer guide (slides): <https://docs.google.com/presentation/d/15lJHyAoRTbNFDWq8UJ6czMaUFck3WYPCRaMYPfO_qew/edit?slide=id.g3be823d0e3d_0_0#slide=id.g3be823d0e3d_0_0> | Design intent and pipeline overviews behind the audio modality (the deck the README operationalizes). |
| `nemo_curator/tasks/audio_task.py` | `AudioTask(Task[dict])` + `_AttrDict` task model (one manifest entry per task; dict keys exposed as attributes for `validate_input`). |
| `nemo_curator/stages/base.py` | `ProcessingStage` contract: `inputs()`/`outputs()`, `process`/`process_batch`, `setup_on_node`/`setup`/`teardown`, `validate_input`. |
| `CONTRIBUTING.md` | DCO sign-off, PR process, required tests, coverage threshold, copyright header. |
| `.cursor/rules/*.mdc` | Always-on engineering contracts (table below). |
| `tutorials/audio/README.md` + per-pipeline READMEs | Runnable reference pipelines (alm, audio_pretrain, callhome_diar, fleurs, readspeech, single_speaker_filter, tagging). |
| `benchmarking/README.md`, `benchmarking/AUDIO_PROFILING.md`, `benchmarking/ALM_BENCHMARK.md` | How audio/ALM benchmarks are wired and the perf expectations to hold a PR to. |
| `tests/stages/audio/` | Test layout that mirrors `nemo_curator/stages/audio/`; the coverage bar. |

### `.cursor/rules/` contracts

| Rule | Covers |
|------|--------|
| `.cursor/rules/processing-stage-patterns.mdc` | `ProcessingStage` subclassing, `inputs()`/`outputs()`, `process()`, lifecycle hooks |
| `.cursor/rules/composite-stage-patterns.mdc` | `CompositeStage` decomposition (e.g. a reader = discovery + shard fan-out) |
| `.cursor/rules/task-patterns.mdc` | `Task` / typed task subclasses, metadata flow |
| `.cursor/rules/executors.mdc` | Xenna / Ray Data backends and when to use each |
| `.cursor/rules/resources-configuration.mdc` | `Resources`, `batch_size`, GPU allocation |
| `.cursor/rules/pipeline-structure.mdc` | Pipeline assembly |
| `.cursor/rules/modality-structure.mdc` | Where modality code/tests/tutorials live |
| `.cursor/rules/coding-standards.mdc` | Ruff, copyright header, loguru, pytest, Python versions |

### Published audio documentation (fern docs site)

The user-facing audio docs live under `fern/versions/<ver>/pages/` (latest
versioned snapshot: **v26.04**; `fern/versions/main/` holds in-progress docs).
These are the canonical "what the docs promise" reference - cite them when a PR's
behavior, config, or API drifts from documented behavior. Paths below are
relative to `fern/versions/v26.04/pages/`.

| Audio concept / stage | Fern doc(s) | Code it documents |
|------------------------|-------------|-------------------|
| Overview / get started | `get-started/audio.mdx`, `curate-audio/index.mdx` | - |
| Concepts (task model, pipelines, metrics) | `about/concepts/audio/{index,audio-task,curation-pipeline,asr-pipeline,alm-pipeline,manifests-ingest,quality-metrics,text-integration}.mdx` | `nemo_curator/tasks/audio_task.py` |
| Load data / manifests | `curate-audio/load-data/{index,fleurs-dataset,custom-manifests,local-files}.mdx` | `nemo_curator/stages/audio/datasets/`, `.../io/` |
| ASR inference | `curate-audio/process-data/asr-inference/{index,nemo-models}.mdx` | `nemo_curator/stages/audio/inference/asr/asr_nemo.py` |
| Audio analysis (duration, format) | `curate-audio/process-data/audio-analysis/{index,duration-calculation,format-validation}.mdx` | `nemo_curator/stages/audio/common.py` |
| Quality filtering (band, VAD, SigMOS, UTMOS, speaker separation, preprocessing, data-filter) | `curate-audio/process-data/quality-filtering/{index,band-filter,vad,sigmos,utmos,speaker-separation,preprocessing,audio-data-filter-stage}.mdx` | `.../filtering/`, `.../segmentation/`, `.../preprocessing/`, `.../advanced_pipelines/` |
| Quality assessment (WER, duration) | `curate-audio/process-data/quality-assessment/{index,wer-filtering,duration-filtering}.mdx` | `nemo_curator/stages/audio/metrics/wer.py`, `.../common.py` |
| ALM (data builder, overlap filtering) | `curate-audio/process-data/alm/{index,data-builder,overlap-filtering}.mdx` | `nemo_curator/stages/audio/alm/` |
| Text integration / tagging | `curate-audio/process-data/text-integration/index.mdx` | `nemo_curator/stages/audio/tagging/` |
| Save / export | `curate-audio/save-export.mdx` | `nemo_curator/stages/audio/io/convert.py` |
| Tutorials | `curate-audio/tutorials/{index,beginner,readspeech,alm}.mdx` | `tutorials/audio/` |
| API reference | `api-reference/tasks/audio-task.mdx`, `api-reference/processing-stage.mdx` | `nemo_curator/tasks/audio_task.py`, `nemo_curator/stages/base.py` |
| Infrastructure (memory, per-stage runtime, backends, GPU, resumable, monitoring) | `reference/infrastructure/{memory-management,per-stage-runtime,execution-backends,gpu-processing,resumable-processing,monitoring}.mdx` | `nemo_curator/backends/` |
| Release notes / migration (`AudioBatch -> AudioTask`, #1608) | `about/release-notes/{index,migration-guide,migration-faq}.mdx` | see PR #1608 |

When a PR changes audio behavior, check whether the matching fern page above
needs updating in the same PR - docs drift is a common review finding.

---

## 1. Audio code map (diff ground truth)

`nemo_curator/stages/audio/` is organized as:

| Area | Path(s) |
|------|---------|
| Task model | `nemo_curator/tasks/audio_task.py` |
| Base contract | `nemo_curator/stages/base.py` |
| Common CPU stages (duration, value filter) | `nemo_curator/stages/audio/common.py` |
| I/O (convert to DocumentBatch, segment extraction) | `nemo_curator/stages/audio/io/convert.py`, `nemo_curator/stages/audio/io/extract_segments.py` |
| Preprocessing | `nemo_curator/stages/audio/preprocessing/mono_conversion.py`, `nemo_curator/stages/audio/preprocessing/concatenation.py` |
| Inference - ASR | `nemo_curator/stages/audio/inference/asr/asr_nemo.py` |
| Inference - VAD | `nemo_curator/stages/audio/inference/vad/whisperx_vad.py` |
| Inference - speaker diarization | `nemo_curator/stages/audio/inference/speaker_diarization/sortformer.py`, `nemo_curator/stages/audio/inference/speaker_diarization/pyannote.py` |
| Filtering (band, SigMOS, UTMOS) | `nemo_curator/stages/audio/filtering/band.py`, `nemo_curator/stages/audio/filtering/sigmos.py`, `nemo_curator/stages/audio/filtering/utmos.py` |
| Segmentation | `nemo_curator/stages/audio/segmentation/vad_segmentation.py`, `nemo_curator/stages/audio/segmentation/speaker_separation.py` |
| Tagging (inference + text) | `nemo_curator/stages/audio/tagging/inference/nemo_asr_align.py`, `nemo_curator/stages/audio/tagging/merge_alignment_diarization.py`, `nemo_curator/stages/audio/tagging/resample_audio.py`, `nemo_curator/stages/audio/tagging/split.py`, `nemo_curator/stages/audio/tagging/text/itn.py` |
| Postprocessing | `nemo_curator/stages/audio/postprocessing/timestamp_mapper.py` |
| Metrics / throughput | `nemo_curator/stages/audio/metrics/wer.py`, `nemo_curator/stages/audio/metrics/bandwidth.py`, `nemo_curator/stages/audio/metrics/squim.py` |
| Datasets (FLEURS, ReadSpeech) | `nemo_curator/stages/audio/datasets/fleurs/create_initial_manifest.py`, `nemo_curator/stages/audio/datasets/readspeech/create_initial_manifest.py`, `nemo_curator/stages/audio/datasets/file_utils.py` |
| ALM (audio-language model data) | `nemo_curator/stages/audio/alm/alm_data_builder.py`, `nemo_curator/stages/audio/alm/alm_data_overlap.py`, `nemo_curator/stages/audio/alm/pretrain/pipeline.py`, `nemo_curator/stages/audio/alm/pretrain/io.py` |
| Composite pipelines | `nemo_curator/stages/audio/advanced_pipelines/audio_data_filter/audio_data_filter.py` |

Backends (how stages run): `nemo_curator/backends/` (`base.py`, `xenna/`,
`ray_data/`, `ray_actor_pool/`); pipeline assembly:
`nemo_curator/pipeline/pipeline.py`; task types: `nemo_curator/tasks/`.

---

## 2. Review lenses (each with code + doc references)

Apply each lens to the diff and cite exact `path:line` evidence. When a change
violates a contract, link it by name. The **References** under each lens are the
canonical sources to cite.

### 2.1 Stage contracts and reuse

- An audio stage subclasses `ProcessingStage[AudioTask, AudioTask]` (or the
  appropriate in/out task types) with explicit `inputs()` / `outputs()`, a
  meaningful `name`, and appropriate `resources` / `batch_size` (set as class
  attributes / dataclass fields, never the read-only `_name` / `_resources` /
  `_batch_size` properties). Field order follows the base convention: `name`
  first, then params, then `resources`, then `batch_size` last.
- GPU/model work (ASR, VAD, diarization, tagging inference) belongs in
  `process_batch()` with a real `batch_size`; `__init__` stays lightweight (it
  runs on the driver). Batch-only stages raise `NotImplementedError` from
  `process()` (the convention used by dedup stages).
- Reuse existing audio readers, manifest writers, and shared model abstractions
  instead of bespoke orchestration. Model a tarred/sharded reader as a
  `CompositeStage` = a discovery stage + a shard-reader fan-out.
- If `inputs()` does not declare a required key (e.g. `audio_filepath_key`,
  `duration_key`), runtime validation won't check it, and a direct
  `task.data[key]` access then raises an opaque `KeyError` instead of a clear
  validation error. Declare required keys.
- Fan-out shard readers need deterministic child IDs and must propagate
  `_metadata` / `_stage_perf`; declare `ray_stage_spec()` with
  `IS_FANOUT_STAGE: True`. First-stage shard discovery from an empty input
  should constrain placement (e.g. one worker per node) so discovery isn't
  duplicated across workers.
- **References:** `nemo_curator/stages/base.py`; `nemo_curator/tasks/audio_task.py`; `nemo_curator/stages/audio/common.py`; `nemo_curator/stages/audio/io/convert.py`; `nemo_curator/stages/audio/datasets/fleurs/create_initial_manifest.py`; rules `processing-stage-patterns.mdc`, `composite-stage-patterns.mdc`, `task-patterns.mdc`; README sections "Writing a CPU stage" / "Writing a GPU stage".

### 2.2 Setup / teardown lifecycle and audio dependencies

- `setup_on_node()` = once-per-node prep (model download / shared cache / output
  truncation). `setup()` = per-worker init (model load, GPU allocation).
  `teardown()` frees GPU memory. Don't conflate them; prefer `setup_on_node` for
  anything that must run exactly once (avoids `_setup_done`-style guards).
- Import heavy/optional audio deps (NeMo, vLLM, whisperx, pyannote, soundfile,
  model-specific utils) lazily inside `setup()` or behind a guarded import,
  never unconditionally at module top level. A top-level `import <optional_pkg>`
  breaks `import nemo_curator.stages.audio...` on any install without that extra
  (including CPU-only / Mac/ARM).
- Declare the full audio runtime dependency set in the appropriate
  `pyproject.toml` optional extra (e.g. a CUDA extra); add an import smoke test.
  Don't rely on an external Docker image to supply deps a reusable stage needs.
- Avoid hard version pins (`pkg==x.y.z`) and global overrides for libraries
  shared across modalities (`transformers`, `huggingface-hub`, `accelerate`);
  prefer compatible ranges.
- Expose model knobs like `cache_dir` and `trust_remote_code` as parameters
  rather than hardcoding them.
- **References:** `nemo_curator/stages/audio/inference/asr/asr_nemo.py` (download-on-node / load-on-worker split, `cache_dir`); `nemo_curator/stages/audio/inference/vad/whisperx_vad.py`; `nemo_curator/stages/audio/inference/speaker_diarization/sortformer.py`, `pyannote.py`; `nemo_curator/stages/audio/filtering/{utmos,sigmos,band}.py`; `pyproject.toml`; rules `resources-configuration.mdc`; README "Stage lifecycle".

### 2.3 Secret-safe logging

- Never log a full resolved config; audio tutorials commonly accept an HF token
  / cloud credentials. Redact secret-like keys before logging (token, password,
  secret, credentials, `*_token`/`*_secret`), or log only a non-secret subset,
  and add a test asserting the secret value never appears in logged output.
- **References:** `tutorials/audio/*/` (token/credential inputs); any stage that
  logs configuration; rules `coding-standards.mdc` (loguru).

### 2.4 Waveform/tensor memory and manifest serialization

- Drop waveform arrays / audio tensors from `task.data` once a stage no longer
  needs them. A leaked numpy `ndarray` / torch tensor that reaches a JSON
  manifest writer raises `TypeError: Object of type ndarray is not JSON
  serializable` and can fail an entire shard/batch, or cause OOM at scale. Strip
  by waveform key plus a duck-typed guard (`.shape`/`.dtype`), and wrap
  `json.dumps` so a stray non-serializable value reports the offending key.
- ALM stages that replace the whole dict must `task.data.clear()` +
  `task.data.update(result)` to preserve the `_AttrDict` type. Remember
  `_AttrDict.__setattr__` routes writes into the dict, so never stamp private
  fields onto a task's data dict.
- Guard audio I/O edge cases: `tarfile.extractfile()` returns `None` for
  non-regular members even after `isfile()` - check before `.read()`. Handle
  resampling / channel-count / dtype mismatches explicitly.
- **References:** `nemo_curator/stages/audio/io/convert.py`; `nemo_curator/stages/audio/io/extract_segments.py`; `nemo_curator/stages/audio/alm/alm_data_builder.py`; `nemo_curator/stages/audio/preprocessing/{mono_conversion,concatenation}.py`; `nemo_curator/tasks/audio_task.py`; README "Memory characteristics"; fern `reference/infrastructure/{memory-management,per-stage-runtime}.mdx`.

### 2.5 Tarred / sharded audio I/O

- Keep infrastructure assumptions (specific object-store URLs, auth env vars,
  cluster-local paths) out of the generic reader/writer. Make remote access
  pluggable: accept local paths, ordinary cloud URIs, or a user-provided
  opener/resolver, and isolate environment-specific support behind a small,
  documented, optional component. For piped/streamed downloads use failure-aware
  flags + retries so truncated streams fail loudly.
- Cache the fsspec filesystem object in `setup_on_node` rather than re-creating
  it per `process()` call (a new HTTP connection per manifest entry is costly on
  S3/GCS at scale).
- Manifest/shard writers that loop per task inside `process_batch()` reopen the
  output repeatedly; group by shard and open each output once per batch. Preserve
  `.done`/checkpoint markers so runs are resumable.
- **References:** `nemo_curator/stages/audio/io/extract_segments.py`; `nemo_curator/stages/audio/datasets/{fleurs,readspeech}/create_initial_manifest.py`; `nemo_curator/stages/audio/datasets/file_utils.py`; `nemo_curator/stages/audio/alm/pretrain/io.py`; rules `composite-stage-patterns.mdc`.

### 2.6 Sample-rate and metadata propagation

- Resampling/mono-conversion/segmentation stages must update the sample-rate and
  duration keys they change and propagate `_metadata` / `_stage_perf`; downstream
  stages that assume a fixed sample rate must declare it in `inputs()`.
- **References:** `nemo_curator/stages/audio/preprocessing/mono_conversion.py`; `nemo_curator/stages/audio/tagging/resample_audio.py`; `nemo_curator/stages/audio/segmentation/vad_segmentation.py`; `nemo_curator/stages/audio/postprocessing/timestamp_mapper.py`.

### 2.7 Streaming, throughput, benchmarking

- Streaming execution with per-stage resource configuration is the default model
  for scalable audio pipelines.
- Emit stage metrics through the supported metrics path, JSON-shaped. For
  multi-node audio runs include worker/node/GPU identity and latency
  distributions (p50/p95/p99) plus audio-seconds-per-wall-second, not just
  totals/averages.
- Standalone audio benchmark scripts belong in the `benchmarking/` flow with
  config entries and comparable parameters - not freestanding.
- **References:** `nemo_curator/stages/audio/metrics/{wer,bandwidth,squim}.py`; `benchmarking/AUDIO_PROFILING.md`, `benchmarking/ALM_BENCHMARK.md`, `benchmarking/README.md`; rules `executors.mdc`; fern `reference/infrastructure/{execution-backends,per-stage-runtime,monitoring,gpu-processing}.mdx`.

### 2.8 Tutorials and docs

- Audio tutorials (`tutorials/audio/`) must be runnable, minimal, and use public
  APIs only - no private APIs, no machine-local absolute paths, no committed
  noisy notebook output, no executor mismatch between a notebook and its CLI.
- Don't duplicate pipeline-construction logic between a tutorial entrypoint and
  an `examples/` script; extract one shared builder both call.
- Every config key documented for users must actually reach a stage; validate
  config-to-stage mapping and fail (or warn) on unused keys.
- Don't commit generated analysis/scratch docs into the source tree.
- **References:** `tutorials/audio/README.md`; `tutorials/audio/{alm,audio_pretrain,callhome_diar,fleurs,readspeech,single_speaker_filter,tagging}/README.md`; `tutorials/audio/readspeech/readspeech_tutorial.ipynb`; `nemo_curator/stages/audio/README.md`; fern `curate-audio/tutorials/{index,beginner,readspeech,alm}.mdx`.

### 2.9 Tests and standards (CONTRIBUTING + coding-standards)

- Source changes need accompanying tests; `tests/stages/audio/` mirrors
  `nemo_curator/stages/audio/`. The project enforces 80% coverage under
  `nemo_curator/`. GPU-only audio tests use `@pytest.mark.gpu`; reuse fixtures
  under `tests/fixtures/audio/` and sample data under `tests/data/audio/`.
- Every non-empty Python file carries the NVIDIA Apache-2.0 copyright header.
- Ruff clean at 119-char lines; type annotations on functions; loguru for logs.
- Every commit is DCO signed-off.
- **References:** `CONTRIBUTING.md`; rules `coding-standards.mdc`, `modality-structure.mdc`; `tests/stages/audio/`.

### 2.10 PR size and reviewability

- Large multi-stage audio PRs are hard to review and defend. Prefer splitting by
  stage / module / tutorial / benchmark. If a single PR must remain, organize it
  as if sliced: no duplicated builder logic, clear module ownership, a reviewable
  commit sequence.

### Smaller recurring nits

- Missing copyright headers on new files.
- Top-level imports that should be lazy/guarded (very common on audio model modules).
- Long inline prompt/config strings that should be module constants.
- Dataclass fields validated in `__post_init__` that could just be typed.
- Dead conditionals / redundant truthiness checks.
- Pytests added for tutorials that don't warrant them.
- Debug-only knobs (e.g. `max_utterances_per_shard`) presented as features - document them as debug throttles.
- `trust_remote_code=True` hardcoded on a model load - expose as a parameter.

---

## 3. Severity mapping

| Severity | Meaning |
|----------|---------|
| P0 | Merge blocker: crash/data-loss on a valid audio config, secret leaked to logs, import breaks a supported install |
| P1 | Fix before merge: undeclared audio deps, missing required tests/coverage, env-specific I/O in reusable reader, PR too large |
| P2 | Should fix: duplicated pipeline construction, inefficient writer batching, incomplete throughput metrics, documented-but-unwired config |
| P3 | Nice to have: temporary model/compat workaround needing test+comment, debug knob docs |

---

## 4. Pre-review corpus (required): learn from post-#1608 audio PRs

PR [#1608](https://github.com/NVIDIA-NeMo/Curator/pull/1608) (`AudioBatch ->
AudioTask` redesign) reset the audio stage contracts. Reviewer feedback on audio
PRs **after** #1608 is the best predictor of what to flag next. **Before every
audio PR review**, build (or refresh) the consolidated corpus of that feedback.
Do not skip this step, even when a prior `audio_pr_corpus_*.md` already exists
on disk - rerun the pull (incremental by default) and render today's file:

```bash
# 1) discover audio PRs after #1608 (open + closed/merged) and pull their
#    reviews + inline comments into .curator-pr-review/audio-corpus/
.cursor/skills/review-curator-audio-pr/scripts/pull_audio_pr_corpus.sh --since 1608

# 2) render one consolidated, reviewer-comment corpus
.cursor/skills/review-curator-audio-pr/scripts/build_corpus.py
```

`pull_audio_pr_corpus.sh` lists every PR with number > `--since` (PR numbers are
monotonic in time, so number > 1608 == opened after #1608), keeps the ones whose
changed files touch audio paths, and pulls each one's reviews, inline comments,
and issue comments. It is incremental - reruns skip PRs already on disk and only
fetch new ones (use `--refresh` to re-pull, e.g. to refresh open PRs).
`build_corpus.py` writes
`.curator-pr-review/audio-corpus/audio_pr_corpus_<date>.md`: one section per
audio PR (number, title, state, author, link) with every reviewer comment
verbatim, anchored to `path:line`, plus a recurring-themes tally.

Use the corpus to (a) recognize patterns reviewers repeatedly raise (the lenses
in section 2 came from exactly this), and (b) check whether the PR in front of
you repeats a mistake already called out elsewhere. It is read-only context; it
never auto-posts anything. If the corpus scripts fail or the consolidated file
is missing, stop the review and fix the failure before writing findings.

---

## 5. GitHub data + scripts reference

`scripts/ensure_repo.sh [CLONE_DIR]` - reuse an existing Curator checkout or
shallow-clone one; prints `CURATOR_REPO=<path>` on its last line.

`scripts/pr_review_pull.sh <N> [--outdir DIR] [--repo OWNER/REPO]` pulls into the
scratch outdir:

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

`scripts/build_digest.py <N> [--outdir DIR] [--today YYYY-MM-DD] [--prev-head SHA] [--baseline-ts TS]`
renders the working digest + the prior-open-threads context file (you add your
own findings).

`scripts/pull_audio_pr_corpus.sh [--since N] [--outdir DIR] [--repo OWNER/REPO] [--limit N] [--refresh]`
(incremental - skips PRs already on disk; `--refresh` re-pulls) and
`scripts/build_corpus.py [--outdir DIR] [--today YYYY-MM-DD]` **must** run before
every review to build the post-#1608 corpus (section 4; SKILL.md step 3).

Default outdir: `.curator-pr-review/` (scratch; gitignored, safe to delete).
