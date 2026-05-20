# Steward: Tutorials & Examples

This domain exists because tutorials are how users first decide
whether NeMo Curator works for them. A tutorial that runs cleanly on
first try recruits the user; one that fails on imports or wrong
extras loses them. Tutorials are also the single biggest doc-rot
surface — they reach into the public API and break whenever that API
changes without a corresponding tutorial update.

Related: root [AGENTS.md](../AGENTS.md),
[tutorials/README.md](README.md), `tutorials/quickstart.py`.

## Point Of View

The user's "does this actually work?" test. The user this domain
serves runs on cluster-scheduled GPU infrastructure (Slurm,
Kubernetes, multi-node Ray) as often as on a single machine — but
the first impression usually happens on a laptop or single GPU, so
both have to work. Every tutorial is a contract that imports,
configs, CLI invocations, and pipeline composition shown here
resolve against the current `nemo_curator/` public API.

## Protect

- **Imports resolve.** Every `from nemo_curator…` line in a tutorial
  resolves against the current installed package.
- **Pipeline composition type-checks.** `pipeline.add_stage(...)`
  examples produce a runnable pipeline against current
  `ProcessingStage` and `Task` types.
- **CLI invocations** match current argparse / entry-point
  definitions.
- **Install extras** must match `pyproject.toml`. Curator splits
  each modality into `_cpu` and `_cuda12` variants — real names are
  `text_cpu` / `text_cuda12`, `video_cpu` / `video_cuda12`,
  `audio_cpu` / `audio_cuda12`, `image_cpu` / `image_cuda12`,
  `math_cpu` / `math_cuda12`, `interleaved_cpu` /
  `interleaved_cuda12`, `sdg_cpu` / `sdg_cuda12`, plus
  `deduplication_cuda12` and the aggregate `all`. Bare `text` /
  `video` / etc. do **not** exist.
- **Resource and hardware claims** reflect actual stage resource
  declarations.
- **Modality coverage.** `audio/`, `image/`, `interleaved/`,
  `math/`, `slurm/`, `synthetic/`, `text/`, `video/` — each should
  have at least one currently-runnable tutorial.
- **Cluster-scheduled examples** (`tutorials/slurm/`) stay aligned
  with current `RayClient` lifecycle. Slurm and other
  cluster-orchestration patterns are first-class, not afterthoughts.

## Contract Checklist

When this domain changes:

- `tutorials/` (entire tree)
- `tutorials/quickstart.py`
- `pyproject.toml` extras (mentions must match `<modality>_cpu` /
  `<modality>_cuda12` naming)
- `fern/` tutorial / how-to pages that reference these tutorials
- `README.md` — its quickstart aligns with `tutorials/quickstart.py`
- `CHANGELOG.md` if tutorial changes reflect user-visible API moves

## Advocate

- **CI smoke job** that imports/parses every tutorial Python file
  (catches `ImportError` early). Export notebooks to `.py` for
  static checks even if full execution is impractical.
- **"Tutorial freshness" report** flagging tutorials touching
  public APIs that changed since their last edit.
- **Tiny, license-clean fixtures** so tutorials run end-to-end on a
  single GPU/laptop where the modality permits.
- **Canonical structure across modality folders.**
- **End-to-end "I curated this corpus from scratch" path.** The
  Nemotron-CC recipe is the reference pattern; tutorials should
  point at it.

## Own

**Code:** `tutorials/` (entire tree); `tutorials/quickstart.py`
(cross-owned with the pipeline-contract steward — it is the canonical
runnable example).

**Tests:** static-import / parse checks for every tutorial Python
file (add a CI gate if not present); notebook execution where
practical.

**Docs (autopilot surface):** `tutorials/README.md`, per-modality
README files, `fern/` how-to and tutorial pages that reference these
tutorials.

**Agent artifacts:** `.claude/skills/getting-started/` — this
steward keeps the skill aligned with current tutorials. Apply the
Docs-First evaluation gate before expanding.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`.
`.github/CODEOWNERS` has no `tutorials/` entries today, so no
per-modality routing fires for tutorial changes — adding explicit
routes for tutorials touching dedup / classifiers / embedders /
video / synthetic is open advocacy.
