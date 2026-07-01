# Steward: Tutorials & Examples

You own runnable examples. First impressions matter more than internal
consistency — a tutorial that runs cleanly on first try recruits the
user; one that fails on imports or wrong extras loses them.

Related: [tutorials/README.md](README.md), `tutorials/quickstart.py`.

## Point Of View

The user's "does this actually work?" test. Your users run on
cluster-scheduled infrastructure (Slurm, Kubernetes, multi-node Ray)
as often as on a single machine — but the first impression usually
happens on a laptop or single GPU, so both have to work. Every
tutorial is a contract that the imports, configs, CLI invocations,
and pipeline composition shown here resolve against the current
public API.

## Protect

- **Imports resolve.** Every `from nemo_curator…` line in a tutorial
  resolves against the current installed package.
- **Pipeline composition type-checks** against current
  `ProcessingStage` and `Task` types.
- **CLI invocations** match current argparse / entry-point
  definitions.
- **Install extras** match `pyproject.toml`. Curator splits each
  modality into `_cpu` and `_cuda12` variants — real names are
  `text_cpu` / `text_cuda12`, `video_cpu` / `video_cuda12`,
  `audio_cpu` / `audio_cuda12`, `image_cpu` / `image_cuda12`,
  `math_cpu` / `math_cuda12`, `interleaved_cpu` /
  `interleaved_cuda12`, `sdg_cpu` / `sdg_cuda12`, plus
  `deduplication_cuda12` and the aggregate `all`. Bare `text` /
  `video` / etc. do **not** exist.
- **Resource and hardware claims** reflect actual stage resource
  declarations.
- **Modality coverage.** `audio/`, `image/`, `interleaved/`,
  `math/`, `slurm/`, `synthetic/`, `text/`, `video/` — each has at
  least one currently-runnable tutorial.
- **Cluster-scheduled examples** (`tutorials/slurm/`) stay aligned
  with current `RayClient` lifecycle.

## Contract Checklist

When this domain changes:

- `tutorials/` (entire tree)
- `tutorials/quickstart.py`
- `pyproject.toml` extras (mentions match `<modality>_cpu` /
  `<modality>_cuda12` naming)
- `fern/` tutorial / how-to pages that reference these tutorials
- `README.md` — its quickstart aligns with `tutorials/quickstart.py`
- `CHANGELOG.md` if tutorial changes reflect user-visible API moves

## Advocate

- **Ruff on `tutorials/`** so tutorial code stays lint-clean. CI
  doesn't execute tutorials; the lint pass is the gate.
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

**Docs (discover by grep — see root AGENTS.md *Impacted-Docs
Discovery*):** when changing a tutorial or its dependencies, search
`fern/`, `tutorials/README.md`, per-modality `tutorials/<mod>/`
READMEs, `README.md` (quickstart section), `.cursor/rules/`, and
`.github/copilot-instructions.md` for:

- Tutorial filenames (e.g. `quickstart.py`,
  `tutorials/synthetic/nemotron_cc/`)
- Extras names: `<modality>_cpu`, `<modality>_cuda12`, `all`,
  `deduplication_cuda12`
- Classes / functions the tutorial imports if they were renamed or
  moved
- Cluster-orchestration patterns (Slurm, `RayClient`) if changing
  cluster examples
- The Nemotron-CC recipe link / reference if changing the canonical
  end-to-end path

Conceptual changes (reshaping the modality coverage map, IA shifts
in how-to pages) delegate to the Docs Steward.

**Agent artifacts:** `.claude/skills/getting-started/`. Apply the
Docs-First evaluation gate before expanding.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`.
`.github/CODEOWNERS` has no `tutorials/` entries today, so no
per-modality routing fires for tutorial changes.
