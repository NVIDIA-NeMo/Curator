# Steward: Tutorials & Examples

End-to-end runnable examples per modality. The single biggest "copy
this, it should work" trust surface — and the single biggest doc-rot
surface when public APIs evolve.

Related: root [AGENTS.md](../AGENTS.md),
[tutorials/README.md](README.md), `tutorials/quickstart.py`.

## Point Of View

The user's "does this actually work?" test. Every tutorial is a
contract that the imports, configs, CLI invocations, and pipeline
composition shown here resolve against the current `nemo_curator/`
public API.

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
- **Modality coverage.** `audio/`, `image/`, `interleaved/`, `math/`,
  `slurm/`, `synthetic/`, `text/`, `video/` — each should have at
  least one currently-runnable tutorial.
- **Slurm / cluster examples** (`tutorials/slurm/`) stay aligned
  with current `RayClient` lifecycle.

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

- CI smoke job that imports/parses every tutorial Python file
  (catches `ImportError` early). Export notebooks to `.py` for
  static checks even if full execution is impractical.
- A "tutorial freshness" report flagging tutorials touching public
  APIs that changed since their last edit.
- Tiny, license-clean fixtures so tutorials run end-to-end on a
  laptop where the modality permits.
- Canonical structure across modality folders.

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

**Agent artifacts:** `.claude/skills/getting-started/` — this steward
keeps the skill aligned with current tutorials. Apply the Docs-First
evaluation gate before expanding.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`.
`.github/CODEOWNERS` has no `tutorials/` entries today, so no
per-modality routing fires for tutorial changes — adding explicit
routes for tutorials touching dedup / classifiers / embedders /
video / synthetic is open advocacy.
