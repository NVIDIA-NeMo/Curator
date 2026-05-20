# Steward: Tutorials & Examples

End-to-end runnable examples per modality. The single biggest source
of "copy this, it should work" trust — and the single biggest doc-rot
surface when public APIs evolve.

Related docs:

- root [AGENTS.md](../AGENTS.md)
- [tutorials/README.md](README.md)
- `tutorials/quickstart.py`

## Point Of View

The user's "does this actually work?" test. Every tutorial is a
contract with users that the imports, configs, CLI invocations, and
pipeline composition shown here resolve against the current
`nemo_curator/` public API.

## Protect

- **Imports resolve**: every `from nemo_curator…` line in a tutorial
  must resolve against the current installed package.
- **Pipeline composition type-checks**: `pipeline.add_stage(...)`
  examples must produce a runnable pipeline against current
  `ProcessingStage` and `Task` types.
- **CLI invocations**: any shell or notebook line invoking a Curator
  CLI / script must match the current argparse / entry-point
  definition.
- **Install steps**: extras must match the current `pyproject.toml`
  extras groups. Curator splits each modality into `_cpu` and
  `_cuda12` variants — the real names are `text_cpu` / `text_cuda12`,
  `video_cpu` / `video_cuda12`, `audio_cpu` / `audio_cuda12`,
  `image_cpu` / `image_cuda12`, `math_cpu` / `math_cuda12`,
  `interleaved_cpu` / `interleaved_cuda12`, `sdg_cpu` / `sdg_cuda12`,
  plus `deduplication_cuda12` and the aggregate `all`. Bare `text` /
  `video` / etc. do **not** exist.
- **Resource and hardware claims**: a "needs N GPUs of M GB" claim
  must reflect actual stage resource declarations.
- **Modality coverage**: `audio/`, `image/`, `interleaved/`, `math/`,
  `slurm/`, `synthetic/`, `text/`, `video/` — each modality folder
  should have at least one currently-runnable tutorial.
- **Slurm / cluster examples** (`tutorials/slurm/`): keep
  cluster-launch scripts aligned with current `RayClient` lifecycle.

## Contract Checklist

When this domain changes:

- `tutorials/` (entire tree)
- `tutorials/quickstart.py`
- `pyproject.toml` extras (mentions must match — `<modality>_cpu` /
  `<modality>_cuda12` naming, see Protect)
- `fern/` tutorial / how-to pages that reference these tutorials
- `README.md` — its quickstart must align with `tutorials/quickstart.py`
- `CHANGELOG.md` if tutorial changes reflect user-visible API moves

## Advocate

- A CI smoke job that imports and parses every tutorial Python file
  (catches `ImportError` early). Notebooks can be exported to .py
  for static checks even if full execution is impractical.
- A "tutorial freshness" report that flags tutorials touching public
  APIs that changed since the tutorial was last edited.
- Tiny, license-clean fixtures so tutorials can run end-to-end on a
  laptop where the modality permits.
- A canonical structure across modality folders so users moving
  between modalities find the same shape.

## Serve Peers

- **To pipeline-contract steward**: tutorials exercise the public
  API harder than tests. Surface awkwardness early.
- **To modality stewards**: own the tutorial(s) for your modality.
  Tutorials touching your modality must round-trip against your
  current public API.
- **To docs steward**: tutorials are linked from `fern/` and embedded
  as snippets — keep them in sync.

## Do Not

- Ship a tutorial that imports a private (`_`-prefixed) module.
- Pin to a version of a model artifact that has been removed.
- Use real network calls without a `# requires network` note and a
  graceful failure path.
- Document install extras that do not exist in `pyproject.toml`.
- Hardcode absolute paths or contributor-specific environment
  assumptions.

## Own

**Code surfaces**:

- `tutorials/` (entire tree)
- `tutorials/quickstart.py` (cross-owned with the pipeline-contract
  steward — it is the canonical runnable example)

**Tests**:

- Static-import / parse checks for every tutorial Python file (add
  a CI gate if not present)
- Notebook execution where practical

**Docs (autopilot audit surface)**:

- `tutorials/README.md`
- Per-modality tutorial README files
- `fern/` how-to and tutorial pages that reference these tutorials
  (to be pinned in the next docs autopilot pass)

**Agent-facing artifacts**:

- `.claude/skills/getting-started/` — this skill points at tutorials;
  keep aligned. Apply the Docs-First Agent Artifact Evaluation gate
  (root AGENTS.md) before expanding.

**CODEOWNERS routing**: default `@NVIDIA-NeMo/curator_reviewers`.
Tutorials touching a modality with named CODEOWNERS (deduplication,
classifiers, embedders, video, synthetic) route additionally to
those teams.
