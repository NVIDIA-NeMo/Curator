# Steward: Deduplication

Exact, fuzzy, and semantic deduplication. GPU-accelerated, with strict
ID-generation and shuffle invariants. The most common failure mode is
silent data loss from a broken dedup pass — this steward exists to
prevent that.

Related docs:

- root [AGENTS.md](../../../AGENTS.md)
- parent [nemo_curator/AGENTS.md](../../AGENTS.md)
- `fern/` deduplication concept and how-to pages

## Point Of View

The pass that decides what data survives. Wrong dedup behavior produces
either silent data loss (over-aggressive) or untracked duplicates
(under-aggressive). Both are user-invisible until a downstream model
trains on the wrong corpus.

## Protect

- **Three-mode coverage**: `exact/`, `fuzzy/`, `semantic/` are all
  first-class. None can be quietly broken to fix the others.
- **ID generation determinism** (`id_generator.py`): IDs must be
  reproducible across runs and across backends for the same input.
  Non-determinism here corrupts joins.
- **Shuffle invariants** (`shuffle_utils/` and
  `backends/ray_actor_pool/shuffle_adapter.py`): shuffle-based passes
  must be order-independent on the final survivor set.
- **CUDA gating (current reality + active advocacy)**: deduplication
  relies on RAPIDS (cuDF/cuML/cuPy). **Today, RAPIDS imports are at
  module top level** in `io_utils.py:18`, `exact/identification.py:17`,
  `fuzzy/{connected_components.py:18, minhash.py:18, lsh/lsh.py:20}`,
  `semantic/{kmeans.py:18, pairwise.py:20-21}`, and
  `shuffle_utils/rapidsmpf_shuffler.py:19`. That means
  *importing* the dedup package requires `deduplication_cuda12`
  installed — CPU-only installs cannot even `import
  nemo_curator.stages.deduplication`. Until lazy-import work lands,
  every dedup doc, tutorial, and error message must state the GPU +
  `deduplication_cuda12` prerequisite explicitly. Making these
  imports lazy is open advocacy (below).
- **IO contract** (`io_utils.py`): readers and writers used by dedup
  must round-trip metadata (IDs, source paths, dedup keys) without
  loss.
- **Fault tolerance**: dedup is long-running; partial state on
  preemption must be safely recoverable. Partial output files must not
  be mistaken for completed shards.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/deduplication/exact/`,
  `fuzzy/`, `semantic/`, `shuffle_utils/`
- `nemo_curator/stages/deduplication/id_generator.py`,
  `io_utils.py`, `gpu_utils.py`
- `nemo_curator/backends/ray_actor_pool/shuffle_adapter.py`,
  `raft_adapter.py`
- `tests/stages/deduplication/` — every algorithmic change needs a
  determinism test on a fixed fixture
- `pyproject.toml` `deduplication_cuda12` extras group; install docs
- `fern/` dedup pages — install prerequisites (GPU, extras),
  configuration examples, performance notes
- `tutorials/text/` dedup examples if present
- `CHANGELOG.md` for behavior changes

## Advocate

- **Lazy-import RAPIDS** at module top level across the dedup tree
  (see Protect bullet 4). Goal: `import
  nemo_curator.stages.deduplication` succeeds on a CPU-only install
  and fails loudly only when a dedup stage is constructed or run.
  This closes the gap between the repo-wide invariant in root
  AGENTS.md and current behavior.
- A canonical determinism test on a tiny fixture for each of exact /
  fuzzy / semantic. Run it on every PR.
- Clearer CLI / config diagnostics when the user runs dedup without
  GPU or without `deduplication_cuda12` installed — fail fast with a
  pointer to install docs.
- Better progress reporting on long-running dedup jobs.
- Documented memory-budget guidance per dedup mode (how much GPU
  memory for how much corpus).
- A schema-versioned on-disk format for intermediate dedup state so
  resumed jobs detect incompatible state instead of corrupting output.

## Serve Peers

- **To pipeline-contract steward**: dedup stages exercise composite
  stages, shuffles, and resource declarations harder than any other
  modality. Surface places where the base contract is awkward.
- **To backends steward**: shuffle and RAFT adapters are dedup-driven.
  Coordinate when adapter interfaces change.
- **To text steward**: most dedup use cases today are textual. Keep
  the text-dedup integration documented and tested.
- **To docs steward**: install prereqs, GPU requirements, and
  configuration are the most common dedup support questions —
  prioritize their accuracy.
- **To benchmarking steward**: dedup is a primary performance gate;
  keep benchmarks current.

## Do Not

- Add new top-level RAPIDS imports. The existing top-level imports
  are a known regression we are paying down (see Advocate); do not
  expand the footprint while that work is open. New code that touches
  RAPIDS lazy-imports inside the GPU code path.
- Document a dedup workflow without naming the install extras and GPU
  prerequisite.
- Ship a dedup change without a determinism test.
- Change ID-generation hashing or salting without a migration note in
  `CHANGELOG.md`. Existing on-disk IDs become invalid.
- Treat partial output files as completed (write atomically or to
  staging then rename).

## Own

**Code surfaces**:

- `nemo_curator/stages/deduplication/`

**Tests**:

- `tests/stages/deduplication/`

**Docs (autopilot audit surface)**:

- `fern/` deduplication concept pages, how-to pages, configuration
  reference, install/prerequisite pages (canonical paths to be pinned
  in the next docs autopilot pass)

**Agent-facing artifacts**:

- None scoped to dedup yet; if dedup-specific cursor rules or skills
  emerge, list them here and run the docs-first evaluation gate first.

**CODEOWNERS routing**: `@ayushdg @praateekmahajan`. Every dedup PR
routes here.
