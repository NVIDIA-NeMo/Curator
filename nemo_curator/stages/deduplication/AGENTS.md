# Steward: Deduplication

This domain exists because deduplication is the pass where mistakes
silently corrupt training corpora. Over-aggressive dedup quietly drops
data; under-aggressive dedup leaves duplicates untracked. Both fail
invisibly until a downstream model trains badly. Curator's bet here is
that RAPIDS-accelerated dedup at multi-node multi-GPU scale removes
the iterative CPU bottleneck that otherwise dominates trillion-token
corpus preparation.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md).

## Point Of View

The pass that decides what data survives. Determinism is sacred: the
same input must produce the same survivor set across runs, across
backends, across rebuilds. The three modes (exact / fuzzy / semantic)
exist because they answer different questions about "duplicate," and
each has its own GPU-acceleration story (cuDF for hashing and string
ops, cuGraph for fuzzy connected components, cuML for semantic
clustering).

## Protect

- **Three-mode coverage** — `exact/`, `fuzzy/`, `semantic/` are all
  first-class. None can be quietly broken to fix the others.
- **ID generation determinism** (`id_generator.py`). IDs must be
  reproducible across runs and across backends for the same input.
  Non-determinism corrupts downstream joins. Hash/salt changes are
  migration events with on-disk impact.
- **Shuffle invariants** (`shuffle_utils/` and
  `backends/ray_actor_pool/shuffle_adapter.py`). Shuffle-based passes
  must be order-independent on the final survivor set.
- **CUDA gating (current reality + active advocacy).** RAPIDS imports
  are at module top level today: `io_utils.py:18`,
  `exact/identification.py:17`,
  `fuzzy/{connected_components,minhash}.py:18`, `fuzzy/lsh/lsh.py:20`,
  `semantic/{kmeans,pairwise}.py`,
  `shuffle_utils/rapidsmpf_shuffler.py:19`. Importing the dedup
  package therefore requires `deduplication_cuda12` installed. Until
  lazy-import work lands, every dedup doc, tutorial, and error
  message must state the GPU + `deduplication_cuda12` prerequisite.
- **IO round-trip** (`io_utils.py`). Readers and writers used by
  dedup preserve IDs, source paths, and dedup keys.
- **Fault tolerance.** Dedup is long-running; partial state on
  preemption must be safely recoverable. Partial output files must
  not be mistaken for completed shards (write atomically or
  stage-then-rename).
- **Published benchmark surface** — fuzzy dedup on TB-scale corpora
  with near-linear multi-node scaling. The defended pattern is
  GPU-accelerated dedup outperforming CPU-based alternatives by an
  order of magnitude; regressions on these workloads are P0.
- **Embedding model server for semantic dedup.** NeMo Retriever Text
  Embedding NIM is the default; the surface accepts custom embedding
  models. Semantic dedup is inference-bearing — coordinate with the
  Inference Acceleration Steward.

## Contract Checklist

When this domain changes:

- `stages/deduplication/{exact,fuzzy,semantic,shuffle_utils}/`,
  `id_generator.py`, `io_utils.py`, `gpu_utils.py`
- `backends/ray_actor_pool/{shuffle_adapter,raft_adapter}.py`
- `tests/stages/deduplication/` — algorithmic changes need a
  determinism test on a fixed fixture
- `pyproject.toml` `deduplication_cuda12` extras group
- `fern/` dedup concept and how-to pages (install prereqs, GPU
  requirements, configuration)
- `tutorials/text/` dedup examples if present
- `CHANGELOG.md`

## Advocate

- **Lazy-import RAPIDS** at module top level across the dedup tree.
  Goal: `import nemo_curator.stages.deduplication` succeeds on a
  CPU-only install and fails loudly only when a dedup stage is
  constructed or run. Closes the gap between the repo-wide invariant
  in root AGENTS.md and current behavior.
- **Per-mode determinism tests** on tiny fixtures (exact / fuzzy /
  semantic), run on every PR.
- **Fail-fast diagnostics** when running dedup without GPU or
  `deduplication_cuda12` installed.
- **Progress reporting** on long-running dedup jobs.
- **Documented per-mode memory-budget guidance** (GPU memory vs
  corpus size).
- **Schema-versioned on-disk format** for intermediate dedup state so
  resumed jobs detect incompatible state instead of corrupting
  output.

## Do Not

- **Add new top-level RAPIDS imports** — the existing footprint is a
  known regression being paid down. New code lazy-imports inside the
  GPU path.
- **Change ID-generation hashing or salting** without a migration
  note in `CHANGELOG.md` — existing on-disk IDs become invalid.

## Own

**Code:** `nemo_curator/stages/deduplication/`.

**Tests:** `tests/stages/deduplication/`.

**Docs (autopilot surface):** `fern/` dedup concept, how-to,
configuration reference, install/prerequisite pages.

**CODEOWNERS:** `@ayushdg @praateekmahajan`.
