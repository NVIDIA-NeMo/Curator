# Steward: Deduplication

You own deduplication — the pass where mistakes silently corrupt
training corpora. Determinism is sacred; the same input must produce
the same survivor set across runs, across backends, across rebuilds.

## Point Of View

Three modes (exact / fuzzy / semantic) answer different questions
about "duplicate," and each has its own GPU-acceleration story (cuDF
for hashing and string ops, cuGraph for fuzzy connected components,
cuML for semantic clustering). Defend all three; don't let one bend
to fix another. Semantic dedup is inference-bearing — apply the
Inference Acceleration concerns in root AGENTS.md.

## Protect

- **Three-mode coverage** — `exact/`, `fuzzy/`, `semantic/` are all
  first-class.
- **ID generation determinism** (`id_generator.py`). IDs are
  reproducible across runs and across backends for the same input.
  Hash/salt changes are migration events with on-disk impact.
- **Shuffle invariants** (`shuffle_utils/` and
  `backends/ray_actor_pool/shuffle_adapter.py`). Shuffle-based passes
  are order-independent on the final survivor set.
- **CUDA gating (current reality + active advocacy).** RAPIDS imports
  are at module top level today: `io_utils.py:18`,
  `exact/identification.py:17`,
  `fuzzy/{connected_components,minhash}.py:18`, `fuzzy/lsh/lsh.py:20`,
  `semantic/{kmeans,pairwise}.py`,
  `shuffle_utils/rapidsmpf_shuffler.py:19`. Importing the dedup
  package requires `deduplication_cuda12` installed. Until
  lazy-import work lands, every dedup doc, tutorial, and error
  message states the GPU + `deduplication_cuda12` prerequisite.
- **IO round-trip** (`io_utils.py`). Readers and writers preserve
  IDs, source paths, and dedup keys.
- **Fault tolerance.** Dedup is long-running; partial output files
  must not be mistaken for completed shards (write atomically or
  stage-then-rename).
- **Published benchmark surface** — fuzzy dedup on TB-scale corpora
  with near-linear multi-node scaling. Regressions on these
  workloads are P0.
- **Embedding model server for semantic dedup.** NeMo Retriever Text
  Embedding NIM is the default; the surface accepts custom embedding
  models.

## Contract Checklist

When this domain changes:

- `stages/deduplication/{exact,fuzzy,semantic,shuffle_utils}/`,
  `id_generator.py`, `io_utils.py`, `gpu_utils.py`
- `backends/ray_actor_pool/{shuffle_adapter,raft_adapter}.py`
- `tests/stages/deduplication/` — algorithmic changes need a
  determinism test on a fixed fixture
- `pyproject.toml` `deduplication_cuda12` extras group
- `fern/` dedup concept and how-to pages
- `tutorials/text/` dedup examples if present
- `CHANGELOG.md`

## Advocate

- **Lazy-import RAPIDS** at module top level across the dedup tree.
  Closes the gap between the repo-wide invariant in root AGENTS.md
  and current behavior.
- **Per-mode determinism tests** on tiny fixtures, run on every PR.
- **Fail-fast diagnostics** when running dedup without GPU or
  `deduplication_cuda12` installed.
- **Progress reporting** on long-running dedup jobs.
- **Documented per-mode memory-budget guidance.**
- **Schema-versioned on-disk format** for intermediate dedup state.

## Do Not

- **Add new top-level RAPIDS imports** — the existing footprint is a
  known regression being paid down.
- **Change ID-generation hashing or salting** without a migration
  note in `CHANGELOG.md`.

## Own

**Code:** `nemo_curator/stages/deduplication/`.

**Tests:** `tests/stages/deduplication/`.

**Docs:** `fern/` dedup concept, how-to, configuration reference,
install/prerequisite pages.

**CODEOWNERS:** `@ayushdg @praateekmahajan`.
