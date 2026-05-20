# Steward: Benchmarking & Performance

Performance gates and profiling for NeMo Curator. Covers the nightly
benchmark suite, ALM (audio language model) and audio profiling, and
the runners/tools that produce comparable numbers across runs.

Related docs:

- root [AGENTS.md](../AGENTS.md)
- [benchmarking/README.md](README.md)
- [benchmarking/ALM_BENCHMARK.md](ALM_BENCHMARK.md)
- [benchmarking/AUDIO_PROFILING.md](AUDIO_PROFILING.md)

## Point Of View

The numbers that decide whether a change is shippable from a
performance perspective. Defends comparability across runs, hardware,
and backends — without comparability, benchmarks become noise.

## Protect

- **Run reproducibility**: a benchmark configuration must produce
  comparable results when re-run on the same hardware. Pin seeds,
  data, and software versions where it matters.
- **Hardware capture**: every result must record what it ran on.
  Numbers without hardware context cannot be compared.
- **`benchmarking/test-paths.yaml`** is the canonical list of what
  the suite runs; PRs that change scope update it.
- **`nightly-benchmark.yaml`** is wired into CI; changes there
  require automation team review (per CODEOWNERS).
- **Result format stability**: downstream tooling consumes results;
  schema changes are user-visible.
- **Data preparation isolation** (`data_prep/`): bench input
  preparation must not silently change between runs.

## Contract Checklist

When this domain changes:

- `benchmarking/run.py`, `benchmarking/runner/`,
  `benchmarking/scripts/`, `benchmarking/tools/`
- `benchmarking/test-paths.yaml`
- `benchmarking/nightly-benchmark.yaml`
- `benchmarking/data_prep/`, `benchmarking/tools/`
- `benchmarking/Dockerfile` — keep aligned with `docker/`
- `benchmarking/ALM_BENCHMARK.md`, `AUDIO_PROFILING.md`, `README.md`
- `fern/` performance / benchmarking pages if present
- `CHANGELOG.md` for user-visible perf regressions or improvements

## Advocate

- A regression-detection step that compares current results against a
  baseline and flags > N% slowdowns.
- A "minimum viable benchmark" recipe for new modality work so perf
  gates exist from day one.
- Clearer cost/throughput reporting per executor (Xenna vs Ray Data
  vs Ray Actor Pool).
- Reproducibility instructions in `README.md` that round-trip
  against current runner code.

## Serve Peers

- **To deduplication, video, text stewards**: provide perf signal
  when their stages change.
- **To backends steward**: cross-executor perf parity is a primary
  concern; surface where one executor falls behind another.
- **To docs steward**: keep perf numbers in `fern/` aligned with
  current bench output; flag stale numbers.

## Do Not

- Compare numbers across different hardware or software versions
  without flagging the change.
- Land a perf-affecting change without re-running affected
  benchmarks.
- Edit `nightly-benchmark.yaml` or `scripts/` without routing to
  `@NVIDIA-NeMo/curator_reviewers` per CODEOWNERS.
- Use private model artifacts or licensed data that cannot be
  re-run by other contributors.

## Own

**Code surfaces**:

- `benchmarking/` (entire tree, minus `scripts/` and
  `nightly-benchmark.yaml` which route to `curator_reviewers`)

**Tests**: benchmark runs themselves; surface failures as test
failures where practical.

**Docs (autopilot audit surface)**:

- `benchmarking/README.md`, `ALM_BENCHMARK.md`, `AUDIO_PROFILING.md`
- `fern/` benchmarking / performance pages if present (to be pinned
  in the next docs autopilot pass)

**Agent-facing artifacts**: none scoped to benchmarking yet.

**CODEOWNERS routing**:

- `benchmarking/`: `@rlratzel @praateekmahajan @sarahyurick @ayushdg`
- `benchmarking/scripts/` and `benchmarking/nightly-benchmark.yaml`:
  `@NVIDIA-NeMo/curator_reviewers` (excludes Rick)
