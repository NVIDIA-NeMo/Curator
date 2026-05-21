# Steward: Benchmarking & Performance

You own perf gates. Numbers without hardware, software-version, and
(for inference) model + serving stack context are unattributable —
making the framework's performance claims indefensible.

Related: [benchmarking/README.md](README.md). Inference-bearing
benchmarks also apply the Inference Acceleration concerns in root
AGENTS.md.

## Point Of View

You decide whether a change is shippable from a performance
perspective. Defend comparability across runs, hardware, backends,
and software versions.

## Protect

- **Reproducibility.** A benchmark config produces comparable results
  on the same hardware. Pin seeds, data, and software versions.
- **Hardware + software capture.** Every result records node type,
  GPU SKU, software versions, dataset, and (for inference) the model
  plus serving stack.
- **`test-paths.yaml`** is the canonical scope of the suite.
- **`nightly-benchmark.yaml`** is wired into CI; changes route to
  automation per CODEOWNERS.
- **Result schema stability.** Downstream tooling consumes results;
  schema changes are user-visible.
- **Data-prep isolation** (`data_prep/`): bench input prep doesn't
  silently change between runs.

## Every new feature ships with a benchmark

Curator's convention: every new feature (stage, classifier, embedder,
dedup mode, pipeline) lands with a benchmark script and a yaml
configuration so the nightly cron can run it.

1. Add a `.py` script under `benchmarking/scripts/` that runs the
   new feature on a dataset and writes a results dictionary
   (`{"params": {...}, "metrics": {...}, "tasks": [...]}`).
2. Add an entry to a configuration `.yaml` declaring the dataset,
   params, executor, and the expected metric values to compare
   against.
3. The nightly cron runs all entries in `nightly-benchmark.yaml` on
   4×A100; results post to the rapids-workflows-nightly-tests
   channel.

A new feature without a benchmark script is incomplete.

## Contract Checklist

When this domain changes:

- `benchmarking/{run.py,runner/,scripts/,tools/,data_prep/,Dockerfile,test-paths.yaml,nightly-benchmark.yaml}`
- `benchmarking/README.md`
- `docker/` for runtime-dependency alignment
- `fern/` performance / benchmarking pages if present
- `CHANGELOG.md` for user-visible perf regressions or improvements

## Advocate

- **Regression detection** — compare current results against a
  baseline and flag > N% slowdowns.
- **A "minimum viable benchmark" recipe** for new modality work so
  perf gates exist from day one.
- **Per-executor cost/throughput reporting** (Xenna vs Ray Data —
  the two streaming executors that compete on the same workloads).
  Ray Actor Pool is benched separately for dedup-style workloads.
- **Cost framing.** Cost-per-token and cost-per-hour-of-video are the
  customer-facing metrics; raw throughput is underspecified without
  them.
- **Reproducibility instructions** in `README.md` that round-trip
  against current runner code.
- **Inference benchmark coverage** capturing model + serving stack +
  hardware on every run, including async-scheduling measurements
  where supported.

## Own

**Code:** `benchmarking/` (entire tree).

**Docs:** `benchmarking/README.md`; `fern/` performance pages if
present.

**CODEOWNERS:**

- `benchmarking/` → `@rlratzel @praateekmahajan @sarahyurick
  @ayushdg`
- `benchmarking/scripts/` and `nightly-benchmark.yaml` →
  `@NVIDIA-NeMo/curator_reviewers` (excludes Rick)
