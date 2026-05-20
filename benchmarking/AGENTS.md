# Steward: Benchmarking & Performance

This domain exists because the framework's performance claims need to
be defensible and reproducible. Curator's value proposition rests on
specific speedup numbers against CPU baselines and near-linear scaling
across multi-node multi-GPU setups. Benchmarks that aren't reproducible
— or aren't captured with hardware and software context — are noise
that erodes the claims they're meant to support.

Related: root [AGENTS.md](../AGENTS.md),
[benchmarking/README.md](README.md),
[ALM_BENCHMARK.md](ALM_BENCHMARK.md),
[AUDIO_PROFILING.md](AUDIO_PROFILING.md).

## Point Of View

The numbers that decide whether a change is shippable from a
performance perspective. Defends comparability across runs, hardware,
backends, and software versions. Special concern for
inference-bearing benchmarks: model + serving stack + hardware are
always captured; otherwise the result is unattributable.

## Protect

- **Reproducibility.** A benchmark config produces comparable results
  when re-run on the same hardware. Pin seeds, data, and software
  versions where it matters.
- **Hardware + software capture.** Every result records node type,
  GPU SKU, software versions, dataset, and (for inference) the model
  plus serving stack. Numbers without this context cannot be compared.
- **`test-paths.yaml`** is the canonical scope of the suite.
- **`nightly-benchmark.yaml`** is wired into CI; changes route to
  automation per CODEOWNERS.
- **Result schema stability.** Downstream tooling consumes results;
  schema changes are user-visible.
- **Data-prep isolation** (`data_prep/`): bench input prep must not
  silently change between runs.

## Contract Checklist

When this domain changes:

- `benchmarking/{run.py,runner/,scripts/,tools/,data_prep/,Dockerfile,test-paths.yaml,nightly-benchmark.yaml}`
- `benchmarking/{ALM_BENCHMARK,AUDIO_PROFILING,README}.md`
- `docker/` for runtime-dependency alignment
- `fern/` performance / benchmarking pages if present
- `CHANGELOG.md` for user-visible perf regressions or improvements

## Advocate

- **Regression detection** — compare current results against a
  baseline and flag > N% slowdowns.
- **A "minimum viable benchmark" recipe** for new modality work so
  perf gates exist from day one.
- **Per-executor cost/throughput reporting** (Xenna vs Ray Data vs
  Ray Actor Pool).
- **Cost framing.** Cost-per-token and cost-per-hour-of-video are
  the customer-facing metrics; raw throughput is underspecified
  without them.
- **Reproducibility instructions** in `README.md` that round-trip
  against current runner code.
- **Inference benchmark coverage** that captures model + serving
  stack + hardware on every run, including async-scheduling
  measurements where supported (coordinate with the Inference
  Acceleration Steward in root AGENTS.md).

## Own

**Code:** `benchmarking/` (entire tree).

**Docs (autopilot surface):** `benchmarking/README.md`,
`ALM_BENCHMARK.md`, `AUDIO_PROFILING.md`; `fern/` performance pages
if present.

**CODEOWNERS:**

- `benchmarking/` → `@rlratzel @praateekmahajan @sarahyurick
  @ayushdg`
- `benchmarking/scripts/` and `nightly-benchmark.yaml` →
  `@NVIDIA-NeMo/curator_reviewers` (excludes Rick)
