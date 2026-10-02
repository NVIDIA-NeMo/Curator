# Benchmark release compatibility

This folder contains explicit, temporary benchmark adapters for older Curator
releases. It never patches or replaces Curator-under-test. Select a profile with
`CURATOR_BENCHMARK_COMPAT_PROFILE=26.07`; without a selection, benchmarks use
their normal API calls. Unknown profiles fail setup and affected scripts.

Each profile may also supply an accompanying YAML file. The shared config loader
merges it after all explicitly supplied workload, SKU, and CI configs, so CI job
generation and runtime execution use the same profile policy. Overrides only
affect entries present in the suite; they do not add missing entries. Individual
requirements use `enabled: false`, which the runner removes after merging.
Document-count and throughput checks remain in force. The run's `env.json`
records the profile's unavailable checks as not evaluated, never as passing.

Each release gets a plain Python module with focused functions, not a registry
or class hierarchy. Scripts branch explicitly at the affected API call. Keep
the same benchmark suite revision and workload config across releases being
compared. The selected profile is recorded in `env.json`.

## 26.07: initial MinHash coverage

Target image: `nvcr.io/nvidian/nemo-curator:26.06.rc7`, the current 26.07 baseline
substitute. Profile selection is explicit, not inferred from the image tag.

`create_minhash_stage` omits the unsupported `normalize_text` keyword only for
normalization-disabled workloads. It preserves input selection, hash settings,
batching, and workers. Normalization-enabled workloads fail explicitly.

Affected entries:
- `minhash_document_batch_ray_data`
- `minhash_document_batch_xenna`
- `minhash_file_group_task_ray_actors`

This addresses their first observed failure, not a claim that release execution
now passes. Unit tests verify argument forwarding and rejection of unsupported
normalization. Release-image validation of output cardinality and signatures is
still required before treating timings as comparable.

The accompanying `curator_26_07.yaml` disables the three custom worker timing
requirements for all six currently named MinHash entries. The release lacks that
instrumentation. MinHash scripts omit unavailable timing metrics rather than
fabricating zeros; elapsed time, throughput, and document counts remain available.

Audio, embedding, PDF, exact/fuzzy dedup, and semantic dedup compatibility remain
to be reviewed separately. Do not omit behavior or precision controls without
validating equivalence. Unsupported product features are not comparable and
must not be backported into Curator-under-test just to obtain a result.
