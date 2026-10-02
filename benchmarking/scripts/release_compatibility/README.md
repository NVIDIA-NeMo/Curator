# Benchmark release compatibility

This folder contains explicit, temporary benchmark adapters for older Curator
releases. It never patches or replaces Curator-under-test. Select a profile with
`CURATOR_BENCHMARK_COMPAT_PROFILE=26.07`; without a selection, benchmarks use
their normal API calls. Unknown profiles fail setup and affected scripts.

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

Audio, embedding, PDF, exact/fuzzy dedup, and semantic dedup compatibility remain
to be reviewed separately. Do not omit behavior or precision controls without
validating equivalence. Unsupported product features are not comparable and
must not be backported into Curator-under-test just to obtain a result.
