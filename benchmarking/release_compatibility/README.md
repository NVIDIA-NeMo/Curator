# Benchmark release compatibility

This folder contains explicit, temporary benchmark adapters for older Curator
releases. It never patches or replaces Curator-under-test. Select a profile with
`benchmarking/run.py --benchmark-compat-profile 26.07`; without a selection,
benchmarks use their normal API calls. Unknown profiles fail argument validation.
Config helpers receive an explicit `benchmark_compat_profile` keyword argument;
the runner forwards the CLI option only to affected benchmark scripts. No
environment variable selects a profile.

Each profile may also supply an accompanying YAML file. The shared config loader
merges it after all explicitly supplied workload, SKU, and CI configs, so CI job
generation and runtime execution use the same profile policy. Overrides only
affect entries present in the suite; they do not add missing entries. Individual
requirements use `enabled: false`, which the runner removes after merging.
Document-count and throughput checks remain in force. The run's `env.json`
records the profile's requirement overrides, not per-entry check outcomes.

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

## 26.07: dedup and TTS tagging

Exact/fuzzy dedup adapters retain the input, hashing, partition, and memory-pool
settings, reject normalization, and use the release's allocator implementation.
The log warns that asynchronous allocator selection is unavailable; allocator
differences must be considered in cross-release performance comparisons.
No requirements are relaxed for these entries.

TTS tagging supplies the older required `hf_token` argument from `HF_TOKEN`
(or `None` for cached/local authentication). The model, batching, and workload
are unchanged, and credentials are not added to benchmark parameters or logs.
The old ASR aligner has no CUDA-graph toggle. The adapter uses the release's
native NeMo decoding defaults and logs that explicit graph enablement is not
guaranteed; it rejects requests to disable graphs. Model, input, batching, and
output validation stay unchanged, but decoder implementation differences must
be considered in performance comparisons.
For Ray Data only, a benchmark subclass converts SQUIM's incoming array batch
to a list before calling the unchanged release implementation. This avoids
ambiguous array truth-value checks without dropping tasks or changing inference
batching, models, metrics, or output validation.
Targeted release-image testing is required before considering these adapters
validated end to end.

## 26.07: workload limitations

The baseline image records Curator revision
`4ad90ff7ada6579f23bdbfd69f71695d113483e9`. Inspection of that revision explains
these remaining failures beyond the first missing keyword or import. They are
not passing results and the profile does not disable their entries or checks.

| Entries | Release limitation | Why an argument/import shim is insufficient |
| --- | --- | --- |
| `minhash_document_batch_ray_data`, `minhash_document_batch_xenna` | `MinHashStage` processes `FileGroupTask`, not `DocumentBatch`. | Supplying a read format does not add the missing task interface. Substituting the file-group pipeline would measure a different execution mode. |
| `embedding_generation_raydata`, `embedding_generation_xenna` | The vLLM stage embeds a whole input batch and lacks `model_inference_batch_size`. | Removing `metadata_fields` alone exposes another missing argument. Dropping the script's 1024-row inference limit changes batching and memory behavior; implementing it in the adapter would backport stage behavior. |
| `audio_librispeech_xenna` | The newer `ASRStage` and NeMo adapter are absent. | The older `InferenceAsrNemoStage` lacks the requested local bucketing, duration limits, error policy, and decoder configuration. An import alias would silently omit those controls. |
| `semdedup_identification_xenna_dataset_size_ratio`, `semdedup_identification_xenna_fit_data_fraction` | KMeans output precision and pairwise compute precision cannot be selected. | Both workloads explicitly request float16 storage and computation. The old stages retain input-derived precision; dropping the controls is not equivalent. |
| `nemotron_parse_pdf_xenna`, `nemotron_parse_pdf_raydata` | The PDF reader lives in `composite.py`; inference hardcodes `max_tokens=9000`. | Fixing `DEFAULT_MAX_TOKENS` and the reader import alone cannot honor the current 8192-token default. Adapting the workload would require an explicitly different token budget. |
| `nemotron_parse_pdf_inference_server_ray_serve`, `nemotron_parse_pdf_inference_server_dynamo` | The PDF composite has only in-process inference, with no inference-server client integration. | Substituting local inference would no longer exercise the named server workflow. |

These limitations need an explicit decision about alternative release-specific
workloads before further adapters are added. Do not omit behavior or precision
controls without validating equivalence, or backport missing product features
into Curator-under-test merely to obtain a result. Model download/cache errors
and performance requirement misses are separate from these API limitations;
retest and investigate them without arbitrarily weakening checks.
