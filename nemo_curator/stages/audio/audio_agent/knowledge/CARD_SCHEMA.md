<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Capability Card Schema (v2)

A capability **card** is the agent's factual, code-independent view of one stage. It lets
the host LLM select, configure, and compose a stage **without reading source**. Cards live
in `knowledge/cards/<name>.yaml`, one per stage, keyed by `stage_id`.

The **conformance gate** (`card_conformance.py`) checks declared mechanical facts such as
stage references and parameter names. Prose, parameter values and model claims still need
review against code/contracts and model evidence; a passing gate does not prevent all drift.

## Golden rule: never fabricate

Fill only what you can verify from the stage's code/contract or an authoritative model card.
If you don't know a value, **leave it empty (null / omit)** and state any decision-relevant
uncertainty in `notes` or `caveats`. YAML comments are not returned to the agent. A wrong
fact is worse than a missing one.

## Honesty tiers (`verified`, required)

Every card declares how each fact group was established:

| tier         | meaning                                                              | examples                          |
|--------------|---------------------------------------------------------------------|-----------------------------------|
| `mechanical` | derived from code/contract; the gate re-checks it                   | `params`, `tags`, `category`      |
| `measured`   | from a real run/benchmark on known hardware/data                    | `resource.gpu_mem_gb`, throughput |
| `best_guess` | author judgment, not yet verified                                   | `use_cases`, `domain`, sweetspots |

```yaml
verified: {params: mechanical, resource: best_guess, model_version: measured, use_cases: best_guess}
```

## Fields

| field | required | notes |
|-------|----------|-------|
| `stage_id` | yes | exact stage class name (must resolve). |
| `category` | yes | one taxonomy category (`ingest`/`preprocess`/`segment`/`diarize`/`transcribe`/`quality`/`filter`/`text_norm`/`export`/`alm`). |
| `summary` | yes | one line: what it does. |
| `tags` | rec. | capability flags: `needs_gpu`, `needs_ffmpeg`, `needs_internet_first_run`, `needs_hf_token`, `writes_disk`, `sink`, `sanitizes_output`, `produces_score`, `is_filter`, `fanout`, `batch_only`. |
| `model_id` | if model | model identifier, else `null`. |
| `model_version` | opt. | informational revision/entrypoint, when known. An entrypoint or release label is not an immutable pin. The gate does not require this field or verify that the runtime loads a pinned revision. |
| `deterministic` | opt. | `false` when the same inputs can legitimately give different output (unseeded randomness, order-dependent decode). Omitted means reproducible. Execution reuse offers a stored result from such a stage with the caveat shown and *fresh* pre-selected, rather than serving it silently — so declaring it costs a prompt, not the feature. |
| `domain` | rec. | `{language, style}` — usually `best_guess`. |
| `constraints` | rec. | only **real** facts: `supported_sample_rates`, `max_speakers`, `batch_size:{fixed,reason}`, `input_duration_sweetspot_sec:{min,max}`. |
| `resource` | rec. | `{cpus, gpu_mem_gb, host_mem_gb, gpu_optional, bound: cpu\|gpu\|io, throughput_hint, disk_expansion}`. Feeds the resource planner. |
| `use_cases` | rec. | `{good_for: [...], avoid_for: [...]}` — `best_guess`. |
| `composition` | rec. | `{typical_upstream: [...], typical_downstream: [...]}` — idiomatic ordering. Every stage id must resolve (gate-checked). Optional `incompatible_upstream: {stage_id: why}` records a pairing that is known to *break*, not merely be unidiomatic — a stage id may not appear in both, and the reason is required, because "don't use X" without "because X writes a list where this reads an int" is a rule the next reader will discount. |
| `params_of_note` | rec. | `{param: description}` — **keys must be real constructor params** (gate-checked). |
| `presets` | opt. | `{name: {param: value}}` — **keys must be real params** (gate-checked). |
| `metrics` | opt. | `{metric: {scale:{min,max,direction}, threshold_param, valid_range:[lo,hi], presets}}` — score directions/targets for configuration. `scale.direction` must be `higher_better`/`lower_better`; `threshold_param` (if set) must be a real param; `valid_range` must be `[lo, hi]` (all gate-checked). Omit `threshold_param` for annotate-only stages filtered downstream. A generic selector needs a compatible scalar key; nested metric outputs may need extraction first. |
| `decision` | opt. | A mechanically checked declaration that annotation is already separate from a downstream selector. The primary declaration remains task-scoped for compatibility; optional full `variants` may declare an exact one-level `segments` scope. See below. |
| `semantic_facts` | opt. | Advisory mapping from an externally consumed output/concept to prose facts such as `{meaning, unit, provenance, scope, propagation, counterexamples}`. It helps the host reason across filters, fan-out and aggregation; it is deliberately not a deterministic ontology or runtime gate. |
| `io` | opt. | Parameter bindings for existing local-source and persisted-output protocols. Requires `verified.io: mechanical`; see below. No stage implementation or contract change is required. |
| `planning` | opt. | Tested, configured facts that cannot be inferred from parameter names or cardinality. Requires `verified.planning: mechanical`; see below. Does not change execution/reuse identity. |
| `versions` | opt. | model-backed stages only: `{model_id: "when-to-use"}` for checkpoints verified interchangeable via `model_name`/`model_path` (same output structure, no module code change). Any version-selecting `preset` (one that sets `model_name`/`model_path`) **must list its model id here** (gate-checked). Mark `verified.versions` `measured` when empirically tested. |
| `conflicts_with` | opt. | stage_ids that are alternatives / shouldn't co-occur. |
| `param_dependencies` | opt. | notes on params that depend on each other or on upstream data. |
| `comparison` | opt. | disambiguation fields for overlapping modules: `{language_support, accuracy_hint, latency_hint, config_complexity, known_limitations}`. |
| `notes` / `caveats` | opt. | free text. |
| `provenance` | yes | `{model_card_url, card_version, last_validated}`. |

## Existing I/O protocols

`io` lets a newly registered stage use existing agent source/output handling
without adding its class name to agent code. It is an opt-in claim about actual
behavior, not permission to infer support from a parameter name.

```yaml
io:
  source:                           # only for a source stage
    kind: manifest                  # one local .json/.jsonl file of object rows
    path_param: catalog_path
    audio_filepath_key_param: audio_filepath_key   # optional; default audio_filepath
    include_files_param: selected_paths            # optional exact file selection
  outputs:                          # only enabled contract-declared output paths count
    destination:
      kind: manifest                # JSONL; other kinds: audio_dir, rttm_dir,
                                    # text_dir, archive, unknown
      primary: true                 # at most one; selects the reusable artifact
      per_item: true                # exactly one saved row per returned logical task
      relocatable: true             # location does not enter output rows/asset identity
  dependency_params: [weights_file]  # optional local file/directory dependencies, including defaults
verified:
  io: mechanical
```

All enabled outputs are required deliverables, including non-primary and
`relocatable` outputs. The primary selects the resume entry and, when declared
`per_item` on a final `1:1` writer, the manifest read back for acceptance evidence.
It does not make its siblings optional. Each declared deliverable is content-bound
for reuse. Current
continuation can relocate one manifest only; changing any location of a
multi-output stage requires fresh execution, with the usual smoke/approval gates.

Declare `per_item` only for a JSONL serializer whose final `1:1` contract agrees
with actual row accounting. `relocatable` is also manifest-only: do not use it
for a generated audio location or a path serialized into downstream rows.
Unannotated output locations remain semantic parameters. Multiple output paths
without a primary or an established legacy preference are not guessed as a
reuse boundary. Contract gates, not the card, determine enabled disk writes and
which paths smoke may redirect. Fixed-file writers continue using the existing
`AGENT_ATOMIC_OUTPUT_PARAMS` and driver lifecycle hooks for safe publication.

File inputs named in task/segment contracts must have known source or generated
asset provenance for reuse. An additional external audio-file column is not
covered merely because the manifest and primary audio are unchanged. Such a
consumer and its suffix run fresh; earlier valid checkpoints remain eligible.
This restriction does not block fresh execution, or guess dependencies from
path-like strings. `per_row_independent` alone does not establish this coverage.

A `source.kind: audio_folder` declaration additionally requires `limit_param`.
It uses the existing sorted folder protocol: nonnegative limits select at most
that many files (zero selects none); negative limits are unbounded. Optional
`extensions_param` and `recursive_param` bind the folder selection settings,
defaulting to the ordinary folder source's extensions and recursive scan. An
optional `include_files_param` must select the exact canonical input file paths
before applying the limit. Prove these semantics with the module's fixtures.
Source delta support additionally requires an explicit `per_row_independent`
contract declaration; bounded folder selections never qualify. Unreadable or
non-path dependency values prevent trusted reuse rather than being ignored.

Sources using configuration files, archives, remote selectors, non-object rows,
other limit conventions, or different selection semantics must not claim these
protocols. They remain unsupported by this extension; card declarations do not
bypass task compatibility, wrapping, credentials, download, or approval gates.
New formats/lifecycles need separately reviewed support, not executable YAML.

## Exceptional planning facts

Most stages need neither declaration. Parameter spelling is not evidence that a
stage verifies a rate, and N:1 cardinality is not evidence of continuous audio.
These optional facts support existing checks/configuration, not a general effects
language. Preconditions are exact equality against real configured constructor
parameters (including defaults); no expressions, scripts, or prose are evaluated.

```yaml
planning:
  sample_rate_verification:       # only if this parameter verifies, never converts
    param: expected_rate
    when: {strict: true}         # optional literal scalar equality conditions
  restores_continuity:           # only a tested continuous-audio reconstruction
    when: {restore: true}        # omit when for an unconditional fact
verified:
  planning: mechanical
```

Use only the relevant fact for the stage. Back it with behavior tests, including
disabled/mismatching modes and source/group boundaries; schema validation cannot
prove signal-processing semantics. These planning-only declarations do not enter
step keys. Unknown audio rewrites invalidate old rate evidence rather than
pretending their outputs retain a previously observed rate.

## Separable decisions (deterministic strategy layer)

`decision` is narrower than `metrics`: it is an executable producer/selector
relationship, not general score documentation. Scalar decisions use
`PreserveByValueStage`. Exact compound decisions use
`PreserveByValueConditionsStage` and must enumerate every producer-declared
threshold/key dimension. Explicit one-level segment decisions use
`PreserveByValueConditionsStage` with a generic configured `items_key`;
recursive nested, metadata-private and batch/corpus decisions remain unsupported.

Any registered agent-ready producer may opt in through a mechanically verified
`decision` declaration; no central producer-name list is required. The gate
checks the configured score binding and selector, and preserves any class-declared
safe settings. A producer declaring `SEPARABLE_DECISION_DIMENSIONS` must use a
compound decision covering every declared dimension; other producers use scalar
decisions. Authors must still verify that annotation and selection reproduce the
documented behavior, rather than treating a writable numeric field as that proof.

```yaml
decision:
  kind: scalar                         # optional for legacy scalar cards
  separable_from_producer: true
  score_key_param: wer_key
  score_key_default: wer_pct
  value_type: number                  # number|string|boolean; checked before planning
  scope: task
  selector:
    stage_id: PreserveByValueStage
    key_param: input_value_key
    value_param: target_value
    operator_param: operator
    allowed_operators: [lt, le, eq, ne, ge, gt]
  missing_score_policy: selector_error
  monotonic_direction: lower_better  # optional; omit when no single direction applies
  atomic: true                       # one selector decision, never a partial multi-gate rewrite
verified:
  decision: mechanical
```

The resolved score key is the producer's actual `score_key_param` value when
configured, otherwise `score_key_default`. Conformance verifies that the param
exists, the declared default equals the constructor default, and changing that
param changes a declared producer write. A threshold value is deliberately
absent: it belongs to the downstream selector's `value_param`, so changing it
does not change producer identity.

`scope: task` scalar selector parameter names must be real
constructor parameters and must be `input_value_key`/`target_value`/`operator`
on `PreserveByValueStage`. Model filters that are safe only under exact
configuration add `producer_constraints` (for example
`{action: annotate, mode: task}`), which conformance compares with the
producer's class-declared safe settings.
`value_type` is a mechanically enforced JSON-scalar type. A `number` declaration
rejects booleans, strings, null, containers,
and non-finite floats before a candidate can be ready or configured.
`allowed_operators` may contain only `lt`, `le`, `eq`, `ne`, `ge`, `gt`.
`missing_score_policy: selector_error` records the default behavior: an absent
score key raises. Exact separation from a native filter that drops unscorable
rows uses `selector_drop` plus
`selector.missing_policy_param: missing_value_policy` and
`selector.required_missing_policy: drop`. `GetAudioDurationStage`'s existing
`-1.0` file-read sentinel is still a present value and is compared normally.

Compound decisions are atomic: the selector conditions are a complete AND
surface with explicit `condition_logic='and'`, never one arbitrarily chosen
"primary" threshold. Generic `PreserveByValueConditionsStage` pipelines may
use OR, but OR is not exact native-filter separation. Each dimension binds
a producer threshold parameter to its configurable score-key parameter/default:

```yaml
decision:
  kind: compound
  separable_from_producer: true
  value_type: number
  scope: task
  producer_constraints: {action: annotate, mode: task}
  dimensions:
    - {threshold_param: noise_threshold, score_key_param: noise_key, score_key_default: sigmos_noise}
    - {threshold_param: ovrl_threshold, score_key_param: ovrl_key, score_key_default: sigmos_ovrl}
    # ...all remaining producer-declared dimensions are required...
  selector:
    stage_id: PreserveByValueConditionsStage
    conditions_param: conditions
    condition_logic_param: condition_logic
    required_condition_logic: and
    missing_policy_param: missing_value_policy
    required_missing_policy: drop
    required_operator: ge
  missing_score_policy: selector_drop
  atomic: true
```

Conformance requires the dimension list to exactly match the producer's
`SEPARABLE_DECISION_DIMENSIONS`; planning then requires conditions for every
enabled (non-null) threshold using the configured score keys. Initial checkpoint
planning is supported. Scalar `decision_value` remains scalar-only and fails
closed for compound decisions. Compound feedback uses the separate
`decision_conditions` surface: a complete non-empty list/mapping of configured
declared score keys, finite targets, and exact `ge` operators. It replaces only
the selector condition set, may omit dimensions to disable them, and may enable
a dimension only when the annotation contract—and an existing checkpoint when
present—proves that score key is available. AND logic, missing-score drop, and
nested item/empty-parent policies are fixed by the card and cannot be changed
through feedback.

An optional `decision.variants` list contains full decision declarations. UTMOS
and SIGMOS use one `scope: segments` variant whose
`producer_constraints` require explicit `mode: segments`; `mode: auto` is
refused because the runtime data chooses its scope. Segment selectors must bind:

```yaml
selector:
  stage_id: PreserveByValueConditionsStage
  conditions_param: conditions
  condition_logic_param: condition_logic
  required_condition_logic: and
  missing_policy_param: missing_value_policy
  required_missing_policy: drop
  required_operator: ge
  items_key_param: items_key
  items_key_source_param: segments_key
  empty_policy_param: drop_parent_if_empty
  required_empty_policy: true
```

The planner resolves `items_key` from the configured producer
`segments_key`, requires exact equality and `condition_logic='and'`, and treats
every enabled SIGMOS threshold as one AND condition. UTMOS uses one condition
over its configured score key and still requires explicit AND in its exact
segment contract. The selector filters only direct children of that list; it
does not recurse. Missing list containers, non-list values, and non-mapping
children are structural errors rather than missing-score policy cases.

## Semantic output facts (LLM reasoning layer)

Constructor params and semantic roles make a stage mechanically connectable;
they do not tell the host what an output means at a particular pipeline
position. Document that meaning for outputs that a downstream stage can filter,
compare, aggregate or otherwise interpret. This is especially important when a
stage changes cardinality (`1:N`, nested output, `N:1`) or preserves a value
whose entity is different from the emitted row.

`semantic_facts` is an optional organization aid, not a fixed semantic ontology.
Each value may be one compact prose string or the richer mapping below. The gate
checks only that the YAML is readable prose; `scope` is not an enum and the
validator does not turn it into a pipeline pass/fail rule.

```yaml
semantic_facts:
  some_output_key:
    meaning: "What the value represents."
    unit: "seconds (float)"
    provenance: "Computed from the original input clip before fan-out."
    scope: "original input clip; parent-level aggregate"
    propagation: "Copied unchanged onto every emitted child row."
    counterexamples:
      - "Filtering this on a child does not measure a property recomputed for that child."
```

For every externally interpreted output, cover:

1. **Meaning** — describe the concept, not just its storage key.
2. **Unit/range/vocabulary** — include boundaries and direction when relevant.
3. **Provenance** — name the input entity and computation/model that produced it.
4. **Scope/granularity** — file, original parent, child, segment, speaker, batch
   or corpus, expressed as precise prose.
5. **Propagation** — explain copying, recomputation, nesting, fan-out and
   aggregation behavior.
6. **Counterexamples** — give at least one tempting wrong interpretation and
   what a downstream filter/consumer would actually do.

Use `notes`/`caveats` instead when a fact spans multiple fields. Do not fabricate
missing semantics and do not grow a per-module Python rule table to compensate:
leave a TODO and make the host ask. Mark code-grounded facts
`verified.semantic_facts: mechanical`; use `measured` or `best_guess` honestly
when that is how the claim was established.

## `resource.gpu_mem_gb` is a reference, not a per-GPU constant

`gpu_mem_gb` is a reference estimate for a workload, not a guarantee on every GPU.
Memory use depends on model, precision, batch size, input length, framework and hardware;
changing precision does not promise a fixed reduction. Put measurement conditions in
agent-visible `notes` or `resource.throughput_hint` (for example: hardware, precision,
batch size, clip length, and allocated/reserved-memory measurement). Comments alone lose
that context when YAML is loaded.

The planner compares resource estimates with the execution-target facts it can establish.
Current calibration may **raise** a card/default estimate when the measured value is
higher; it does not lower that baseline. A bounded smoke measurement does not prove that
the full configured workload fits.

## Gate (what is enforced)

`python -m nemo_curator.stages.audio.audio_agent.card_conformance` fails if any card:
- has a `stage_id` that doesn't resolve, or a missing required field;
- lists a `params_of_note` / `presets` key that isn't a real constructor param;
- uses an unknown `resource` key / non-numeric numeric / bad `bound`;
- declares a `versions` block that isn't a `{model_id: string}` map, sets it without a `model_id`, or has a model-selecting `preset` (sets `model_name`/`model_path`) whose model id isn't documented in `versions`;
- declares a `metrics` block with a bad `scale.direction`, a `threshold_param` that isn't a real param, or a `valid_range` that isn't `[lo, hi]`;
- declares a `decision` for an unregistered producer, uses unknown/missing
  fields or unsupported values, drifts from producer/selector constructor
  params/defaults/writes, omits exact producer constraints/compound dimensions,
  or omits `verified.decision: mechanical`;
- declares `semantic_facts` with a non-mapping top level, non-prose values, or malformed `counterexamples`;
- declares `semantic_facts` without a corresponding
  `verified.semantic_facts` evidence tier, or uses a `verified` tier outside
  `{mechanical, measured, best_guess}`.

`semantic_facts`, `notes`, `use_cases` and other intent-facing prose are
advisory reasoning material, not deterministic validation rules. Review them
against code/model evidence when authoring the card; green conformance means the
mechanical surface is honest, not that every intended use is semantically sound.

Coverage gaps (stages with no card) are reported by `audit()` and fail the default
CLI gate. Use `--allow-uncarded` only when deliberately checking a work-in-progress
catalog; discovery alone does not replace the release requirement for a truthful card.

Card coverage is also required by the normal audio test suite. The requirement follows
agent discovery: a registered stage with `AgentReady` and its own `describe()` must
ship a valid card. Stages outside that discovery catalog do not need cards. No fixed
stage inventory or module-specific exemption is used for this coverage check.
