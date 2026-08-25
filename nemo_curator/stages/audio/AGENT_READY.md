# Making an Audio Stage Agent-Ready

This is the **short** checklist for stage owners. "Agent-ready" means an LLM agent can
**discover** your stage, **configure** it, and **chain** it with others to build a pipeline —
without reading your source. The design goal is **minimal burden**: you declare a small core,
the framework auto-derives the rest, and one test tells you if anything is missing.

> **Golden rule:** every new knob defaults to today's behavior. Agent-readiness must not change
> how your stage runs in existing pipelines.

---

## TL;DR — the mechanical contract is 3 things

1. Inherit `AgentReady` and implement **`describe()`** returning a `StageContract` with
   **`reads`, `writes`, `cardinality`** (+ honest **`gates`**).
2. Make every `task.data` key you read/write a **`*_key` constructor field** (no bare key
   literals in `process()`).
3. Add one test: **`assert_agent_ready(MyStage(...), fixture_factory=...)`**.

Those three items make the stage mechanically composable. Also update its capability card
with the meaning of externally consumed outputs (especially filterable fields and anything
crossing a fan-out/aggregation boundary). `assert_agent_ready` can prove keys and
cardinality; it cannot prove that a reasoner will interpret a value correctly.

---

## What you MUST declare (only you know these)

```python
from dataclasses import dataclass
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    Gates,
    IOSpec,
    StageContract,
)
from nemo_curator.stages.base import ProcessingStage

@dataclass
class MyStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """One-line summary becomes the agent-facing description.

    Args:
        audio_filepath_key: Path to the input audio.   # docstring Args -> param descriptions
        score_key: Where the score is written.
    """
    audio_filepath_key: str = "audio_filepath"
    score_key: str = "my_score"
    name: str = "MyStage"

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            writes=IOSpec(data_keys=[self.score_key]),
            cardinality="1:1",                       # see "Cardinality" below
            gates=Gates(requires_gpu=self.resources.requires_gpu),
        )
```

- **`reads` / `writes`** — the `task.data` keys consumed/produced. Use `segment_data_keys` for keys
  written *inside* `segments[]` items. Use `reads_one_of=[IOSpec(...), ...]` if input can take more
  than one shape (e.g. waveform **or** file).
- **`cardinality`** — one of `"1:1"`, `"1:1 nested-list"`, `"1:N fan-out"`, `"N:1"`, `"filter"`.
  (`"filter"` = `process` may return `None`/`[]` to drop items.)
- **`gates`** — be honest about side effects: `requires_gpu`, `writes_to_disk`,
  `requires_internet_first_run`, `requires_ffmpeg`. Serializability (both exist on `Gates`): a sink
  that `json.dumps` `task.data` as-is must set `requires_serializable_input=True`; a converter that
  strips tensors/audio blobs sets `sanitizes_output=True`. A stage that derives durable filenames,
  member names, or row identity from framework `task.task_id` sets
  `requires_stable_task_id=True`; metadata-manifest resume boundaries cannot restore that identity
  and will reject the suffix.

### Conditional/data-dependent writes

Keep `writes` as the existing mechanical output declaration. When a listed
field is only written on a runtime branch—or a non-preserving stage may copy an
upstream field only when it is present—add factual possibility metadata:

```python
return StageContract(
    writes=IOSpec(data_keys=[self.score_key], segment_data_keys=[self.score_key]),
    conditional_writes=[
        ConditionalWrite(
            writes=IOSpec(data_keys=[self.score_key]),
            condition=f"'{self.segments_key}' is absent, so the task-level branch runs",
        ),
        ConditionalWrite(
            writes=IOSpec(segment_data_keys=[self.score_key]),
            condition=f"'{self.segments_key}' is present, so the per-segment branch runs",
        ),
        ConditionalWrite(
            writes=IOSpec(data_keys=list(self.passthrough_keys)),
            condition="the same input key is present, non-null, and allowed by the output whitelist",
            value_origin="upstream_same_key",
        ),
        ConditionalWrite(
            metadata_writes=[self.metadata_key],
            condition="the metadata value is computed and attached to the emitted task",
        ),
    ],
)
```

`ConditionalWrite` does not execute a predicate and does not change processing,
defaults, or the validator's legacy mechanical interpretation of `writes`. It
labels output possibility and value provenance for `validate.semantic_review`.
The `upstream_same_key` origin is important for allowlist/rebuild stages: it
keeps the original producer's meaning visible instead of falsely presenting the
copier as a new metric producer.

Use `metadata_writes` for a conditional `task._metadata` output. Unconditional
metadata inputs/outputs remain declared through
`StageContract.metadata_reads`/`metadata_writes`; semantic review traces all
three scopes (`task`, `segment`, and `metadata`) independently.

Use `augments_upstream_same_key` when the stage adds entries to an existing
mapping (for example a shared metrics mapping), and
`transforms_upstream_same_key` when it replaces the same key with a derived
value. These are objective lineage facts, not claims about whether the value is
appropriate for the user's goal.

Conditions must describe actual code branches using configured key values. They
are not an intent checker, a module-specific validator, or a centralized
field-scope ontology. The host LLM still decides whether a conditional field's
meaning and granularity fit the request.

### `gates.per_row_independent` — usually nothing to do

This one decides whether a *delta run* (reprocessing only the files that changed, instead of the
whole corpus) may include your stage. **Most stages declare nothing and are handled
automatically.**

`delta.region()` refuses to assume anything about a stage that could see a row other than the one
it was handed. It checks three things, and if **none** of them is true your stage is treated as
independent with no declaration from you:

- you override `process_batch` (you are handed several rows at once)
- `gates.writes_to_disk=True`
- `gates.lifecycle_side_effects=True`

**If one of those IS true**, the delta refuses your stage by name until you answer this question:

> If I ran this stage over file `X` alone, versus over `X` plus 999 others, would `X`'s output row
> be identical?

- **Yes** → `per_row_independent=True`, with a comment saying *why* it survives batching.
- **No** → `per_row_independent=False`. Nothing is lost: the delta simply stops at your stage, and
  every stage above it still reprocesses only the changed files.

The reference pair — both batch, opposite answers:

| Stage | | Why |
|---|---|---|
| `ASRStage` with `NeMoASRAdapter` | `True` | prepares each waveform independently and passes the batch to NeMo with lengths preserved |
| `TorchSquimQualityMetricsStage` | `False` | pads to the batch max with **no lengths**, so padding reads as silence and scores move |

Typical reasons for `False`: a corpus statistic or percentile threshold, a running counter that
picks output names, appending to a file shared across rows, an unseeded RNG advanced per row,
batch padding without lengths.

Three more rules:

- **Declare per instance when the unsafety is conditional.** `SplitLongAudioStage` is independent
  only while no shared `output_dir` flattens every source's splits into one namespace, so it
  declares `per_row_independent=(self.output_dir is None)` rather than a flat `False` that would
  cost the default configuration — the safe one — its delta. `SplitASRAlignJoinStage` and
  `InferenceSortformerStage` do the same with their own output directories.
- **A source accepting `include_files` MUST declare**, `True` or `False` — silence is a conformance
  error. That parameter is how a delta narrows a source, so it has to be answerable.
- **Getting it wrong is asymmetric.** `False` when you were safe costs a full rerun — annoying and
  harmless. `True` when you were not silently produces rows a full run would never have produced,
  and republishes them as the corpus's reusable result. **When unsure, declare `False`.**

The two `CreateInitialManifest*` sources with `max_samples` are a **deliberate exception** to that
last rule, not an example of it. `max_samples` truncates the *sorted* listing, so a delta over a
bounded source can select files a full run would not have — yet both declare a flat `True`, because
the conditional `False` denied reuse to the configuration nearly everyone runs (ReadSpeech defaults
to 5000). The limitation is recorded at each declaration. Do not copy this into a new stage; if you
find yourself wanting to, declare `False` and raise it instead.

## What is AUTO-DERIVED — do NOT hand-write these

| Field | Derived from |
|---|---|
| `params` (names, types, defaults, `choices` from `Literal[...]`) | your dataclass fields / `__init__` |
| param `description`s | your class docstring `Args:` section |
| key `role`s | your `*_key` field names (shared `_roles.KEY_ROLES`) |
| `dispatch` | whether you override `process_batch` |
| `description`, `stage_id` | class docstring / class name |

You never put `params` in `describe()`.

## What is OPTIONAL — set only if it's obvious

Declared via one class attribute, `AGENT_STATIC = StaticHints(...)`, or on the contract:

- **StaticHints-settable** (instance-free): `cardinality_options` (e.g. `["fan_out", "nested"]`),
  `gates`, `dispatch`, `error_policy` (`"skip" | "fail" | "annotate"` — default `"unknown"`; set
  only if your stage has a clear, uniform policy), `description`, `stage_id`.
- **Contract-only** (return them from `describe()`; StaticHints has no such fields):
  `iteration_key`, `size_envelope`.
- `BATCH_ONLY = True` — only if your `process()` raises and just `process_batch` works.

If you're unsure, leave them. The agent treats missing optionals safely.

---

## Naming rule: config-knobs-only

Keep **today's default key names**. Do **not** rename keys to a global vocabulary. Just expose a
`*_key` field for each so an agent can remap when wiring two stages. Compatibility comes from
**semantic roles** (below), not from everyone using the same strings.

## Semantic roles — the compatibility contract

An agent chains a producer's output to a consumer's input by **role**, not key string. Roles are
resolved automatically from your `*_key` **field name** via `nemo_curator/stages/audio/_agent/_roles.py`.

- If your key fields use existing names (`audio_filepath_key`, `waveform_key`, `score_key`,
  `text_key`, `segments_key`, …) you get the right role for free.
- If you add a **brand-new** `*_key` concept, add one line to `KEY_ROLES` in `_roles.py` (or, for a
  truly stage-internal key, list it in `INTERNAL_KEY_FIELDS`). The conformance test fails if you
  forget — it won't let a key silently fall through.

## Output meaning — the reasoning contract

Roles answer “can these stages connect?” They do not answer “what does this
value mean here?” A pipeline can connect perfectly and still apply a valid
filter to the wrong entity. Put that semantic knowledge in the capability card,
where the host LLM can reason over it; do not add a per-field rule or a hard
`field_scope` ontology to the Python core.

For each output that another stage may select, aggregate, compare or filter,
document:

- **meaning** — what the value actually represents, not merely “the key where it
  is written”;
- **unit/range** — seconds, Hz, speakers, MOS range, category vocabulary, etc.;
- **provenance** — which input/entity and computation produced it;
- **scope/granularity** — file, original parent item, emitted child, segment,
  speaker, batch or corpus, written as factual prose rather than an enum;
- **propagation** — whether fan-out copies a parent value to every child, nesting
  moves it into segments, aggregation summarizes many rows, or a transform
  recomputes it;
- **counterexample** — at least one plausible but wrong interpretation and its
  pipeline consequence.

Use the card's optional `semantic_facts` mapping for structured prose, or
`notes`/`caveats` when the fact spans several outputs. For example, if a
speaker-separation stage computes the original clip's detected-speaker count
once and copies it to every per-speaker child, say so explicitly: filtering that
child field to `== 1` selects children whose **parent source** had one detected
speaker; it does not test whether each already-separated child track is
single-speaker.

Only document facts grounded in code, a measured run or an authoritative model
source, and mark their honesty tier in the card's `verified` block. Missing
meaning should remain an explicit TODO; the host must ask rather than invent it.
See `nemo_curator/audio_agent/knowledge/CARD_SCHEMA.md`.

## Discovery — how the agent finds your stage

Nothing to do: your stage auto-registers (via `StageMeta`) and appears in the catalog. Consumers
go through the public entry point — `agent.py` is the sanctioned public surface; don't import the
private `_catalog` module directly:
```python
from nemo_curator.stages.audio import agent

agent.list_agent_ready_stages()  # -> [... "MyStage" ...]
agent.describe_stage("MyStage")  # -> StageContract (static, instance-free)
agent.catalog_as_json()          # -> JSON the agent/UI consumes
```

---

## The safety net: `assert_agent_ready`

Add one test. It runs the static checks (contract shape, valid roles, JSON-serializable, reads
satisfiable by role) and — with a fixture — runs your stage and verifies declared writes appear,
no undeclared top-level keys leak, and cardinality matches runtime:

```python
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready

def test_my_stage_is_agent_ready(tmp_path):
    def fixture():
        return AudioTask(data={"audio_filepath": str(_write_wav(tmp_path / "a.wav"))})
    assert_agent_ready(MyStage(), fixture, expected_cardinality="1:1", available_keys={"audio_filepath"})
```

For GPU/model stages, reuse the existing fake-model/stub setup (see
`tests/stages/audio/test_agent_simulation_pipelines.py`) so the test needs no GPU. You don't need to
memorize the rules — if the test passes, the contract is honest.

---

## Checklist (copy into your PR)

- [ ] `AgentReady` + `describe()` with `reads`, `writes`, `cardinality`, honest `gates`
- [ ] every read/written `task.data` key is a `*_key` constructor field (no bare literals)
- [ ] new `*_key` concepts have a `_roles.KEY_ROLES` entry (or `INTERNAL_KEY_FIELDS`)
- [ ] capability card explains each externally consumed output's meaning, unit,
      provenance, scope/granularity, propagation and a counterexample
- [ ] new `AudioTask`s preserve `_metadata` and `list(_stage_perf)` (manual — not covered by `assert_agent_ready`)
- [ ] if you override `process_batch`, write to disk, or set `lifecycle_side_effects` — decided
      `gates.per_row_independent` (`True`/`False`, per instance if conditional); otherwise left it
      alone and let the delta derive it
- [ ] `assert_agent_ready(...)` test added and green
- [ ] defaults unchanged → existing pipelines behave exactly as before

Auto-derivation handles params/roles/dispatch/description; the card supplies meaning only the
stage author knows. Neither documentation step changes runtime defaults.
