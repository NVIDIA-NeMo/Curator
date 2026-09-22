---
name: audio-stage-authoring
description: Make a new or existing NeMo Curator audio stage agent-ready so the audio agent can discover, configure and chain it. Use when adding an audio stage, writing or fixing its describe() contract or StageContract, adding *_key fields or semantic roles, writing a capability card, or when card conformance or assert_agent_ready fails. This is for authoring stage code in a Curator checkout, not for curating a dataset - use the audio-curation skill for that.
---

# Making an audio stage agent-ready

This skill is the **procedure**. The authoritative checklist it drives is
`nemo_curator/stages/audio/AGENT_READY.md` — read that file for the mechanical
contract rather than working from memory, and do not restate it here.

**Golden rule, from that document:** every new knob defaults to today's behavior.
Agent-readiness must never change how a stage runs in an existing pipeline. If a
change alters existing output, it is a behavior change and needs its own justification.

## Before writing code, read these

- `nemo_curator/stages/audio/AGENT_READY.md` — the contract: `AgentReady` + `describe()`
  returning `reads`, `writes`, `cardinality`, honest `gates`; `*_key` constructor fields;
  `ConditionalWrite`; what is auto-derived and must not be hand-written.
- `nemo_curator/audio_agent/knowledge/CARD_SCHEMA.md` — the capability card schema.
- `.cursor/rules/processing-stage-patterns.mdc` and `.cursor/rules/composite-stage-patterns.mdc` —
  the repo's own conventions for `ProcessingStage` and `CompositeStage`. These are
  upstream-maintained framework rules; read them, do not copy from them, and never edit them.
  (Checkout-only: they are not part of the installed package.)

## The order that works

1. **Write the stage the normal way first**, following the two pattern rules above.
   Agent-readiness is a declaration layer over working code, not a substitute for it.
2. **Turn every `task.data` key into a `*_key` constructor field.** A bare key literal
   inside `process()` is invisible to the agent and cannot be remapped when wiring two
   stages together. This is the single most common omission.
3. **Implement `describe()`** with `reads`, `writes`, `cardinality`, and honest `gates`.
   Do not put `params` in it — those are auto-derived from the dataclass fields.
4. **Declare a new key concept in `_roles.KEY_ROLES`** if the stage introduces one, or in
   `INTERNAL_KEY_FIELDS` if it is genuinely stage-internal. `INTERNAL_KEY_FIELDS` is unioned
   across the MRO, so a subclass declaring its own set does not disown its parent's.
5. **Write the capability card** with the meaning of every externally consumed output:
   meaning, unit or range, provenance, scope or granularity, propagation across
   fan-out/nesting/aggregation, and at least one counterexample. Roles prove that two
   stages *can* connect; only the card lets the host judge whether connecting them
   achieves the user's intent.
6. **Add the conformance test**: `assert_agent_ready(MyStage(...), fixture_factory=...)`.
   For GPU or model stages, reuse the fake-model harness in
   `tests/stages/audio/test_agent_simulation_pipelines.py` so no GPU is required.
7. **Run the card gate**: `card_conformance.audit()` must report zero violations, and an
   unknown top-level card key is a violation rather than something silently ignored.

## Declare honestly, especially where it costs you

The contract is a promise the agent plans against, so an optimistic declaration is worse
than a missing one. Three traps worth naming:

- **A stage that may drop rows is `cardinality="filter"`**, even when dropping is a side
  effect of something else. A strict sample-rate check that returns `[]` for a mismatched
  row is filtering, and declaring `1:1` makes validation treat a silently discarded corpus
  as a pass-through.
- **`gates` are environment facts, not aspirations.** `writes_to_disk=True` with an empty
  `output_path_params` is refused by smoke, because it claims disk writes while offering
  nothing to redirect into the sandbox.
- **Durable task-ID dependencies are resume facts.** If framework `task.task_id` enters an
  output filename, archive member, or durable row identifier, declare
  `gates.requires_stable_task_id=True`; metadata checkpoints do not serialize that identity.
- **`ConditionalWrite` labels possibility, not intent.** Its `condition` must describe a
  real code branch using configured key values. Use `value_origin="upstream_same_key"` when
  the stage merely copies a field, so the original producer keeps credit for the meaning and
  the copier is not presented as a new metric producer.

## Verify

```bash
.venv/bin/python -m pytest tests/stages/audio -m "not gpu" -q
.venv/bin/python -m nemo_curator.audio_agent describe MyStage --params '{"score_key": "my_score"}'
```

`describe` is the agent's view of the stage. If its `reads`/`writes` do not match what the
code actually touches for those params, the contract is wrong no matter what the tests say.

Then confirm the stage is reachable through the public surface, not the private catalog:

```python
from nemo_curator.stages.audio import agent
agent.list_agent_ready_stages()   # your stage should appear
```

## Checklist

Copy the PR checklist at the end of `nemo_curator/stages/audio/AGENT_READY.md` rather than
a paraphrase of it — that list is maintained with the framework and this file is not.
