# Audio stages — agent guardrails

These instructions apply to any AI coding agent working in `nemo_curator/stages/audio/`.
Codex and Cursor load this nested `AGENTS.md` automatically; Claude Code reaches it
through the sibling `CLAUDE.md` import.

Two different jobs happen in this directory, and they have opposite rules about editing.

## If you are curating a dataset, this directory is read-only

Stage source is shared library code. Never change a stage, a threshold, a filter or
windowing logic to make one user's dataset produce the output they hoped for — diagnose
from the data instead. The full guardrail, and what to do when output comes back empty,
is in `nemo_curator/audio_agent/AGENTS.md`. Drive the work through the
`nemo_curator.audio_agent` verbs rather than reading stage source to answer what a stage
reads or writes: `describe <Stage> --params '{...}'` is the sanctioned answer, and
`producers <key>` says who wrote a key.

## If you are authoring or fixing a stage, start here

Read `AGENT_READY.md` in this directory. It is the authoritative checklist and it is
maintained with the framework; work from it rather than from memory. The
`audio-stage-authoring` skill (`nemo_curator/audio_agent/skills/audio-stage-authoring/SKILL.md`)
is the procedure that drives it.

**Golden rule: every new knob defaults to today's behavior.** Agent-readiness is a
declaration layer over working code. If a change alters what an existing pipeline
produces, it is a behavior change and needs its own justification, not a checklist entry.

The mechanical contract is three things, each detailed in `AGENT_READY.md`:

1. Inherit `AgentReady` and implement `describe()` returning a `StageContract` with
   `reads`, `writes`, `cardinality` and honest `gates`.
2. Make every `task.data` key you read or write a `*_key` constructor field — no bare key
   literals in `process()`, or the key is invisible to the agent and cannot be remapped.
3. Add `assert_agent_ready(MyStage(...), fixture_factory=...)` as a test.

Then give the stage a capability card under
`nemo_curator/audio_agent/knowledge/cards/`, documenting what each externally consumed
output *means* — roles prove two stages can connect, only the card lets the host judge
whether connecting them serves the user's intent. Schema:
`nemo_curator/audio_agent/knowledge/CARD_SCHEMA.md`. `card_conformance.audit()` must come
back with zero violations.

Follow the repo's existing stage conventions while you do it:
`.cursor/rules/processing-stage-patterns.mdc` and
`.cursor/rules/composite-stage-patterns.mdc`. Those are upstream-maintained framework
rules — read them, never edit them.

Declare honestly even where it costs you: a stage that may drop rows is
`cardinality="filter"` even when dropping is incidental, and `gates` are environment facts
rather than aspirations. An optimistic contract is worse than a missing one, because the
planner treats it as ground truth.

## Verify before you claim done

```bash
.venv/bin/python -m pytest tests/stages/audio -m "not gpu" -q
.venv/bin/python -m nemo_curator.audio_agent describe MyStage --params '{...}'
```

If `describe` does not match what the code actually touches for those params, the contract
is wrong no matter what the tests say.
