---
name: checkpoint-placement
description: Select a core-proven metadata checkpoint for same-dataset parameter feedback. Use after recipe validation and semantic critique, before authoritative smoke, when the user expects to tune a declared downstream decision after seeing results.
---

# Checkpoint placement

Use this skill only for an unchanged dataset and an iterative scalar or atomic compound
decision that a capability card declares separable from its annotation producer.

Replacing a stage, metric, acceptance criterion, or checkpoint topology is a
mid-workflow recipe branch, not threshold feedback, delta work, or continuation.
Return to the audio-curation skill's full branch-reset sequence. A scalar or
compound condition change also invalidates prior recipe validation, critique,
smoke, and approval, even when this skill can reuse its producer.

## Procedure

1. Start from the exact validated recipe. Do not edit stage modes, score keys, selectors,
   residency, thresholds, or paths yourself.
2. Call `plan-checkpoint` with `--data` and without `--output-path` to inspect core-proven
   options. The core derives the checkpoint location itself; a path is never yours to invent.
3. Offer at most one non-dominated option, and only when:
   - the user explicitly or strongly implies they will inspect output and tune a decision;
   - the result has `status: needs_decision`, `needs_output_path`, or `ready`;
   - meaningful expensive/model work is listed above the decision; and
   - no existing checkpoint already covers it.
4. Show the candidate's concise benefit, cost, checkpoint effect, and baseline trade-off
   through the host's structured AskQuestion UI. The earlier soft curation-mode choice is
   not this recipe-specific accept/decline decision. Never call `plan-checkpoint --choice
   baseline`, `--choice checkpoint`, `--output-path`, or an equivalent MCP choice on the
   user's behalf; wait for and use only the selected response.
5. If the user declines, only then call `plan-checkpoint --choice baseline` and use only
   its returned baseline recipe. The core binds that explicit choice to the exact recipe
   and option set. Do not offer again in this workflow.
6. If they accept, call `plan-checkpoint --choice checkpoint` and use only the returned
   complete recipe. Do NOT ask the user for a path: a checkpoint is recomputable cache,
   addressed by the step key it belongs to, and asking makes them name a file they have no
   reason to care about. Pass `--output-path` only when the user has asked, unprompted, to
   keep the metadata somewhere of their own. On `status: needs_output_path` the core could
   not derive a location -- confirm `--data` was supplied and names a readable source
   before falling back to asking.
7. Re-run validation and the full host semantic critique. Any transformed recipe has a new
   `config_hash`.
8. Run `reuse-scan`, then authoritative smoke on that exact final recipe. Smoke
   isolates its writes in temporary storage; never create or pre-clean the real
   checkpoint path.
9. At the approval gate disclose:
   - checkpoint path and overwrite/collision behavior;
   - rows and bytes measured by smoke, projected full size and confidence;
   - measured prefix work the checkpoint can save;
   - cumulative trust/TTL and any non-deterministic prefix;
   - retention seconds and who owns deletion.
   Immediately before asking, inspect deterministic `output_targets`, the checkpoint
   writer's resolved output-path contract, the continuation/reuse card when applicable,
   and safe read-only current path facts. If the checkpoint or another execution target
   is occupied and proven to be replaced/overwritten, name its exact path and remind the
   user to copy or save work they need first. Never copy, delete, rename, clean, truncate,
   or pre-create it. Do not warn for new targets, infer overwrite from an append contract
   (or append from a replace contract), or guess when behavior is unproven—say so and ask.
10. Present those facts and ask for post-smoke execution approval, then end the response.
    Run only after a subsequent user answer, exact-hash approval, and a smoke token
    for this candidate, plus the reminder when applicable. A hash/token proves artifact
    identity, not AskQuestion provenance or consent.
11. After a successful run, include the checkpoint under **Intermediate/checkpoint
    files** in the final **Saved files** block, separate from terminal deliverables and
    existing reused/served artifacts. Claim it was saved only from returned
    `output_paths`/`output_targets`, published lineage/artifacts, the report, deterministic
    stage-path discovery plus a safe read-only existence check, or run-record recipe
    metadata. Never list the smoke-isolated checkpoint or a planned-but-unexecuted path as
    durable; if the core did not report a requested path, say so rather than guessing.

Authoritative smoke and full run refuse a recipe with a recommended expensive checkpoint
until it contains either the checkpoint or the exact baseline-decline attestation returned
by `plan-checkpoint`. Never bypass that refusal by rebuilding or hand-editing the recipe.

## Pipeline policy

- Keep paths plus offsets when audio can be recreated from source.
- Carry a waveform only across adjacent consumers that need it.
- Never write intermediate audio merely to make a checkpoint.
- Persist complete serializable row metadata before the first declared destructive decision.
- Explicit UTMOS/SIGMOS segment decisions are eligible only when the card and core bind the
  generic selector `items_key` to the producer's exact `segments_key`, preserve every enabled
  threshold with explicit `condition_logic='and'`, drop missing scores and empty parents, and
  carry no waveform. Generic OR is valid for non-reuse pipelines but is not exact native-filter
  separation. Path plus offset metadata is serializable and may be checkpointed.
- Keep native filters when scope is `auto`, nesting is recursive, a segment list carries a
  waveform, or decisions are private-metadata-dependent, batch/corpus-dependent, or not
  card-declared as separable.
- Never place more than one new checkpoint.

## Feedback after a completed run

Call `plan-checkpoint --from-run RUN_ID --data DATA --decision-stage PRODUCER
--decision-value VALUE`. The core adopts the exact recipe and changes only the declared
scalar selector. For a card-declared compound decision, use
`--decision-conditions '<JSON>'` instead. It is a complete replacement, not a patch:
name each enabled configured score key once, use finite numeric targets and `ge`, and
omit a dimension to disable it. The core keeps AND logic, missing-score drop behavior,
and nested `items_key`/empty-parent policy fixed, and refuses enabling a score not proven
present in the retained annotation checkpoint.

That returned selector change is a new recipe hash: retain still-valid same-chat
context/profile/preference, but follow the branch reset from validate -> host critique ->
checkpoint decision (if any) -> reuse scan -> authoritative smoke -> presentation ->
subsequent approval before executing. If the dataset key changed, stop this flow and follow
the returned `delta_run` route. For unchanged data, use `reuse-scan` and `continue` only
after those recipe-level gates; do not execute a hand-edited suffix. Validation may inspect
an occupied checkpoint only when its exact dataset/prefix artifact and content digest prove
it complete; direct full execution still refuses to recreate or overwrite that path.

Tightening may be possible from a retained final subset, but loosening needs rows a prior
filter discarded. Do not claim final-manifest reuse unless the core proves it; the dedicated
pre-gate checkpoint is the supported evidence.
