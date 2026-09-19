# Smoke, the confirm gate, the run, and acceptance verification

Loaded on demand from `SKILL.md` steps 5 to 7. Read this once the semantic critique passes.

## Branch-reset precondition

Smoke evidence is recipe-specific. If a stage was added, removed, replaced, or
reordered; a semantic stage parameter or acceptance criterion changed; or
checkpoint topology changed, discard the prior validation, host critique,
checkpoint choice, reuse scan, smoke, and approval. Follow the complete
mid-workflow recipe branch reset in `SKILL.md`. A still-valid same-chat dataset
profile and soft curation preference may be retained, but a metric/stage/criteria
replacement is not threshold feedback, delta work, or continuation.

Do not treat `validate.semantic_review`, `review_required`, or its
`required_response` schema as a completed critique. Before this file's smoke
step, the host must have emitted `mechanically_runnable: true`, the exact final
`recipe_config_hash`, and `intent_status: pass`.

## 5. Smoke (empirical loop, <= 2 iterations)

```bash
python -m nemo_curator.audio_agent smoke --recipe recipe.yaml --sample 10 --data /path/to/data --bootstrap-ray
```

`--bootstrap-ray` lets the agent start a correctly-configured local Ray head
itself (free port, plasma on /tmp, API limit) so no manual Ray setup is needed.
If a cluster already exists, set `RAY_ADDRESS` and omit the flag.

Show the user `retained` / `rejected` + examples. If `goals_met` is false (0
retained, errors), read the structured `diagnosis` (when present). Adjust a
threshold only for a grounded data/filter failure; environment/action-required
failures go through the decision policy in `SKILL.md` step 2.

On a GPU smoke the result also carries a `calibration` block (measured per-stage
VRAM/throughput). Pass it to the full run so the resource planner can raise a
card/default estimate when the smoke observed a larger peak; because a bounded
smoke cannot prove the full-run maximum, calibration never lowers that baseline:
`run ... --calibration calib.json` (extract it with `calibrate --smoke
smoke.json`). On a CPU smoke there's no VRAM to measure, so the planner keeps
using the card facts.

## 6. Confirm gate -> run

Immediately before the final approval question, refresh the path-impact facts for the
execution being proposed. Inspect:

- deterministic `output_targets` and each target's current existence/row/file facts;
- the configured stages' resolved output-path contracts, including whether each sink
  replaces, appends, creates, or only reads/serves;
- for delta or continuation, the returned delta/reuse/continuation card and its exact
  manifests, stale outputs, checkpoints, and remaining-stage targets; and
- safe read-only existence checks when the structured facts need refreshing.

If these facts prove that execution will replace or overwrite an occupied path, briefly
name each exact path and say: copy or save any work there that must be kept before
proceeding. Never copy, delete, rename, clean, truncate, or pre-create a target yourself.
Do not infer replacement from a stage name or an apparent append-mode implementation, and
do not claim append where the contract says replace (or vice versa). If mutation behavior
is not grounded, say it is unproven and ask before execution. Omit the reminder when every
target is new, or when successful reuse/serve-as-is returns an existing artifact without
mutating it.

Present the plan, the semantic critique (`intent_status: pass`), the smoke
evidence, the scale/time estimate, **and the acceptance-criteria contract** —
stating for each criterion **what its metric captures and what it does NOT**
(e.g. "UTMOS measures naturalness/overall quality, not background-noise level —
add a noise/SIGMOS criterion?"). Include the occupied-path reminder above when
applicable. Never silently decide which metric stands for a fuzzy word ("clean",
"good"); surface it here. Then ask the user to confirm. This is still one gate:
the path reminder supplements rather than replaces explicit approval. **End the
response after asking. Never call `run` in the response that presents the smoke
result.** Only a subsequent user answer can supply the post-smoke execution
decision. Only then:

```bash
python -m nemo_curator.audio_agent run --recipe recipe.yaml --confirm <config_hash> --data /path/to/data --bootstrap-ray
```

Passing the `config_hash` (from the refusal output) enforces plan-execution
integrity: the candidate being run is exactly the one smoked. It is not evidence
that the user approved it. A smoke token likewise proves a core event, not
AskQuestion provenance or consent. `--bootstrap-ray` starts the Ray head if
needed (same as smoke).

Guardrails enforced in the tool: paths are restricted to `AUDIO_AGENT_WORKSPACE`
(when set); secrets/transcripts are stripped from tool output; and if
`AUDIO_AGENT_REQUIRE_SMOKE` is set, also pass `--smoke-token <token>` (from the
`smoke` output) or `run` refuses. The resource planner auto-picks streaming/batch
and refuses if the recipe can't fit the machine.

## 7. Report + verify acceptance

Summarize the returned `report` (retained/rejected, per-filter counts, failure
reasons, output paths) in plain language. `run` also returns `acceptance`, verified
against the recipe's embedded contract and terminal-output evidence; treat that as
the primary post-run verdict. Use standalone `verify` only for an explicitly
post-hoc evidence set or a newly proposed contract:

```bash
python -m nemo_curator.audio_agent verify --criteria criteria.yaml --evidence evidence.json \
  --recipe recipe.yaml   # frozen contract -> runs the honesty guard
```

Report the `AcceptanceReport`: `overall` (`met` iff every `must` criterion is met)
plus each criterion's state — `met` / `not_met` / `unverifiable` (no evidence, e.g.
WER with no references) / `unachievable` (the data cannot reach an absolute target).
Only declare success when `overall` is `met`. `unachievable`/`not_met` are honest
outcomes: offer options (adjust thresholds, provide references, relabel an absolute
bar with the user's consent) — never silently relax a `must`.

**Reviewer charter (you, the host, are the reviewer).** After the deterministic
verify:
1. Resolve any `semantic_fit` criteria (they come back `unverifiable` — that's your
   job): judge, grounded in the evidence/examples, whether the result coheres with
   intent.
2. Read the `honesty` section — the guard flags goalpost-moving (a confirmed `must`
   dropped/downgraded/relaxed vs the frozen contract). If non-empty, `overall` is
   forced `not_met`: do **not** present success. The contract is frozen into the
   recipe and covered by `config_hash`, so relaxing a bar means re-confirming a new
   contract with the user, never editing it silently.
3. You may surface semantic concerns but **may not override the deterministic
   verdict** — if unresolved, escalate to the user.

### Always finish with the durable path inventory

After every successfully completed full run, delta run, executed continuation, or
checkpoint-producing run, and after every successful `already_done`, reused, or
serve-as-is result, end the final user response with a concise **Saved files** block.
This obligation is about path honesty and is independent of the acceptance verdict:
do not call unmet acceptance a success, but still tell the user where a completed
execution persisted its files.

Build the block only from returned `output_paths`, deterministic `output_targets`,
published artifacts/lineage, the report, run-record recipe metadata (including a
reported scratch recipe path), and deterministic configured stage-path discovery.
When safe, check a path's existence read-only before claiming it was saved. Include,
when reported:

- **Recipe used/saved:** the exact recipe path actually executed or persisted. Never
  invent a repository/current-directory recipe location merely because recipe content
  was supplied in memory. If no durable recipe path was returned, say it was not
  reported.
- **Final outputs:** final deliverable manifests, generated audio/output directories,
  and other terminal persisted artifacts. Mark whether each was newly written or
  replaced in this execution.
- **Reused/served existing outputs:** paths returned from prior artifacts without
  mutation. For reuse-as-is, explicitly say **No new output was written** when that is
  what the result proves, then list the existing served artifact.
- **Intermediate/checkpoint files:** checkpoint manifests and other persisted stage
  outputs, clearly separated from final deliverables.

Do not list in-memory handoffs, smoke-isolated paths, temporary files, or planned but
unexecuted targets as saved. For a checkpoint-planned recipe that did run, list the
checkpoint only when the result or safe read-only verification proves it exists; if
the core omitted a requested path, write “not reported” rather than reconstructing or
guessing it. Keep the block concise even when the run has many artifacts.
