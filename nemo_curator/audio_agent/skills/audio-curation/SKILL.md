---
name: audio-curation
description: Build and run a NeMo Curator audio curation pipeline from a natural-language goal (host-driven planner over the audio_agent tool core). Use for any audio, speech or audio-dataset task - quality filtering, transcription, WER filtering, VAD, diarization, ALM windowing, resampling, channel conversion, or inspecting an audio corpus. Route such work through this skill rather than writing ad-hoc ffprobe, librosa or soundfile scripts.
---

# Audio Curation Agent (P1)

Turn a user's audio-curation goal into a validated, runnable NeMo Curator recipe,
with pre-flight checks, a bounded smoke test, a confirmation gate, and an
evidence-backed report. **You (the host model) are the planner and critic.** The
`nemo_curator.audio_agent` tool core is deterministic and grounds every decision:
it tells you which stages exist, whether a recipe composes, and what happened.

All commands print JSON. Run them with the repo virtualenv interpreter (base
`python` may lack Curator's deps) from the Curator repo root:

```bash
.venv/bin/python -m nemo_curator.audio_agent <verb> [args]   # repo virtualenv, from the repo root
# or: source .venv/bin/activate  &&  python -m nemo_curator.audio_agent <verb> [args]
```

From an installed package (no checkout), the same verbs are available as
`nemo-curator-audio <verb> [args]`.

Detailed procedure lives beside this file and is loaded only when the step is reached:

- `references/routing.md` — step 3: pick stages coarse-to-fine, read contracts, resolve outcomes to params.
- `references/smoke-and-run.md` — steps 5 to 7: smoke, the confirm gate, the run, acceptance verification.
- `references/reuse.md` — step 8: reuse prior runs instead of recomputing.

## Golden rules (non-negotiable)

- **Never invent** stage or parameter names. Only use stages from `discover` /
  `catalog-tree` / `cards`, and only params the cards/contracts list.
- **0 silent full-scale runs.** Never call `run` with `--confirm` until the user
  has explicitly approved, after seeing a smoke result and the scale/cost estimate.
- **Nothing is written before approval.** Do not create, delete, move or truncate any file or
  directory ahead of the confirm gate — above all not the user's output. A gate the agent has
  already prepared the ground for is not a gate. `validate` returns `output_targets`, stating
  what already exists at every path the recipe writes to (with row and file counts): put that
  in the plan and let the user decide. Never pre-clean an output "so the rerun is clean."
  Determine append/replace behavior from the resolved stage output-path contract and current
  core facts, not a writer's apparent file-open mode, and never generalize one sink's behavior
  to another. In particular, the core-proven `ManifestWriterStage` setup replaces its manifest
  rather than accumulating rerun rows; that fact does not make an unrelated append sink a
  replacing sink. This is not hypothetical: an agent deleted a user's file before the gate on
  exactly this misreading.
- **Warn before replacing occupied paths.** Immediately before the final approval for a full
  `run`, `delta-run`, executed `continue`, or any other execution that can mutate an existing
  output, inspect deterministic `output_targets`, resolved stage output-path contracts, the
  continuation/reuse card, and safe read-only current path facts. Briefly name every exact
  occupied path that those facts prove will be replaced/overwritten and remind the user to
  copy or save anything they need before proceeding. Do not copy, delete, rename, clean, or
  pre-create it yourself. Do not warn when all targets are new or successful reuse only serves
  an existing artifact without mutation. If whether a path is mutated is not proven, say that
  and ask instead of guessing. This warning supplements the explicit confirmation,
  exact-`config_hash`, and smoke-token gates; it replaces none of them.
- **End every successful result with saved-path facts.** After a successful full run, delta,
  executed continuation, `already_done`/reuse, or serve-as-is result, include a concise
  **Saved files** block grounded only in structured results. Distinguish final newly
  written/replaced deliverables, existing reused/served paths, and intermediate/checkpoint
  artifacts. Include the exact recipe path actually used or saved when reported, output
  manifests/deliverables, checkpoint paths, generated output directories, and other persisted
  stage artifacts. Use only returned `output_paths`/`output_targets`, published
  artifacts/lineage, run-record recipe metadata or scratch recipe path, the report, or
  deterministic stage-path discovery. Check existence read-only when safe before saying a file
  was saved. Never present smoke-isolated, in-memory, or temporary paths as durable; for
  reuse-as-is, say that no new output was written when true and list the existing served
  artifact. If a requested path was not reported, say so rather than inventing it. This path
  summary is required even when acceptance is reported separately.
- **Evidence only.** Never claim quality/throughput improved without before/after
  numbers from a `report`.
- **Define success up front, verify it after.** Derive `acceptance_criteria` (the
  success contract) from the request, confirm them at the gate, and `verify` the
  result against them. Never declare success without an AcceptanceReport whose
  `overall` is `met`; never silently relax a `must` criterion.
- **Refuse / redirect**: work that needs human annotation, or model
  training/eval/deployment; large runs without confirmation. Offer a safe alternative
  (e.g. a manifest + quality report). This is a scope boundary, not a capability list:
  it does not depend on which attribute is asked for. Whether an attribute can be
  produced *automatically* is a question for the catalog — `producers <role>`, or
  `catalog-tree` then `cards --category` — never for a list in this document, which
  would go stale the day a stage ships.
- Separate **local-ready** from **GPU/endpoint/approval-needed** (see gate flags).
- **User-facing questions only.** Never ask about internal parameters (thresholds,
  `*_key`s, residency, batch size, model IDs, task-types). Ask at the **outcome layer** —
  quality *level* ("studio / broadcast / general?"), language, output format — and
  resolve to params via the card's `anchors`/`presets`. Asking a **module choice** is
  fine, framed as plain capability trade-offs from the cards. Ask only decisions that
  are **material + preference-dependent + not inferable**; otherwise use a safe
  default/inference and note it.
- **An unchecked remedy is a guess, not an option.** When a result comes back thin or failed,
  a parameter change is a *fix* only if the data you just observed can actually produce the
  outcome it promises. Check that before offering it; if you cannot, present it as unverified
  and name what would settle it. A knob whose whole range is ruled out by the input's shape —
  a grouping that must contain two of something the upstream stage emitted once, a threshold
  below every value present — is not a choice worth putting to the user, at any setting. And
  once you expect a choice to fail, say so **before** executing it, not in the caveats
  afterwards: running it spends the user's full compute to produce data you already doubt.
- **Report the unit the request was about.** A row count is not a deliverable count — one row
  can carry a list of segments, windows or snippets, and a row can be blank. `run` reports
  `output_rows_written` (read back from the file rather than counted in memory) and
  `sparse_fields`, naming each written field left blank in some rows — surface both. Never
  call an output complete or ready while a field the request depends on is empty in most rows.

## Choose one soft curation mode at workflow start

At the start of every **new executable audio-curation workflow**, before calling
`context`, routing, or constructing a recipe, use the host's fancy structured
**single-select** question UI with exactly:

- Title: **Optimize this curation**
- Question: **How should I optimize this curation? This only guides choices between equally correct pipelines.**
- **Easy to refine later** (`refine_later`): prefer reusable file-backed handoffs,
  explicit task/segment scope, exact annotate-then-selector forms, and at most one
  worthwhile metadata checkpoint. Demerit: the first run may use more metadata
  storage and I/O.
- **Fastest first run** (`fast_first`): prefer adjacent in-memory handoffs, native
  filters, early row reduction, and minimal intermediate I/O. Demerit: later
  threshold tuning may repeat model work.

Do not ask when the request already answers it: wording such as "fast" or "as
quickly as possible" means `fast_first`; "I'll tune/refine thresholds" or "reuse
later" means `refine_later`. Record the result as:

```yaml
planning_preference:
  schema_version: 1
  curation_mode: refine_later  # or fast_first
  source: explicit_user_choice  # or inferred_from_request
```

Ask once per workflow. Do **not** repeat the question during validation, semantic
critique, smoke, approval, execution, reporting, threshold feedback, or an
explicit continuation. A continuation inherits the stored preference from its
recipe/run. The same folder alone does not prove that a request is a
continuation; a new unrelated request starts a new workflow and asks again.

This mode is only a soft tie-breaker between semantically equal, legal plans. It
is never a correctness constraint or hard bound. User intent, acceptance
criteria, mechanical validation, semantic critique, safety, and available stage
contracts always win. If the preferred form is unavailable, illegal,
semantically different, or not worthwhile, choose the best correct plan and
briefly explain the deviation. The preference never authorizes a pointless
checkpoint, delayed useful filtering that causes excessive model work, or
intermediate audio written solely for reuse. The refine-later preference
considers no more than one worthwhile metadata checkpoint; existing
recipe-specific checkpoint gates remain authoritative.

The question belongs to the host UI. Do not add or emulate an AskQuestion verb
in the deterministic core. When available, pass the preference through
`context` (`--planning-mode` / `--planning-source`) and include it in the Recipe.

## Mid-workflow recipe branch reset (mandatory)

After work starts, any stage add/remove/replace/reorder, semantic stage-parameter
change, acceptance-criterion change, or checkpoint-topology change creates a new
recipe branch. A scalar threshold change also invalidates recipe evidence; a
stage, metric, topology, or criteria replacement must never be mislabeled as
threshold feedback, delta work, or continuation to skip the reset.

The same-chat goal/context, still-valid dataset profile, and soft
`planning_preference` may be inherited without asking or profiling again.
Recipe-level validation, host critique, checkpoint decisions, reuse scan, smoke,
and execution approval may not: all are invalid for the changed branch.

Restart in exactly this order:

1. Construct the exact new recipe with its embedded acceptance contract.
2. `validate` it, then emit the mandatory host `semantic_critique` response
   (`mechanically_runnable`, exact `recipe_config_hash`, and `intent_status`).
   A returned `semantic_review`/`review_required` packet is evidence for that
   critique, not proof that the host performed it.
3. Run checkpoint placement / `plan-checkpoint`. Before every offered
   accept/decline or path choice, show the concise candidate trade-off through
   the host's structured AskQuestion UI. Never call `--choice baseline`,
   `--choice checkpoint`, `--output-path`, or an equivalent MCP choice until the
   user selects it. The workflow's planning-mode answer is not this
   recipe-specific decision; retain the one-checkpoint policy.
4. If checkpoint selection transforms the recipe, `validate` and perform a new
   exact-hash semantic critique again.
5. Run `reuse-scan`, then authoritative `smoke` on the exact final hash.
6. Present smoke evidence and limitations, checkpoint effects, and any grounded
   occupied-output copy/save warning; then ask for explicit post-smoke approval.
7. Stop. Never call `run` in the response that reports smoke. Only a subsequent
   user answer can authorize `run`; a config hash or smoke token proves recipe
   integrity, not consent.
8. Run, report/verify acceptance, and finish with the existing **Saved files**
   contract.

The core cannot prove that AskQuestion produced a user answer or that the host
actually wrote the critique. Its advisory fields and exact-hash/token gates must
not be described as consent provenance; do not invent a token or hard gate that
would break existing SDK/tutorial flows.

## The loop

### 1. Interpret + build a capability plan

Turn the request into a small goal **and a capability plan**: `task` (validate /
quality-filter / VAD / transcribe / WER-filter / diarize / ALM-windows / convert),
`domain` (conversational / read / long-form / multilingual), plus `expected_outputs`
(what the user wants to end up with), `capability_areas` (which categories likely
apply), `constraints` (quality/hardware/deps), and `open_questions` (what's still
missing). The capability plan is your **coverage checklist** for later steps.
Ask 1-2 short **user-facing** questions only if something material is ambiguous
(e.g. no quality *level*). If the request hits the refuse list, stop and redirect.

Also derive `acceptance_criteria` — the **success contract** (what "done" means):
`output_completeness` (required outputs, e.g. transcripts), `quality_standard`
(a metric target, `absolute` like "studio" or `relative` like "best 20%"), and
`yield` (how much to keep). Classify each `absolute`/`relative`. These drive both
`validate` (output-completeness + request-type sanity) and the final `verify`.
For `quality_standard` / `distribution` with `scope: aggregate`, normal
run/report/reuse evidence is the arithmetic mean of that finite numeric field
across every valid terminal-manifest row. A partial scan or a missing/non-numeric
value on any retained row remains unverifiable.

### 2. Inspect (always before planning)

```bash
python -m nemo_curator.audio_agent context --goal '{"task":"quality_filter","domain":"read"}' --data /path/to/data
```

`context` returns the L0 `category_tree`, the profiler's `data_profile`
(sample rate, channels, transcripts present?, file count) and `env_profile`
(GPU? ffmpeg? installed extras?), matched blueprints/recipes, and patterns.
Tell the user what you found (it is often news to them).

**Same folder already curated?** Before inventing a recipe, list priors and compare
the user's current request to each prior's recorded prompt and `pipeline_summary`:

```bash
python -m nemo_curator.audio_agent runs --data /path/to/data --goal "the user's current request"
```

Prefer a prior whose capabilities cover the full request over one that only covers a
subset. Show the top 2–3 with stats; on pick use `delta-run --from-run <run_id>` rather
than retyping stages. `pipeline_summary` is the full stage+param list for comparison —
never paste it verbatim; retell it in one line, naming only what differs between the
priors you show. See `references/reuse.md`.

For environment health specifically, run `doctor` (the single source of truth for env):

```bash
.venv/bin/python -m nemo_curator.audio_agent doctor --json
```

Run it before a heavy GPU run, and first whenever anything env-related looks wrong (import /
CUDA / model-load errors). Do NOT diagnose env from stage cards — cards defer to `doctor`
(details in ENVIRONMENT.md).

**A "no GPU" reading is NOT a hardware fact when the run may be sandboxed.** The tool
sandbox (and containers) can block GPU device access (`/dev/nvidia*`), so `doctor`/`context`
report `has_gpu=false` even when a GPU is physically present and usable. If the env shows
`gpu_possibly_masked=true` — or `nvidia_smi`/`nvidia_device_nodes` indicate hardware, or torch
is a CUDA build (`torch_cuda_built=true`) — do **NOT** tell the user "no GPU". Say "GPU not
reachable from this (sandboxed) run" and **re-verify with full device access**: rerun `doctor`,
and any `smoke`/`run`, with full permissions (outside the sandbox). Only conclude "no GPU"
after a full-access probe still finds none. GPU‑touching verbs (`smoke`/`run`, and the GPU
probe) should run with full device access, not sandboxed.

The deterministic core supplies the facts and grounded options; **you are the
environment analyst**. `context` includes machine-wide `env_health`, and
`validate` / `smoke` / `run` return a recipe-aware `environment_decision`.
When `decision_required` is true:

1. Stop before execution. State the detected fact and confidence separately from
   any inference, then explain how it affects the selected recipe/stages.
2. Recommend the best **available** option using the user's stated constraints
   (for example: host changes allowed but environment changes forbidden), explain
   the material trade-offs of the other viable choices, and ask one outcome-level
   question.
3. Never silently install/upgrade/downgrade, change the launch command, request a
   secret value in chat, switch to CPU, change a model/decoder/stage, or retry the
   same action. The core's choices are proposals, not authorization.
4. Offer CPU only as a conditional candidate when the packet proves every affected
   flattened execution leaf has a supported CPU path; do not call it executable
   until the candidate builds and smokes. A CPU/CTC/model/stage alternative is a **new recipe**
   and config hash: validate -> smoke -> present -> confirm again.
5. After a host/environment/credential fix, rerun `doctor --json` and recipe
   preflight. Never assume the change worked.
6. Scope evidence to the execution target. Do not treat driver GPU, ffmpeg,
   credential, disk, Python, or uv-launch facts as external-worker facts. A
   driver/toolkit mismatch blocks a known runtime-PTX/JIT path; other GPU stages
   require bounded smoke evidence before proposing an invasive host change.

For an execution error, call:

```bash
.venv/bin/python -m nemo_curator.audio_agent diagnose --error '...' --recipe recipe.yaml
```

Use its sanitized classification, live preflight, `attempted_actions`, and
grounded choices. If status is `unknown`, say so and collect only the packet's
minimal diagnostics; do not invent a root cause or fix.

### 3. Route coarse-to-fine, then resolve outcomes to params

Read `references/routing.md` now. It covers the L0 -> L3 narrowing, how to read a
resolved contract (and why you must pass your intended params), closing key
mismatches without adding a stage, composites hiding their requirements, choosing
between overlapping stages, pruning to the request, and `resolve` for turning an
outcome label into a threshold. Do not pick thresholds by hand.

### 4. Plan -> validate -> critique (static loop, <= 3 iterations)

Emit a Recipe (YAML): `{stages: [{ref, params}], inputs, preset,
acceptance_criteria, planning_preference}`. The complete success contract **must
live inside this recipe before validation, smoke, confirmation, and run**. That
makes it part of `config_hash`; a separate criteria file alone is not executable
intent and must never be the only copy. `planning_preference` is different:
optional, non-semantic planning provenance that does not change any recipe hash.

For every recipe-driven verb, the first supported source stage's configured
parameter is execution truth. `Recipe.inputs` and `--data` are optional
consistency assertions: they never populate or rewrite that stage. Omit
`--data`, or pass the same canonical source. A missing, mismatched, unsupported,
or unsafe ambiguous source makes validation fail and execution refuse. A
multi-manifest `ManifestReader` may run as authored only with singular `--data`
omitted; it remains unkeyed and unreusable until aggregate identity is
supported. (`context --data` is the pre-recipe exception: it profiles that path
directly for planning.)

Save the recipe **under the scratch directory**, not in the current directory. A recipe written
for one request is working material, and the working directory is usually a git checkout, where
it shows up as an untracked file that looks like unfinished work:

```bash
python -c 'import nemo_curator.audio_agent as aa; print(aa.scratch_dir())'
# -> <workspace>/.audio_agent_runs/recipes   (git-ignored, moves with AUDIO_AGENT_RUNS_DIR)
```

Put it somewhere else only when the user asks for the recipe itself as a deliverable. Then:

```bash
python -m nemo_curator.audio_agent validate --recipe "$(python -c 'import nemo_curator.audio_agent as aa; print(aa.scratch_dir())')/recipe.yaml" \
  --data /path/to/data --acceptance-criteria criteria.yaml --request-type quality_filter
```

The three templates in `nemo_curator/audio_agent/recipes/` each ship a filled-in
`acceptance_criteria` block; start from one rather than writing the list from scratch.

The optional `--acceptance-criteria` file is a cross-check and must match the
recipe's embedded `acceptance_criteria`; validation fails if they differ.
Together with `--request-type`, the criteria compile each criterion's
output/metric into a producible-role check (so "success needs transcripts, no ASR
stage" fails here as `missing_output_producer`) and runs **request-type sanity**
(a filtering request with no `yield` criterion is flagged `missing_implied_criterion`).

Read the `Verdict`. If not `runnable`, fix from the issues and re-validate. **A gap
here means your candidate-card set was incomplete — you are NOT limited to the first
set.** When `validate` names a missing role, do a **targeted re-retrieval** (it tells
you *which* role, so query only that: the producing `cards --category`, or the role
graph in `context`), add the producer, and re-plan. This is the retrieve↔plan loop.

- `unsatisfied_reads` / `unproducible_roles`: insert an upstream producer via targeted
  re-retrieval (role graph in `context`); only if a role is truly unproducible across
  the whole catalog is the goal impossible with these stages -- tell the user.
- `dangling_key`: align the producer's `*_key` value with what the consumer reads.
- `tensor_into_sink`: insert `AudioToDocumentStage`, then write its
  `DocumentBatch` with `DocumentBatchJsonlWriterStage`. Keep
  `ManifestWriterStage` for already-serializable `AudioTask` flows only.
- `card_*`: honor the model constraint (e.g. `batch_size` fixed, `<= max_speakers`).
- `ffmpeg_missing` / `missing_secret` / `gpu_unavailable`: surface as setup steps.

#### Mandatory semantic critique (mechanical pass != intent approval)

A `runnable: true` / `status: pass` Verdict proves that the recipe is mechanically
composable under the checks the core can enforce. It does **not** prove that a
valid field means what the user meant, that a filter is applied at the right
entity/granularity, or that the chosen model/metric is a good proxy for the
request. Treat green validation as necessary plumbing evidence, never as an
intent verdict. Likewise, `semantic_review`, `review_required: true`, and the
included `required_response` are only evidence/instructions: they do not mean
the host emitted or satisfied the response contract.

**Semantic verification checklist — run it on every field you filter and every
stage you pick, grounding each answer in the packet's `semantic_facts`/notes (or
the source), NOT the key name.** For each such field/stage, answer five questions:

1. **Meaning + unit** — what does this field actually represent, in what unit?
   Read the producer's `semantic_facts[field].meaning`/`unit`; if absent, inspect
   the stage source. A plausible key name is NOT meaning — `num_speakers`,
   `num_segments`, `duration`, `sample_rate` all read one way and behave another.
2. **Scope / entity / granularity** — WHOSE value is it at THIS point in the
   pipeline: the original file, a fan-out child (per speech-segment / per speaker
   turn), or a recombined/aggregate output? Scope is set by where the field is
   produced. A per-segment field *after a fan-out* is not a per-file property, and a
   count/aggregate over children (`num_segments`, a distinct-speaker count) is not a
   per-child attribute. A whole-clip property (e.g. "single speaker") is not something
   a per-segment field can filter.
3. **Provenance** — is it measured, a configured TARGET, or a relative label? A
   target (`output_sample_rate`, resample `sample_rate`) verifies/produces a value;
   it does not describe the input. A diarizer `speaker` is a per-recording cluster id
   (speaker_0 ≠ a person, ≠ comparable across files), not a global identity or a count.
4. **Stage effect vs intent** — does the stage TRANSFORM what the user said stays
   fixed, or DROP rows they expected kept? Separation/resample/mono change the audio;
   strict-rate mono DROPS rate-mismatched rows (it does not convert them); every
   filter DROPS. If intent implies "unchanged / just measured", a transform/mixer is
   the wrong tool.
5. **Direction** — for a metric filter, is lower or higher "better"
   (`metrics.scale.direction`)? Keep the correct side of the threshold (WER/CER are
   error rates → drop ABOVE, not below).

If the honest answer to "does filtering/using this field at this point achieve the
user's stated intent?" is no — or you cannot ground meaning/scope from cards or
source — the critique is `revise` (choose the right field/stage) or `ask`, never a
silent green. **Trigger points that demand this check before you proceed:** a
fan-out or nesting seam, a join/aggregate/concatenation, any transform/materialization
whose output a later stage may or may not read, a diarizer speaker label, and any
count/`num_*`/rate/duration field.

**Worked trap (why this checklist exists).** "Keep only single-speaker clips" does
NOT mean `SpeakerSeparationStage` + filter `num_speakers == 1`: `num_speakers` is the
count of speakers the diarizer found in the ORIGINAL clip (scope = whole recording, an
aggregate), and Separation TRANSFORMS the audio into per-speaker streams — so that
recipe both mis-scopes the field and mangles the audio. Correct: diarize, then keep
rows whose distinct-speaker count is 1 — no separation, no per-segment speaker filter.
The plumbing (a filterable `num_speakers` key) is green either way; only the
meaning + scope + effect check catches it.

After validation is mechanically runnable, **do not smoke yet**. Inspect the
Verdict's `semantic_review` packet. It is built from configured dynamic contracts
and automatically co-locates exact-key lineage, latest/prior producers,
fan-out/nesting/aggregation/filter seams, and the producer/consumer cards'
`semantic_facts`, notes, metrics, domain and limitations. For a generic consumer
such as a value filter, use the packet's exact upstream producer rather than
inferring from the key name. If the packet reports unresolved lineage, missing
card semantics, or opaque visibility, do targeted retrieval/source inspection;
if meaning still is not grounded, mark the critique `ask` rather than inventing
it. A matching key/role proves connectivity, not meaning.

Emit one compact `semantic_critique` with:

- `mechanically_runnable`: copied from the deterministic Verdict;
- `recipe_config_hash`: copied exactly from
  `semantic_review.recipe.config_hash`; it binds the critique to this canonical
  recipe, and any recipe change requires another validation and critique;
- `intent_status`: exactly `pass`, `revise`, or `ask`;
- `stage_reviews`: for **every** stage, the user goal/acceptance criterion it
  serves (a stage with no justification means `revise`);
- `field_reviews`: for every filter/generic field consumer, its exact producer,
  the field's meaning and unit, its entity/granularity at that point in the
  pipeline, and how fan-out, nesting, aggregation or copying changes—or does not
  change—that meaning;
- `behavior_checks`: representative values/rows on both sides of every
  filter—including the boundary—stating what would be retained and dropped;
- `transform_checks`: for every conversion/materialization, which downstream
  stage consumes the transformed waveform/path/key; flag transforms whose result
  is discarded while downstream reads the original;
- `model_checks`: language/domain, model limits, metric coverage and known
  caveats against the profiled data and the user's wording;
- `assumptions_or_questions`: every unresolved semantic assumption, phrased as an
  outcome-level question if user input is required.

The result controls the loop:

- `pass` — intent and plumbing both look coherent; proceed to smoke.
- `revise` — change the recipe, then re-run **validate -> semantic critique**.
  Any recipe change invalidates the prior critique and config hash and follows
  the mid-workflow recipe branch reset above.
- `ask` — stop before smoke and ask the minimum user-facing question. After the
  answer, revise as needed and repeat validation + critique.

On `pass`, before authoritative smoke, read the `checkpoint-placement` skill and call
`plan-checkpoint`; select only a core-generated candidate, or use the core-returned baseline
after the user explicitly declines. Smoke will refuse an unresolved recommended choice.

This is an LLM judgment over grounded card/data evidence, not a request for a new
deterministic rule per module. Crashes, corruption, impossible composition and
other universal hard invariants remain core gates; intent-dependent meanings and
trade-offs remain here.

### 5 to 7. Smoke -> confirm gate -> run -> report

Read `references/smoke-and-run.md`. It covers the bounded smoke and its
`calibration` block, what must be presented at the confirm gate, the
`config_hash` integrity check on `run`, the guardrails the tool enforces, and how
to report and verify acceptance including the reviewer charter.

### 8. Don't redo finished work

Read `references/reuse.md` before smoking or running. It covers `reuse-scan`, the
three conversation rules (never read prior artifact content, never reuse silently,
never nag), and acting on the choice with `continue`.

## Control conditions

- **Success**: `runnable` + smoke goals met -> confirm -> run -> report.
- **Escalate**: static/empirical loops exhausted, or an unproducible role -> hand
  the user the best candidate + the blocking issues.
- **Refuse**: intent on the refuse list -> stop + safe alternative.

## When the catalog cannot help (skill fallback)

If the goal matches no recipe and you cannot ground a plan from the cards
(e.g. a brand-new stage without a card), say so explicitly and reason from the
stage contracts (`describe`) rather than guessing parameters.
