# Audio Agent — agent guardrails and workflow

These instructions apply to any AI coding agent working in `nemo_curator/audio_agent/`.
Every host reads this file: Codex and Cursor load a nested `AGENTS.md` automatically for
work in this directory, and Claude Code reaches it through the sibling `CLAUDE.md` import.

The full procedure lives in `skills/audio-curation/SKILL.md`, with its long sections under
`skills/audio-curation/references/`. This file carries only what an agent must know
*before* it decides to read that skill.

## Treat the Curator repo as read-only for dataset use cases

While using the audio agent to build/curate a dataset (ALM windows-type datasets,
quality/duration filtering, resampling, etc.), you are a **consumer** of Curator,
never an editor of it.

- NEVER create, edit, or delete any file inside this repository (`nemo_curator/`,
  pipeline/stage source, the `audio_agent/` source, repo configs, tests, etc.) in
  order to satisfy a particular user's dataset or use case.
- NEVER tweak stage/pipeline source, thresholds, filters, windowing logic, or agent
  code because a single dataset produced empty or unexpected output.
- Treat everything under the Curator repo as READ-ONLY reference you may read and run,
  but not change, while serving a use case.

A dataset that comes out empty or small is a data/config problem, not a reason to patch
shared library source. Editing the repo to force a single use case corrupts the library
for everyone and hides the real, data-level explanation the user needs.

Scope note: this bans repo edits made to force a specific dataset/run to "work". It
does not ban deliberate, separately-requested development of Curator or the audio
agent itself — that is a different task, done explicitly.

### What to do instead when output is empty/unexpected

1. Diagnose first — do not "fix" code. Empty output is usually correct for the input.
   Example: ALM speaker-turn `windows` are only produced from multi-speaker
   turn-taking audio; single-speaker clips legitimately yield `"windows": []` (seen as
   `lost_win` in the stats). Expected, not a bug.
2. Explain the cause to the user from the data (`duration`, `num_speakers`,
   `stats`/`lost_*` fields, thresholds actually applied).
3. Adjust inputs and configuration, not the repo. Point the pipeline at appropriate
   data, or edit recipe/criteria/config files OUTSIDE the repo (e.g. under the user's
   working directory such as `adv_test/`); write outputs to user-owned directories.
4. If the goal genuinely requires a Curator code change, STOP and ask for explicit
   confirmation, describing the change and impact, before touching any repo file.
   Never do it silently as part of "getting the dataset to work".

## Route audio work through the tool core

For ANY task involving audio files or audio datasets — file durations, quality filtering,
VAD/segmentation, resampling, mono conversion, transcription, WER, diarization, building a
TTS/ASR training set, exporting a manifest — drive the `nemo_curator.audio_agent` verbs.
**Do NOT write ad-hoc shell/Python** (ffprobe, soundfile, librosa, torchaudio loops) to
inspect or process the audio, even for a "quick" one-off like reading durations — use
`GetAudioDurationStage` through the agent.

On a fresh clone there is no `.venv/` yet, and every verb dies on `ModuleNotFoundError: No
module named 'cosmos_xenna'` before it can diagnose anything — `doctor` cannot tell you this
itself. Create the environment first, once:

```bash
uv sync --extra audio_cuda12   # --extra audio_cpu if there is no GPU
```

All verbs print JSON. Run them with the repo virtualenv interpreter from the repo root
(base `python` lacks Curator's deps), or as `nemo-curator-audio <verb>` from an install:

```bash
.venv/bin/python -m nemo_curator.audio_agent discover        # stages (name, category, one-liner)
.venv/bin/python -m nemo_curator.audio_agent catalog-tree    # L0 category tree (route over this)
.venv/bin/python -m nemo_curator.audio_agent cards --category quality        # L1 one-liners
.venv/bin/python -m nemo_curator.audio_agent cards --names UTMOSFilterStage  # L2 full cards
.venv/bin/python -m nemo_curator.audio_agent describe UTMOSFilterStage --params '{...}'
.venv/bin/python -m nemo_curator.audio_agent producers duration             # who writes a key
.venv/bin/python -m nemo_curator.audio_agent context --goal '{...}' --data DATA
.venv/bin/python -m nemo_curator.audio_agent runs --data DATA --goal '...'  # BEFORE inventing a recipe
.venv/bin/python -m nemo_curator.audio_agent doctor --json
.venv/bin/python -m nemo_curator.audio_agent diagnose --error '...' --recipe R.yaml
.venv/bin/python -m nemo_curator.audio_agent resolve --stage UTMOSFilterStage --label studio
.venv/bin/python -m nemo_curator.audio_agent validate --recipe R.yaml --data DATA
.venv/bin/python -m nemo_curator.audio_agent reuse-scan --recipe R.yaml --data DATA
.venv/bin/python -m nemo_curator.audio_agent smoke --recipe R.yaml --sample 10 --data DATA --bootstrap-ray
.venv/bin/python -m nemo_curator.audio_agent run --recipe R.yaml --confirm <hash> --data DATA --bootstrap-ray
.venv/bin/python -m nemo_curator.audio_agent report --output OUT --data DATA
.venv/bin/python -m nemo_curator.audio_agent verify --criteria C.yaml --evidence E.json --recipe R.yaml
.venv/bin/python -m nemo_curator.audio_agent continue --recipe R.yaml --data DATA --execute --choice extend --confirm <hash>
.venv/bin/python -m nemo_curator.audio_agent add-checkpoint --recipe R.yaml --data DATA  # make GPU work resumable (location derived; never ask for a path)
.venv/bin/python -m nemo_curator.audio_agent plan-checkpoint --recipe R.yaml --data DATA  # core-proven pre-gate candidates (location derived; never ask for a path)
.venv/bin/python -m nemo_curator.audio_agent checkpoints            # what the managed cache holds
.venv/bin/python -m nemo_curator.audio_agent checkpoints --gc       # drop orphaned/expired checkpoints
.venv/bin/python -m nemo_curator.audio_agent delta-run --recipe R.yaml --data DATA --confirm <hash>  # only the files that changed
.venv/bin/python -m nemo_curator.audio_agent delta-run --from-run RUN_ID --data DATA  # same, adopting a prior run's own recipe
```

`smoke` and `run` need a Ray cluster. `--bootstrap-ray` starts a correctly-configured
local head (free port, plasma on /tmp, API limit) so no manual setup is needed; if a
cluster already exists, set `RAY_ADDRESS` and omit the flag (it is respected, never
clobbered).

## Ask for the soft curation mode once

At the start of each **new executable** audio-curation workflow, before
`context`, routing, or recipe construction, use the host's fancy structured
single-select UI:

- Title: **Optimize this curation**
- Question: **How should I optimize this curation? This only guides choices between equally correct pipelines.**
- **Easy to refine later** (`refine_later`): reusable file-backed handoffs,
  exact annotate/select forms, and at most one worthwhile metadata checkpoint;
  the first run may use more metadata storage/I/O.
- **Fastest first run** (`fast_first`): adjacent in-memory handoffs, native
  filters, early row reduction, and minimal intermediate I/O; later tuning may
  repeat model work.

Infer `fast_first` from "fast/as quickly as possible" and `refine_later` from
"tune/refine thresholds/reuse later"; otherwise ask. Ask only once: validation,
smoke, approval, reporting, threshold feedback, and explicit continuations do
not ask again. Continuations inherit the stored `planning_preference`. The same
folder alone does not prove continuation, and a new unrelated request asks
again.

This is a soft tie-breaker, never a correctness constraint or hard bound.
Correctness, user intent, acceptance criteria, validation, semantic critique,
safety, and stage contracts win. If the preferred form is illegal,
semantically different, unavailable, or not worthwhile, use the best correct
plan and briefly explain why. Never add pointless checkpoints, delay a useful
filter into excessive model work, write intermediate audio solely for reuse, or
bypass the existing recipe-specific checkpoint decision gate. The question is
host-UI behavior; do not add an AskQuestion verb to core.

## Reset on a mid-workflow recipe branch

Any stage add/remove/replace/reorder, semantic stage-parameter change,
acceptance-criterion change, or checkpoint-topology change creates a new recipe
branch and invalidates every prior recipe-level validation, semantic critique,
checkpoint decision, reuse scan, smoke, and execution approval. In particular,
a topology, metric, stage, or criteria replacement is not threshold feedback,
delta work, or continuation. A threshold change also invalidates recipe
evidence. Same-chat context, a still-valid dataset profile, and the soft
curation preference may be inherited; do not needlessly ask/profile again.

The required restart is: construct the exact new recipe + embedded contract ->
`validate` -> host semantic critique with `mechanically_runnable`, exact
`recipe_config_hash`, and `intent_status` -> checkpoint placement /
`plan-checkpoint` -> explicit user selection for every offered accept/decline
and path decision -> if transformed, validate + critique again -> `reuse-scan`
-> authoritative smoke of the exact final hash -> present smoke/limitations,
checkpoint effects, and any occupied-output copy/save warning -> explicit
post-smoke user approval -> `run` -> report + **Saved files**.

Never invoke `plan-checkpoint --choice baseline`, `--choice checkpoint`,
`--output-path`, or an equivalent MCP choice for the user. First show the
candidate trade-off with the host's structured AskQuestion UI and use only the
selected response; the earlier soft planning-mode choice is not checkpoint
consent. Never call `run` in the same response that reports smoke: only a
subsequent user answer can approve execution. Hashes and smoke tokens bind
artifacts, not consent.

`validate` returning `semantic_review`, `review_required`, or a response schema
does not mean the host performed semantic critique; the complete response
contract above is required for the exact hash. The core cannot prove
AskQuestion provenance or host judgment, so do not invent consent tokens or
hard gates that break existing SDK/tutorial flows.

## The loop

1. After the once-per-workflow mode choice/inference, interpret + clarify the
   goal (task, domain, quality bar, output) and derive
   `acceptance_criteria`. Refuse if out of scope.
2. Inspect: `context --data DATA` for the data profile, environment and matched
   blueprints. Report the findings — they are often news to the user.
   **Also** `runs --data DATA --goal "..."` before inventing a recipe: compare the
   current request to each prior's recorded prompt + `pipeline_summary`, show the
   top 2–3, and on pick adopt with `delta-run --from-run` (never invent a competing
   recipe first). `pipeline_summary` is the full stage+param list for comparison —
   retell it in one line rather than pasting it.
3. Route coarse-to-fine: `catalog-tree` -> prune categories -> `cards --category` ->
   `cards --names` -> `describe` with the params you intend to use. Prefer adapting a
   matched blueprint over composing from scratch, and prune to the request.
4. Plan -> `validate` -> fix from the issues -> re-validate (at most 3 rounds). A missing
   role means the card set was incomplete, so re-retrieve for that role rather than
   stopping at the first set.
5. Mandatory semantic critique -> `pass` / `revise` / `ask`. Only `pass` may continue.
   On `pass`, use the `checkpoint-placement` skill before smoke; it may select only a
   complete candidate returned by `plan-checkpoint`. Ask before applying any offered
   checkpoint or baseline choice; use only the user's selected response. Smoke/run refuse an
   unresolved recommendation; never hand-edit a checkpoint strategy or decline marker.
6. `reuse-scan` before spending compute, then `smoke --sample N` and show
   retained/rejected plus examples (at most 2 rounds). When the scan answers
   `decision: delta` (`delta.status: ready`), a few files changed since a prior run: offer
   `delta-run` instead of recurating the whole corpus, and relay its `reason` when it refuses.
   `delta-run` answering `tail_required` is NOT a finished curation -- the merge brought
   the checkpoint up to date and the files in `tail.stale_outputs` still describe the old
   corpus. Run the `continue` in its `next` before reporting anything as done.
7. Present the plan, the semantic pass, the smoke evidence, the scale estimate and the
   acceptance contract. Immediately before final approval, use deterministic
   `output_targets`, resolved stage path contracts, continuation/reuse cards, and current
   read-only path facts to name exact occupied paths proven to be replaced/overwritten;
   remind the user to copy/save needed work. Never mutate or pre-create a target, guess
   append-versus-replace behavior, or warn when all targets are new or reuse only serves
   an existing artifact. Then get explicit approval and `run --confirm <hash>`.
8. Summarize the `report`, verify acceptance, and end every successful full/delta/continue/
   reuse/serve-as-is result with a concise **Saved files** block. From structured results
   only, separate newly written/replaced final outputs, existing reused/served paths, and
   intermediate/checkpoint files; include the exact reported recipe path and generated
   output directories. Say no new output was written for non-mutating reuse when true,
   and say a path was not reported instead of guessing.

## Non-negotiables

- **Never invent** stage or parameter names. Only what `discover` / `cards` / `describe`
  return exists.
- **Ask the tools what a stage reads and writes; never grep the stage source for it.**
  Pass the params you will actually use — the contract is resolved FROM them.
- **0 silent full-scale runs.** Never `run --confirm` before the user approves, having
  seen a smoke result and the scale estimate. Report smoke and ask in one response, then
  wait for a subsequent answer before running. The `config_hash` binds the plan; it does
  not prove approval.
- **Nothing touches the filesystem before approval.** No creating, deleting, moving or
  truncating files ahead of the gate, least of all the user's output. A gate the agent has
  already prepared the ground for is not a gate. Never pre-clean an output "so the rerun is
  clean": use the resolved output-path contract to distinguish replace from append, and
  never generalize one sink's behavior to another. An agent once deleted a user's file over
  exactly that misreading.
- **A green Verdict is mechanically runnable, not intent-approved.** The semantic critique
  between `validate` and `smoke` is mandatory.
- **Evidence only.** No quality or throughput claim without before/after numbers from a
  `report`.
- **Never recurate files that are already done.** `decision: delta` means prior work covers
  every file but a few. Offer `delta-run` with its `estimated_saving_sec` before any full
  `run` — do not branch on `decision` alone and stop there. When a delta is unavailable the
  scan says why in `delta.reason`; relay that reason rather than inventing a cause. A host
  once read `fresh`, recurated a whole corpus over one added file, and reported a missing
  checkpoint that the recipe did not need.
- **`prior_on_same_path` / same-folder `runs` means this folder was curated before.**
  Call `runs --data DATA --goal "..."` *before* inventing a recipe and compare the
  current request to each prior's `prompt` + `pipeline_summary`. When keys miss after a
  recipe exists, `prior_on_same_path` still appears — surface it; on a yes adopt with
  `delta-run --from-run <id>` rather than retyping the pipeline.
- **Always pass `--goal`** on `run` / `continue` / `delta-run` so the next session has a
  prior prompt to compare against (together with the stored `pipeline_summary`).
- **Environment questions have one home:** `doctor`. Do not diagnose the environment from
  stage cards, and never silently install, upgrade, switch CPU/GPU, or retry a failure.
- **Write scratch recipes to `scratch_dir()`**, not the working directory — that is a git
  checkout, where a one-off recipe reads as unfinished work someone forgot to remove.
- **User-facing questions only.** Ask at the outcome layer, never about thresholds, keys,
  residency, batch size or model IDs.

Everything above is the short form. `skills/audio-curation/SKILL.md` is authoritative and
explains why each rule exists; read it before planning a pipeline.
