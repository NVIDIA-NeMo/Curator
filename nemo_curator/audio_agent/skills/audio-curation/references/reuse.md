# Don't redo finished work: reusing prior runs

Loaded on demand from `SKILL.md` step 8. Read this once you know the data path —
**before inventing a recipe** — and again once you have a candidate recipe, before smoking.

## Before inventing a recipe: same-folder priors by prompt + summary

Ranking a freshly invented pipeline against prior runs by stage edit-distance is what made
a shallow convert-only prior outrank richer work that already covered most of the folder.
Do not invent first. Compare the **current user request** to each prior's recorded
**prompt** (`goal`) and durable **`pipeline_summary`**:

```bash
python -m nemo_curator.audio_agent runs --data /path/to/folder \
  --goal "the user's current request"
```

Each card carries `prompt`, `pipeline_summary`, stats, and (when `--goal` is set) a
`match` score: the fraction of the current request's tokens covered by that prior's
prompt + summary. Prefer a prior whose capabilities cover the **full** request over one
that only covers a subset. Show the top 2–3; on pick:

```bash
python -m nemo_curator.audio_agent delta-run --from-run <run_id> --data /path/to/folder
```

Successful runs persist `pipeline_summary` automatically. Older records without it are
summarized at read time from the stored recipe. Always pass `--goal` on `run` /
`delta-run` so the next session has a prior prompt to compare.

`pipeline_summary` is the **complete** stage list with every behavioural param — written to
be compared, not read aloud. It is deliberately not truncated: clipping it would decide for
you which threshold matters, and two runs differing only in that threshold would look
identical. **Never paste it verbatim.** Retell it in one plain line: what the run produced,
plus only the settings that differ between the priors you are showing or that the user asked
about. The payload repeats this in `host_directive`.

Once you have a candidate recipe — **before** smoking or running — also scan:

```bash
python -m nemo_curator.audio_agent reuse-scan --recipe recipe.yaml --data /path/to/data
```

## A recipe branch is not reuse or continuation

Replacing one component with another (for example, BandFilter with UTMOS), changing
the metric or acceptance criteria, changing a semantic stage parameter, reordering
stages, or changing checkpoint topology creates a new recipe branch. Do not call
that threshold feedback, delta work, or continuation to inherit recipe evidence.
The old validation, host critique, checkpoint choice, reuse scan, smoke, and
approval are invalid. Same-chat context, a still-valid dataset profile, and the
soft curation preference may be inherited without needless repetition.

Construct the exact new recipe and embedded contract, validate it, emit the
exact-hash host semantic critique, resolve checkpoint placement from explicit
user choices, and revalidate/re-critique any transformed recipe before this
reference's `reuse-scan`. Then smoke the exact final hash, present its evidence
and limitations, checkpoint effects, and any grounded occupied-output warning,
ask for approval, and stop. Only a subsequent user answer may authorize the
execution verb; hash/token gates do not prove consent.

For an expected same-dataset feedback loop, the proactive first-run decision happens even
before this scan: after validation and semantic critique, use the `checkpoint-placement`
skill and `plan-checkpoint`. It may choose only a core-generated candidate and must first
show its trade-off through structured AskQuestion. Never call `--choice baseline`,
`--choice checkpoint`, `--output-path`, or an equivalent MCP choice until the user selects
that response; ask separately for the path after acceptance. The soft planning-mode answer
is not checkpoint consent, and at most one checkpoint may be added. Any returned
transformation requires validation and a new exact-hash host critique. Unresolved
recommendations are refused by smoke and run. Do not confuse that opt-in preparation with
the reactive `prior_unsaved` offer below, which explains why work that already ran cannot
now be resumed.

For feedback on a completed retained checkpoint, scalar decisions use
`plan-checkpoint --from-run ... --decision-value <scalar>`. Card-declared compound
decisions use the separate `--decision-conditions '<JSON>'` surface. Supply the
complete desired non-empty condition set (configured score keys, finite numeric targets,
`ge` only); omission disables a dimension. Never put a list/mapping into
`decision_value` or edit selector recipes manually. The core preserves AND logic,
missing-score drop, nested-list parent policy, producer identity, and checkpoint identity,
and refuses a newly enabled dimension when the completed annotation checkpoint does not
prove that score is present.

Every completed step publishes a content-addressed **artifact**, so a later request can
reuse it instead of recomputing it (design: `nemo_curator/audio_agent/REUSE_ARCHITECTURE.md`).

The `decision` is `already_done` (this pipeline matches a prior computation and
dataset key at the reported trust tier), `incremental` (the first *N* stages are
already done — e.g. resample + VAD + quality-filter exist and only ASR is new),
`delta` (the corpus key missed, but only a few files changed and the rest of the
prior result still stands — see [When only a few files changed](#when-only-a-few-files-changed)),
or `fresh`. Reuse survives things that do NOT change output bytes: a different
batch size, different `resources`, a different output path, or a **stricter
success bar** (the data is reused and the contract re-verified), and a Curator
build change that did not touch the stages in play. A detected dataset-key
change, a missing completion marker, or an edit to a stage's own implementation
prevents reuse. Shape-tier matches are low trust and default fresh because
metadata gaps may hide changes. A stage declared non-deterministic is not
refused—it is offered with that said and `fresh` pre-selected, because the result
is real, just not one a rerun is promised to match.

**Three rules for the conversation:**

- **Never read prior artifact content.** `runs` returns artifact URIs so humans can find
  their outputs — do NOT open, `cat`, or inspect those files. A prior manifest records what a
  *different* pipeline did to the data, under criteria the current request may not share, so
  reading it anchors your plan to the old design before the current recipe exists — including
  its empty or filtered-out fields, which say nothing about what this request can achieve.
  Reuse decisions go through `reuse-scan` exclusively — it works on config hashes, not
  content. Artifact URIs are for human reference only.

- **Never reuse silently.** If `prompt_user` is true, disclose before running. Usually that
  means a reuse candidate: show its card — objective, pipeline, input/output, key params, date,
  metrics, estimated time saved — and offer the three choices, defaulting to the scan's
  `recommended` (which is `fresh` whenever `trust` is `low`; say why — the `weaknesses` list is
  written for a human). But `prompt_user` is *also* set when there is no reuse candidate and only
  a `prior_on_same_path` notice — a `fresh`/`delta` result over a folder curated before. Then
  there is no as_is/extend card to show; surface the notice instead (see
  [When you've curated this folder before](#when-youve-curated-this-folder-before-with-a-different-pipeline))
  and let the user choose to align or proceed. Read `prior_on_same_path` whenever it is present —
  do not summarise a scan from `decision`/`prompt_user` alone, which is how a correct notice went
  unspoken and the user was told "nothing to reuse" over a folder they had just curated.
- **Never nag.** `prompt_user: false` means don't ask: either there is nothing to reuse
  (just run) or the saving was *measured* and is trivial (take it, and mention it in your
  summary). When `unpriced_stages` is non-empty the question is not about the size of the
  saving — nobody timed those stages and the cards call them expensive — so say that rather
  than quoting the `estimated_saving_sec`, which is a floor and will look absurdly small.

Then act on the choice — don't hand-edit the recipe:

Before an executed `extend` or `fresh` continuation reaches its final confirmation,
inspect the continuation card together with deterministic `output_targets`, resolved
stage output-path contracts, and current read-only path facts. Name each exact occupied
manifest/data/output/checkpoint path that the chosen execution is proven to replace or
overwrite, and remind the user to copy or save anything there they need. Do not mutate,
clean, rename, pre-create, or back up the path for them. If the card does not prove
append-versus-replace behavior, say so and ask. Do not issue an overwrite warning for
`as_is` when it only serves an existing artifact without mutation.

```bash
python -m nemo_curator.audio_agent continue --recipe new.yaml --data /path/to/data \
  --execute --choice extend --confirm <config_hash>
```

- **as_is** — serve the completed output and re-check it against today's criteria.
- **extend** — rewrite the recipe to start from the reused artifact, re-validate the
  remaining stages against what that artifact actually carries, and run only those.
- **fresh** — ignore prior work and recompute.

After the choice succeeds, include the **Saved files** block required by
`smoke-and-run.md`. For `as_is`/`already_done`, distinguish the existing served/reused
artifact from newly written files and say **No new output was written** when the result
proves that. For an executed `extend` or `fresh`, distinguish its newly written/replaced
terminal outputs from the existing artifacts it read and from checkpoint/intermediate
files. Report the exact recipe path only when run-record recipe metadata or a returned
scratch recipe path identifies it; otherwise say the recipe path was not reported.

`--parent-run-id <id>` is optional and additive: it diffs against that specific run, and
whichever engine reuses more wins. `runs --data /path/to/data` shows everything already
done to a corpus; `reindex` rebuilds the lookup index from the JSON records if it is lost.

## When only a few files changed

A dataset key names the whole corpus, so adding one file to a curated folder misses every step
key and the plain reading of that miss is "recompute all thousand files". When the scan can do
better the `decision` itself is **`delta`**, not `fresh`, with `recommended: delta` and a
`choices` list whose first entry is running the changed files only. `key_matched: false` records
the miss the decision rests on — the key did miss; a full rerun is still the wrong response to
it. **A `decision: delta` card must never be answered with a full `run`** without putting the
delta to the user first, with its `estimated_saving_sec`. Recurating files that are already
done is the failure this exists to prevent, and it has happened: a host read `fresh`, recurated
the whole corpus, and told the user a checkpoint was missing when none was needed. The `delta`
block names the changed files
(`change.added_files`, `modified_files`, `removed_files`), says which stages would run
(`run_stages`), how many prior rows survive (`rows_kept`) and how many are dropped and
recomputed (`rows_dropped`).

```bash
python -m nemo_curator.audio_agent delta-run --recipe recipe.yaml --data /path/to/data \
  --confirm <config_hash>
```

The unconfirmed delta card is the pre-run source of truth. Before asking for that
confirmation, combine its manifests/targets and change/merge facts with deterministic
`output_targets`, resolved stage contracts, and current read-only path facts. Name every
exact occupied path the delta is proven to rewrite, and remind the user to save/copy work
they need first. Never clean or back it up automatically. Do not generalize the in-place
manifest rewrite below to append stages or unrelated outputs; if another target's behavior
is unproven, say so and ask.

This runs the user's own stages over the changed files, merges the rows into the existing
manifest, and republishes it under the key the full pipeline has for the enlarged corpus — so
the next `reuse-scan` answers `already_done` by an ordinary probe. It rewrites the manifest in
place after merging, which is why it is confirm-gated like `run`.

`status: tail_required` means the merge is done but the curation is not: the delta owned only a
prefix of the recipe, so the stages after the checkpoint still have to see every row. The files
listed in `tail.stale_outputs` are on disk describing the corpus as it was *before* the change,
so do not report the work as finished — run the `continue --execute --choice extend` from its
`next` first, then report. Only `status: completed` means the deliverable is current.

When a delta (and any required tail continuation) completes, the final response must
separate the manifest/output paths newly written or replaced by this execution from prior
rows/artifacts that were reused and from checkpoint/intermediate files. Include generated
audio/output directories and the exact executed recipe path when the structured result
reports them. Never list `tail.stale_outputs` as current saved deliverables, and say “not
reported” for any requested durable path absent from the structured evidence.

`status: no_delta` is an answer, not an error, and its `reason` is worth relaying because it
says what would have to change. Common ones: nothing is persisted early enough to merge into
(`add-checkpoint` fixes it), a stage in the prefix has not declared that it computes each row
from that row alone, the prior rows cannot be traced back to the files that produced them, or
the two corpora share no file at all — a different dataset rather than a changed one. A refusal
means a full run; it never means a partial result presented as a whole one.

## When you've curated this folder before with a different pipeline

`already_done`, `incremental` and `delta` all match on the step-key chain, which a changed
source stage or changed corpus moves wholesale — so a folder you curated an hour ago becomes
invisible to them the moment the recipe drifts, even slightly. And it drifts easily: the same
request planned twice is not bit-identical, and one threshold written out where it was defaulted
before is enough to miss every key. The scan closes that blind spot with `prior_on_same_path`,
present on a `fresh` or `delta` result whenever a prior **completed run read the same source
folder**, matched by path rather than by recipe. It is advisory: it never changes `decision` and
reuses nothing. It carries the closest match at the top level, plus `count` and the ranked
`matches` (closest pipeline first, then most recent) when the folder was curated more than once:

- `created_at` and `run_id` — when, and which run.
- `recipe_diff` — `added_stages`, `removed_stages`, `changed_params` (`{stage, param, from, to}`),
  and a human `phrase`. This is how you see that last time used a different source stage, or a
  threshold set to one value where you now have another.
- `data_delta` — added / modified / removed / unchanged file counts and names since that run
  (`basis: inventory`), or a labelled count comparison when no per-file record was kept.
- `recommendation` — `delta` (same pipeline, only the corpus moved → the changed-file path is the
  cheap answer), `align` (a different pipeline → adopting that run's recipe is what makes its work
  reusable), or `fresh`.
- `next` — the two commands, with the real ids filled in: `inspect` and `adopt`.

### The conversation this is for

A user who curated a folder last week and comes back with the same intent worded differently
should hear about the earlier work **before** anything runs. Four steps, in order:

1. **Say what exists, briefly.** "This folder was curated on <date> (<`goal`>). <`data_delta.phrase`>."
   With `count` above 1, name the closest two or three from `matches` — date, objective, and how
   each differs — rather than listing everything.
2. **Ask whether to build on it**, and offer to show more. Do not decide for them, and do not
   start a fresh run while the question is open.
3. **If they want detail, get it from the record** — never by reading the prior output:

```bash
python -m nemo_curator.audio_agent runs --run-id <run_id>
```

   The `overview` block is written for this moment: `pipeline` (the stage chain), `key_params`
   (what each stage was set to), `data` (source, dataset key, file count), `stats` (elapsed,
   accepted rows, slowest stages), `acceptance` (its verdict, per criterion) and `outputs`.

4. **If they say yes, adopt that run's own recipe and run only the delta.** Do not retype the
   pipeline — that is what made the prior work invisible in the first place:

```bash
python -m nemo_curator.audio_agent delta-run --from-run <run_id> --data /path/to/folder
```

   Without `--confirm` this returns the card: `adopted_from` (the run, its objective, the
   pipeline being adopted), the `delta` block (which files are new/changed/gone, which manifests
   get rewritten, `rows_kept` / `rows_dropped`, `estimated_saving_sec`) and the `config_hash` to
   confirm with. Relay it, then confirm. The result merges the new rows into the prior manifest,
   so the deliverable covers the whole folder — the prior work plus the delta — not just the new
   files. Report it that way, and say which part was reused.

`--from-run` refuses rather than guessing: passing a recipe *and* `--from-run` together is an
error (adopting means running that run's stages, not yours), a run that did not complete has no
result to extend, and a recipe whose credential was masked in history cannot be reproduced — that
last one names the param and asks for the recipe with the value supplied. If the delta itself is
unavailable, `status: no_delta` carries the reason (see [When only a few files changed](#when-only-a-few-files-changed));
adopting the recipe is still what makes the *next* run reusable, so keep it and run it normally.

**Never silently run fresh over prior work.** Proceeding fresh is a legitimate choice — it is
just not yours to make quietly. "I've done this here before" is a fact the user should hear, and a
step-key miss is not allowed to hide it. It stays a notice, never an action: reuse happens only
through `continue` / `delta-run`, never by editing bytes.

`runs --data /path/to/folder` answers the same question outside a scan: it lists runs on that
exact corpus *and* runs that read the folder when its contents differed, with the latter named in
`same_folder_only`. Pass `--goal` to rank those cards by current-request coverage of each prior's
`prompt` + `pipeline_summary` before inventing a recipe.

## When the scan says the work was done but nothing was saved

A stage only leaves something to resume from if it was configured to write somewhere. A
pipeline whose GPU stages hand their rows to the next stage in memory has nothing on disk, so
`decision` is `fresh` even though the transcription ran last week — the scan discloses this as
`prior_unsaved` and attaches an `offer`.

When the offer's `action` is `add_checkpoint`, relay it: one `ManifestWriterStage` in their
recipe makes the expensive stages resumable from then on. Get the recipe rather than editing
by hand — the position is not always where the expensive work ends, because a manifest cannot
hold a waveform that is still in memory, nor state some stages pass to each other outside the
row:

```bash
python -m nemo_curator.audio_agent add-checkpoint --recipe recipe.yaml --data DATA
```

The location is derived from the dataset and the step it resumes, so do not ask the user
where to put it; pass `--output-path` only when they asked to keep the metadata somewhere of
their own. This returns the recipe with the writer in place and changes nothing on disk. Save it, then
`validate` → smoke → `run` as usual. In this reactive post-miss path,
`action: no_checkpoint` means don't raise it: the `why` says whether the work is too cheap
to be worth a file, a writer is already there, or the pipeline holds audio in memory to the
end. Never invent a reactive checkpoint the offer did not. Proactive checkpoint selection
for a not-yet-run recipe belongs only to `checkpoint-placement` + `plan-checkpoint`.

Record what a run was FOR with `run --goal "..."` — that objective is the prior **prompt**
the next session compares against, together with the stored `pipeline_summary` written on
success. Empty goals force ranking to fall back to the summary alone.
