# Routing: pick stages coarse-to-fine, then resolve outcomes to parameters

Loaded on demand from `SKILL.md` step 3. Read this before selecting stages.

## Route coarse-to-fine (do NOT read every card)

- **L0**: from `catalog-tree`, pick only the categories the goal needs; prune the
  rest (no filtering intent -> skip `quality`/`filter`; no transcripts -> `transcribe`/WER
  is usually unproducible).
- **L1**: `cards --category <cat>` for the chosen categories -> shortlist stages.
- **L2**: `cards --names A B C` for the finalists -> read full cards (model
  constraints, presets, ordering hints) and select stages.
- **L3**: `describe <Stage> --params '{...}'` for the exact keys a stage reads and writes.

**Always pass the params you intend to use.** A contract is resolved FROM the params: every
`*_key` param you set becomes a key the stage really reads or writes, so a `describe` without
them describes the defaults rather than your pipeline. Read `contract_resolution`: `configured`
is the answer for those params; `static_params_and_hints` means it could not be built and
`contract_unresolved` names what to supply. An empty `reads`/`writes` under
`static_params_and_hints` means **unknown**, never "requires nothing".

**Close a key mismatch with the key param, not an extra stage.** When the producer writes a key
under a different name than the consumer's default, point the consumer at it — set its `*_key`
param to the name that was actually written, and confirm with `describe` plus those params. Use
`producers <key>` to establish who wrote it. Adding a stage to copy, rename or re-derive a value
that already exists spends compute and can invent data; redirecting the reader costs nothing.

**A default is not the only option.** A resolved contract answers for one configuration. Read
`contract_varies_with`: it names each enumerable param whose other settings would change the
reads or writes, with the resulting keys. `input_residency` is the one that decides pipeline
shape — at `file` a stage reads `audio_filepath`, at `waveform` it reads `waveform` +
`sample_rate`, at `auto` either — so a stage that looks unable to consume what you have upstream
may simply be set the other way. Check this before inserting a conversion or dropping the stage.
`reads_one_of` means "any one of these sets", not "all of them": satisfy one.

**A composite hides its requirements in its own contract**, which is empty by design. `describe`
returns `expands_to` for one: `requires_upstream` is what must exist before it starts, resolved
through your params. Check that list against the keys your upstream stages actually write — a
mismatch here is invisible to `validate` for stages after the composite, and otherwise surfaces
only after the models have downloaded and the GPU work has run.

**To find who produces a key, ask instead of reading source.** `producers <role-or-key>` matches
on either a semantic role or a literal key name. `producers` are proven writers; `candidates` are
stages that declare the role but need params before their contract can be resolved — confirm one
with `describe`. `not_searched` means the answer is incomplete, so an empty `producers` list is
not proof that nothing makes the key. Never grep the stage source to answer this.

**When two stages overlap** (e.g. two diarizers/VADs), do NOT pick arbitrarily.
Compare them on card facts (supported I/O, accuracy, language, hardware, latency,
resource, config complexity, known limitations, compatibility, goal-suitability),
then apply the decision policy: **auto** if one clearly fits best or the choice is
low-impact; **recommend** if there's a trade-off (state it, allow override); **ask**
only if the choice is material *and* preference-dependent *and* not inferable — with a
one-line plain-language difference + your recommendation (never expose internal params).

Prefer adapting a `matched_blueprint` (it encodes idiomatic ordering with
`enforced`/`advisory` tags and `topology_selection`) over composing from scratch.
Adapt, do not blindly copy.

**A blueprint's source stage is corpus-specific.** An `enforced` ingest stage marks the source
*slot* as required, not that class — a blueprint derived from one dataset names that dataset's
source, so re-pick the source from the `ingest` cards for the corpus in hand and keep the rest of
the topology. A dataset-specific source pointed at a corpus that merely resembles it does not
refuse; it emits a partial manifest under that dataset's assumptions.

**Prune to the request.** A stage earns its place only if it serves a stated goal. Every
filter DROPS data, so each filter must trace to a criterion the user actually asked for
(e.g. "clean" -> a noise gate; "high-quality" -> a MOS gate) — do NOT add a filter for a
dimension the user never mentioned (bandwidth, VAD, an extra quality gate). A blueprint is
a menu, not a mandate: keep the `enforced` stages plus only the `advisory` ones that match
the goal, and drop the rest — never carry a template's (or composite's) full stage set
wholesale. Rule of thumb: if a filter has no matching acceptance criterion, it should not
be in the recipe.

This applies to **preprocess**, not just filters: add mono/resample only if a downstream
stage actually needs that form — UTMOS/SIGMOS/SQUIM accept any channel count and resample
internally, so they do NOT require an upstream mono/resample. And avoid **no-op** stages: a
mono/resample with `keep_waveform_in_task=false` AND `write_to_disk=false` (or `write_to_disk`
without `update_audio_filepath`) while the next stage reads from file **converts the audio and
then discards it** — downstream still scores the ORIGINAL files. Either make it effective
(`write_to_disk=true` + `update_audio_filepath`, or keep the waveform and have the next stage
read it via `input_residency`) or drop the stage.

## Reuse-aware pipeline construction

Correct semantics and the user's request come first; never add a stage solely for reuse.
Apply the workflow's curation mode only when both shapes are correct:

| Decision | `refine_later` — Easy to refine later | `fast_first` — Fastest first run |
|---|---|---|
| Residency | Prefer durable file-backed boundaries and serializable path/offset metadata. | Prefer adjacent in-memory handoffs when contracts prove compatibility. |
| Annotate/filter | Prefer exact card-declared annotate then selector forms. | Prefer a native filter and early row reduction. |
| Checkpoint | Consider at most one worthwhile metadata checkpoint; never intermediate audio solely for reuse. | Minimize intermediate I/O, but still obey the recipe-specific checkpoint decision gate. |
| Ordering | Keep reusable measurement above its exact destructive selector; do not delay another useful filter enough to cause excessive model work. | Put cheap/native filters early when that reduces expensive downstream work without changing meaning. |
| First run | May spend extra metadata I/O and storage. | Usually minimizes first-run latency and storage. |
| Future tuning | Threshold-only changes can reuse retained annotations when the core proves the boundary. | Later threshold changes may rerun model work. |

This matrix is soft preference guidance, never a correctness constraint or hard
bound. If the preferred form is illegal, unavailable, semantically different, or
not worthwhile, use the best correct authored shape and explain the deviation.
No mode authorizes a pointless checkpoint, a delayed useful filter that causes
excessive model work, or more than one metadata checkpoint for this preference.

Under `refine_later`, inspect each finalist's full decision card while authoring.
Use annotate-before-filter only when its `decision` contract proves exact
separation and retaining the annotation is worthwhile. UTMOS task separation
uses `PreserveByValueStage`; explicit segment separation uses one
`PreserveByValueConditionsStage` condition. That stage supports generic
`condition_logic='or'` pipelines, but card-declared UTMOS/SIGMOS exact reuse
always sets `condition_logic='and'`; never suggest OR as a native-filter
equivalent. SIGMOS requires one compound AND selector containing every enabled
threshold with its configured score key; never reduce it to OVRL. For segment
mode, set the selector's generic `items_key` exactly to the producer's
configured `segments_key`, keep missing-score `drop` and
`drop_parent_if_empty=true`, and persist only JSON-serializable path/offset
metadata. Prefer explicit `mode=task` or
`mode=segments` over `mode=auto` only when scope is mechanically proven. If
scope is not proven, keep the best correct mode and explain why the
refine-later form was unavailable; never invent task, segment, nested, tensor,
multi-scope, or corpus-level equivalence.

Keep cheap, recursive-nested, missing-sensitive, private-metadata-dependent and
corpus-dependent filters native. Under `fast_first`, native filtering remains a
preference, not permission to bypass the existing `plan-checkpoint` decision
gate. A candidate returned by `plan-checkpoint` is still a new recipe: validate
and semantically critique it again before authoritative smoke. When the core
marks an expensive candidate recommended, smoke/run require either that
candidate or the exact baseline recipe returned after
`plan-checkpoint --choice baseline`.

For `refine_later`, read `validate.planning_advisories` during semantic review.
It is non-blocking evidence that the current recipe remains valid while an exact
card-declared reusable alternative exists. Never treat it as an Issue, rewrite
the recipe automatically, or force a checkpoint. For `fast_first` or an absent
preference, no preference advisory should appear.

**Measuring, selecting and converting are different requests.** A stage with an `action` param
does exactly one of them per instance, and picking the wrong one is silent: converting a corpus
the user wanted narrowed keeps rows they meant to exclude, and narrowing one they wanted
converted throws away most of it. So decide from the wording — a property stated as a
requirement of the OUTPUT ("make it mono", "at 16 kHz") is a conversion, while one stated as a
property of the INPUT they want kept ("the mono recordings", "only the 48 kHz ones") is a
selection, and a request to find out ("how many channels do these have?") is neither. When the
wording carries both readings, ask rather than guess: `annotate` first is the cheap way to show
the user what their corpus actually holds before either choice is made. Stages that can drop
rows say so in `cardinality_options` even when the action you configured does not.

## Resolve outcomes to parameters (never expose internal numbers)

Do not hand-pick thresholds. Map the user's **outcome** to a concrete param with
`resolve`, which reads the card's `metrics` anchors/presets:

```bash
python -m nemo_curator.audio_agent resolve --stage UTMOSFilterStage --label studio
# -> params {mos_threshold: 4.0}   (+ an auditable strategy trail)
```

- `--label <outcome>` (e.g. `studio`, `wideband`, `transcription_grade`) maps via
  the card anchors to the stage's own threshold param, or — for an annotator like
  WER — to a `PreserveByValueStage` filter (`filter_stage` in the result).
- `--use-case <preset>` applies a named card preset; `--explicit '{"p": v}'` uses a
  value the user gave.
- If it returns `asks` (unknown label, or a **relative** objective like "best 20%"
  which needs data / Path B), ask the user a plain-language question or get an
  explicit bar — never invent the number.

Apply the returned `params` to the stage (and insert any `filter_stage`), and keep
the `strategy` trail for the plan/report (it records *why* each value was chosen).

**`resources` is a knob too** — set it from the card's `resource` facts, don't leave the
default. If a stage's card is `bound: gpu` / `gpu_optional: false`, set
`resources=Resources(gpus=1)` (size VRAM from the card's `gpu_mem_gb`); a GPU-*optional*
stage may stay on CPU or opt into GPU for throughput. A GPU-required stage left at its CPU
default runs on CPU (very slow) and can over-parallelize into many model-loading actors.
