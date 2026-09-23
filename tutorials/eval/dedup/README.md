# Fuzzy dedup evaluation example

This example builds an evaluation dataset from a completed fuzzy-deduplication
run and uses an LLM judge to check whether fuzzy dedup's keep/remove/group
decisions are actually correct. It does not change fuzzy dedup itself; it
audits the decisions fuzzy dedup already made.

This is an illustrative worked example, not a calibrated production
evaluation. The judging rubric, model, and per-field thresholds in
`judge_config/` are starting points -- re-validate them on your own data
before trusting the results at scale. The LLM judge is an automated opinion,
not ground truth: treat disagreements between the judge and fuzzy dedup as
things to look into, not proven errors.

## Pipeline

```text
1. 1_data_prep.py           -> download & extract a Common Crawl sample with jusText
2. 2_run_fuzzy_dedup.py     -> FuzzyDeduplicationWorkflow (identification only)
3. 3_build_pair_dataset.py  -> labeled document pairs from fuzzy dedup's decisions
4. 4_span_alignment.py      -> add span-alignment evidence (semantic_diff/truncated) to each pair
5. 5_run_llm_judge.py       -> LLMJudgeWorkflow judges each pair
6. 6_analyze_results.py     -> summarize judge verdicts vs. fuzzy dedup's decisions
```

Step 1 downloads a small Common Crawl sample and extracts HTML content with
jusText. Common Crawl pages are full of real exact- and near-duplicate
boilerplate -- cookie notices, legal disclaimers, navigation chrome, mirrored
or syndicated articles -- so fuzzy dedup and the LLM judge have real
decisions to make, and it's the same class of content the judge rubric in
`judge_config/` is written for.

**Prerequisites:**

- Step 1 needs network access to Common Crawl.
- Steps 2-3 need the RAPIDS/cuGraph GPU stages `FuzzyDeduplicationWorkflow`
  normally needs (MinHash/LSH/connected components).
- Step 4 is CPU-only (`difflib`-based span alignment run through Ray Data).
- Step 5 needs GPU(s) to serve a local judge model through Dynamo --
  `LLMJudgeWorkflow` only supports locally served models, there's no
  hosted-inference-API backend.

## Quick start

Each script runs standalone -- see the docstring at the top of each file for
a full example invocation.

```bash
# Step 1: download & extract a Common Crawl sample.
python tutorials/eval/dedup/1_data_prep.py \
  --download-dir output/dedup_eval/cc_warcs \
  --output-path output/dedup_eval/raw_corpus

# Step 2: fuzzy dedup identification.
python tutorials/eval/dedup/2_run_fuzzy_dedup.py \
  --input-path output/dedup_eval/raw_corpus \
  --input-filetype jsonl \
  --cache-dir output/dedup_eval/fuzzy_cache \
  --output-dir output/dedup_eval/fuzzy_ids

# Step 3: build the labeled pair dataset -- one strategy per run, one
# --output-path per strategy (see "Pairing strategies" below).
# --input-path/--input-filetype/--input-blocksize MUST match step 2 exactly
# -- see "Keep step 2 and step 3 inputs matched" below.
# These are raw pairs -- no semantic_diff/truncated yet, see step 4.
python tutorials/eval/dedup/3_build_pair_dataset.py \
  --input-path output/dedup_eval/raw_corpus \
  --input-filetype jsonl \
  --cache-dir output/dedup_eval/fuzzy_cache \
  --fuzzy-output-dir output/dedup_eval/fuzzy_ids \
  --output-path output/dedup_eval/keeper_removed_pairs_raw \
  --pair-strategy keeper_removed

# Step 4: add span-alignment evidence (semantic_diff/truncated) to each pair --
# see "Span alignment and truncation" below.
python tutorials/eval/dedup/4_span_alignment.py \
  --input-path output/dedup_eval/keeper_removed_pairs_raw \
  --output-path output/dedup_eval/keeper_removed_pairs

# Step 5: judge each pair. Edit judge_config/fuzzy_pair_judge.yaml first --
# set models[0].model to a local model path or HF repo id, and size
# num_replicas/tensor_parallel_size to your GPUs (bundled default: 1 GPU).
# --input-path accepts a glob, so point it at one strategy's directory, or at
# several (e.g. output/dedup_eval/*_pairs) to judge them together.
python tutorials/eval/dedup/5_run_llm_judge.py \
  --input-path output/dedup_eval/keeper_removed_pairs \
  --output-path output/dedup_eval/judged_pairs

# Step 6: summarize, and write disagreeing pairs out for manual review.
python tutorials/eval/dedup/6_analyze_results.py \
  --judge-output-path output/dedup_eval/judged_pairs \
  --disagreements-output output/dedup_eval/disagreements.jsonl
```

## Pairing strategies

`3_build_pair_dataset.py --pair-strategy` takes exactly one of the following.
Run the script once per strategy you want, each with its own `--output-path`
(they all read the same step-2 outputs, so re-running is cheap) -- keeping
one strategy per directory avoids ambiguity about which pairs came from
which strategy once they reach the judge.

- `keeper_removed`: pairs the document fuzzy dedup kept in a group against
  each document it marked for removal from that group. Directly answers
  "was this removal decision correct?" with one pair per removed document.
- `all_pairwise`: every pair of distinct documents within a group, including
  removed-vs-removed pairs `keeper_removed` skips. Costs more LLM calls
  (`C(n, 2)` per group of size `n`) but catches within-group grouping errors
  that keeper-vs-removed comparisons alone would miss.
- `cross_group_sample` (size it with `--cross-group-samples N`): samples `N`
  pairs of documents from *different* groups (including singleton documents
  with no group). These are pairs fuzzy dedup did **not** treat as
  duplicates -- a check for missed duplicates / false negatives, mirroring
  what `keeper_removed`/`all_pairwise` check for false positives.

Each pair is stamped with `pair_type` and `expected_duplicate`: `true` for
`keeper_removed`/`all_pairwise` (fuzzy dedup grouped them together) and
`false` for `cross_group_sample` (fuzzy dedup did not). `6_analyze_results.py`
compares the judge's verdict against this label.

## Keep step 2 and step 3 inputs matched

`3_build_pair_dataset.py` recovers each document's text by re-reading the
*original* input corpus and replaying `fuzzy_id_generator.json` (written by
`2_run_fuzzy_dedup.py`) to reassign the exact same `_curator_dedup_id` values
fuzzy dedup used. That replay is keyed by a hash of the exact set of files
grouped into each reader batch, which depends on `--input-path`,
`--input-filetype`, and `--input-blocksize`. Passing different values for any
of these to `3_build_pair_dataset.py` than you passed to `2_run_fuzzy_dedup.py`
raises a `KeyError` during id reassignment -- it fails loudly rather than
silently pairing the wrong text with an id. Keep the flags above matched
between steps 2 and 3.

The re-read result is cached as sharded Parquet under
`--cache-dir/CorpusWithIds/` and reused across `--pair-strategy` re-runs
against the same `--cache-dir` (delete `CorpusWithIds/` to force a re-read).
The cache records which `--input-path`/`--input-filetype`/`--input-blocksize`
built it and raises if a later run requests different ones, instead of
silently reusing a stale cache built from different inputs.

## Output shape

`pairs/*.jsonl` (step 3): `pair_id`, `pair_type`, `expected_duplicate`,
`group_id_a`/`group_id_b`, `id_a`/`id_b` (`_curator_dedup_id` values),
`doc_id_a`/`doc_id_b` (original `url` field), `text_a`/`text_b` (full,
untruncated -- kept for manual review; the judge is not shown these
directly). No `semantic_diff`/`truncated` yet -- step 4 adds those (see
below). Filenames and shard count differ by `--pair-strategy` (see
"Scaling" below): `keeper_removed`/`all_pairwise` are written by Ray Data's
own distributed `Dataset.write_json()`, so file count follows Ray's
partitioning, not `--pairs-per-file`; `cross_group_sample` still writes
`part_*.jsonl` files sized by `--pairs-per-file`, since step 4 can
parallelize across whatever files land here either way.

### Span alignment and truncation

`4_span_alignment.py`'s `build_semantic_diff()` aligns each pair's visible
text (the same `--max-visible-chars`-truncated view the judge is shown,
currently 6000 characters per side -- keep this in sync with
`judge_config/fuzzy_pair_judge.yaml`'s `max_model_len`) into `SHARED`/
`A_ONLY`/`B_ONLY` spans with stable IDs (`S001`, `A001`, `B001`, ...), using
`difflib.SequenceMatcher` over normalized word/punctuation tokens. It runs as
three Curator `ProcessingStage`s -- `TokenizerStage` (tokenizes `text_a`/
`text_b` once), `SpanAlignmentStage` (diffs the token streams into segments),
`SpanChunkingStage` (splits any segment longer than `_MAX_SPAN_CHUNK_CHARS`
characters into multiple spans, re-slicing `TokenizerStage`'s token lists
instead of re-tokenizing, and assembles the final packet) -- so alignment
happens per reader task via Ray Data -- one raw pairs file from step 3
becomes one task by default (`--files-per-partition 1`) -- instead of a
single-process Python loop over every pair. No single span is ever more than
`_MAX_SPAN_CHUNK_CHARS` characters, so one long run of unchanged text or one
long difference doesn't become a single undifferentiated block. Side-only
spans are additionally padded, at their outer edges only (not between
chunks), with a little neighboring shared text (`_SPAN_CONTEXT_CHARS`) so a
short changed value isn't shown in isolation -- e.g. for

```text
A: Applicable to Model X100
B: Applicable to Model X200
```

the packet includes an `A_ONLY`/`B_ONLY` pair covering "Applicable to Model
X100"/"...X200" rather than just the bare `X100`/`X200` tokens, so the judge
sees what the changed value actually refers to. This is stored as the
`semantic_diff` field (`status`, `truncated`, `truncated_a`, `truncated_b`,
`span_counts`, `spans`) and is what `judge_config/pair.jinja` renders to the
judge -- the judge never sees raw `text_a`/`text_b` directly. `status` is
`INCOMPLETE_LIMIT` instead of `COMPLETE` if a pair produces more than
`_MAX_SPANS_PER_KIND` spans of one kind; `system.jinja` treats an incomplete
packet as unresolved rather than trusting a silently-clipped span list.

Truncation is explicit rather than silent: if either document exceeded the
visible-character limit, `pair.jinja` renders a `<truncation_notice>` telling
the judge a decisive difference could exist only in the cut-off portion, and
`system.jinja`'s deterministic policy requires `unresolved`/low confidence
rather than a conclusive verdict built only on "no difference was visible."
`6_analyze_results.py` excludes truncated pairs from the headline
disagreement rate for the same reason, reporting them as `num_truncated`
instead.

`judged_pairs/*.jsonl` (step 5): the same fields (now including
`semantic_diff`/`truncated` from step 4) plus a `pair_semantic_judgment`
column holding one nested `{"score": ..., "reasoning": ...}` result per
rubric field -- `span_content_profile_a`/`span_content_profile_b`,
`span_shared_basis`, `span_a_delta`/`span_b_delta`, `span_hard_conflict`,
`span_translation_status`, `a_can_replace_b`/`b_can_replace_a`,
`relation_type`, `material_difference`, `primary_material_difference`,
`dominant_overlap_source`, `primary_risk_factor`, `confidence_tier`. See
`judge_config/fuzzy_pair_judge.yaml` for the full rubric and
`nemo_curator/eval/llm_judge/LLM_JUDGE_CONFIG_SKILL.md` for how to change it.

`6_analyze_results.py` (step 6) buckets `relation_type` into a coarse
duplicate/not_duplicate/unresolved verdict (`exact`/`canonical_exact`/
`near_surface`/`containment` -> duplicate; `version_related`/
`related_non_duplicate`/`unrelated` -> not_duplicate -- see the bucketing
rationale in the script's docstring) and prints, per `pair_type`: relation
counts and a disagreement rate against `expected_duplicate` -- for
`keeper_removed`/`all_pairwise`, the share bucketed `not_duplicate` despite
fuzzy dedup grouping them (candidate wrong-grouping rate); for
`cross_group_sample`, the share bucketed `duplicate` despite fuzzy dedup
keeping them apart (candidate missed-duplicate rate). These are diagnostics
from one unvalidated judge, not accuracy numbers -- for anything more than a
coarse rate, read `relation_type`/`material_difference`/
`primary_material_difference` directly rather than the bucketed verdict.

## Scaling `3_build_pair_dataset.py`

- `keeper_removed`/`all_pairwise` build pairs as a distributed Ray Data
  pipeline (`build_grouped_pairs_distributed()`): `ray.data.read_parquet()`
  streams `CorpusWithIds/` across the cluster, `map_batches()` tags each
  document with its duplicate-group id from a broadcast lookup, and
  `groupby(...).map_groups(...)` builds pairs one group at a time, distributed
  -- no single task ever loads every duplicate group's documents into one
  process, so this scales with cluster size rather than driver memory.
- `all_pairwise` still generates `C(n, 2)` pairs per duplicate group, and real
  web corpora routinely produce one oversized cluster (cookie banners, empty
  pages, templated legal boilerplate). `--max-group-size` (default 200) skips
  and warns about any group larger than that instead of letting one cluster
  generate an unbounded number of pairs -- raise it deliberately if you want
  larger groups included.
- `cross_group_sample` draws a bounded, approximately-uniform sample instead
  of materializing every document -- each Parquet part file contributes a
  subsample sized by its share of the total row count (from Parquet
  metadata, without reading the data first), capped around
  `max(cross_group_samples * 10, 2000)` documents regardless of corpus size.
  It stays driver-side (`build_cross_group_sample_pairs()`): it compares
  documents *across* groups, so there's no per-group unit of work to
  distribute, and its input is already bounded independent of corpus size --
  it isn't the bottleneck the Ray Data rewrite above targets.

What's still loaded fully into memory for every strategy: `group_labels_df`
(`ConnectedComponentsStage`'s output) and `removed_ids` (`FuzzyDuplicateIds`)
-- both scoped to documents that are part of *some* duplicate group, not the
full corpus, so they're bounded by your corpus's duplicate rate rather than
its total size. This is what `map_batches()`'s broadcast lookup above is
built from, and it's also `keeper_removed`/`all_pairwise`'s only remaining
non-distributed step. If your corpus is pathological enough that most of it
ends up in one enormous duplicate group, that assumption breaks down -- at
that point, sample duplicate groups before loading.

`4_span_alignment.py` doesn't share this limitation either: it's a `Pipeline`
of `JsonlReader -> TokenizerStage -> SpanAlignmentStage -> SpanChunkingStage ->
JsonlWriter` stages, so span alignment itself already runs distributed across
reader tasks via Ray Data rather than loading pairs into driver memory -- the
per-partition granularity is set by step 4's `--files-per-partition`.

## Files

- `1_data_prep.py` -- downloads and extracts a Common Crawl sample with jusText.
- `2_run_fuzzy_dedup.py` -- `FuzzyDeduplicationWorkflow(perform_removal=False)`.
- `3_build_pair_dataset.py` -- re-reads the corpus with replayed ids (cached
  under `--cache-dir/CorpusWithIds/`), joins duplicate-group labels and
  removal decisions, emits raw labeled pairs (no `semantic_diff`/`truncated` yet).
- `4_span_alignment.py` -- `build_semantic_diff()` and the `TokenizerStage`/
  `SpanAlignmentStage`/`SpanChunkingStage` Curator stages that add
  `semantic_diff`/`truncated` to each pair, run through a `JsonlReader ->
  TokenizerStage -> SpanAlignmentStage -> SpanChunkingStage -> JsonlWriter`
  `Pipeline`.
- `judge_config/fuzzy_pair_judge.yaml`, `judge_config/system.jinja`,
  `judge_config/pair.jinja` -- the LLM judge config for step 5, a
  semantic-retention rubric.
- `5_run_llm_judge.py` -- runs `LLMJudgeWorkflow` over the span-aligned pairs
  part files with this example's judge config.
- `6_analyze_results.py` -- summarizes step 5's output.
