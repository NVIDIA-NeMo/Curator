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
4. 4_run_llm_judge.py       -> LLMJudgeWorkflow judges each pair
5. 5_analyze_results.py     -> summarize judge verdicts vs. fuzzy dedup's decisions
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
- Step 4 needs GPU(s) to serve a local judge model through Dynamo --
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
python tutorials/eval/dedup/3_build_pair_dataset.py \
  --input-path output/dedup_eval/raw_corpus \
  --input-filetype jsonl \
  --cache-dir output/dedup_eval/fuzzy_cache \
  --fuzzy-output-dir output/dedup_eval/fuzzy_ids \
  --output-path output/dedup_eval/keeper_removed_pairs \
  --pair-strategy keeper_removed

# Step 4: judge each pair. Edit judge_config/fuzzy_pair_judge.yaml first --
# set models[0].model to a local model path or HF repo id, and size
# num_replicas/tensor_parallel_size to your GPUs (bundled default: 1 GPU).
# --input-path accepts a glob, so point it at one strategy's directory, or at
# several (e.g. output/dedup_eval/*_pairs) to judge them together.
python tutorials/eval/dedup/4_run_llm_judge.py \
  --input-path output/dedup_eval/keeper_removed_pairs \
  --output-path output/dedup_eval/judged_pairs

# Step 5: summarize, and write disagreeing pairs out for manual review.
python tutorials/eval/dedup/5_analyze_results.py \
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
`false` for `cross_group_sample` (fuzzy dedup did not). `5_analyze_results.py`
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

`pairs/part_*.jsonl` (step 3, sharded to `--pairs-per-file` pairs per file so
step 4 can parallelize across reader tasks): `pair_id`, `pair_type`,
`expected_duplicate`, `group_id_a`/`group_id_b`, `id_a`/`id_b`
(`_curator_dedup_id` values), `doc_id_a`/`doc_id_b` (original `url` field),
`text_a`/`text_b`.

`judged_pairs/*.jsonl` (step 4): the same fields plus a
`pair_semantic_judgment` column holding one nested `{"score": ..., "reasoning": ...}`
result per rubric field -- `span_content_profile_a`/`span_content_profile_b`,
`span_shared_basis`, `span_a_delta`/`span_b_delta`, `span_hard_conflict`,
`span_translation_status`, `a_can_replace_b`/`b_can_replace_a`,
`relation_type`, `material_difference`, `primary_material_difference`,
`dominant_overlap_source`, `primary_risk_factor`, `confidence_tier`. See
`judge_config/fuzzy_pair_judge.yaml` for the full rubric and
`nemo_curator/eval/llm_judge/LLM_JUDGE_CONFIG_SKILL.md` for how to change it.

`5_analyze_results.py` (step 5) buckets `relation_type` into a coarse
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

Building on the `CorpusWithIds/` cache described above:

- `keeper_removed`/`all_pairwise` only load documents that belong to a
  duplicate group, streaming through `CorpusWithIds/` rather than loading the
  whole corpus at once -- bounded by how many documents fuzzy dedup actually
  grouped, not by total corpus size.
- `cross_group_sample` draws a bounded, approximately-uniform sample instead
  of materializing every document -- each Parquet part file contributes a
  subsample sized by its share of the total row count (from Parquet
  metadata, without reading the data first), capped around
  `max(cross_group_samples * 10, 2000)` documents regardless of corpus size.
- `all_pairwise` generates `C(n, 2)` pairs per duplicate group, and real web
  corpora routinely produce one oversized cluster (cookie banners, empty
  pages, templated legal boilerplate). `--max-group-size` (default 200) skips
  and warns about any group larger than that instead of letting one cluster
  generate an unbounded number of pairs -- raise it deliberately if you want
  larger groups included.

What's still loaded fully into memory: `group_labels_df`
(`ConnectedComponentsStage`'s output) and `removed_ids`
(`FuzzyDuplicateIds`) -- both scoped to documents that are part of *some*
duplicate group, not the full corpus, so they're bounded by your corpus's
duplicate rate rather than its total size. If your corpus is pathological
enough that most of it ends up in one enormous duplicate group, that
assumption breaks down -- at that point, sample duplicate groups before
loading, or move the join/pairing logic into a distributed Curator stage.

## Files

- `1_data_prep.py` -- downloads and extracts a Common Crawl sample with jusText.
- `2_run_fuzzy_dedup.py` -- `FuzzyDeduplicationWorkflow(perform_removal=False)`.
- `3_build_pair_dataset.py` -- re-reads the corpus with replayed ids (cached
  under `--cache-dir/CorpusWithIds/`), joins duplicate-group labels and
  removal decisions, emits labeled pairs.
- `judge_config/fuzzy_pair_judge.yaml`, `judge_config/system.jinja`,
  `judge_config/pair.jinja` -- the LLM judge config for step 4, a
  semantic-retention rubric.
- `4_run_llm_judge.py` -- runs `LLMJudgeWorkflow` over the pairs part files
  with this example's judge config.
- `5_analyze_results.py` -- summarizes step 4's output.
