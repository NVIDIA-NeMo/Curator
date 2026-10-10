# Fuzzy dedup evaluation example

Fuzzy deduplication groups near-duplicate documents and picks one "keeper" per group, removing the rest. This example checks whether those decisions were actually correct, by having an LLM judge look at the pairs of documents fuzzy dedup grouped together (and some it didn't) and render its own verdict on whether they're really duplicates. It doesn't change fuzzy dedup itself -- it's a way to audit and spot-check the decisions a completed fuzzy dedup run already made.

This workflow has been tested at the 100 million document scale, and the judge config in `judge_config/` was tuned on 8xH100 nodes. The LLM judge is an automated opinion, not ground truth: treat disagreements between the judge and fuzzy dedup as things to look into, not proven errors.

## Pipeline

The example is six standalone scripts, run in order. Each also has a full example invocation and flag reference in its own docstring.

**1. `1_data_prep.py`** downloads and extracts a Common Crawl sample with jusText. This just gets you a realistic corpus to run fuzzy dedup against -- Common Crawl pages are full of real near-duplicate content (cookie notices, legal boilerplate, syndicated articles), which is exactly the kind of thing this eval is meant to check.

**2. `2_run_fuzzy_dedup.py`** runs `FuzzyDeduplicationWorkflow` in identification-only mode: it groups near-duplicate documents and decides which one to keep in each group, but doesn't actually delete anything. Its output -- duplicate groups and removal decisions -- is what the rest of the pipeline evaluates.

**3. `3_build_pair_dataset.py`** turns fuzzy dedup's output into a dataset of document *pairs*, since that's the unit the LLM judge reasons about ("are these two documents duplicates?") rather than whole groups. Pick one of three `--pair-strategy` values depending on what you want to check:

- `keeper_removed` -- pairs the document fuzzy dedup kept with each document it removed from that group. This is the core check: "was this removal decision correct?"
- `all_pairwise` -- pairs every document in a group with every other document in that group, including removed-vs-removed pairs `keeper_removed` skips. More expensive, but also catches cases where fuzzy dedup grouped two documents together that shouldn't have been in the same group at all. Groups larger than `--max-group-size` (default 200) are not expanded to every pair; instead `--oversized-group-pairs` (default 50) random pairs are sampled from each.
- `cross_group_sample` -- samples pairs of documents from *different* groups, i.e. pairs fuzzy dedup did **not** treat as duplicates. This checks the opposite failure mode: documents that should have been grouped together but weren't (missed duplicates).

Run the script once per strategy you want to evaluate, each with its own `--output-path`. Every pair is labeled `expected_duplicate=true` (for `keeper_removed`/`all_pairwise`) or `false` (for `cross_group_sample`) -- this is fuzzy dedup's original decision, and step 6 later checks the judge's verdict against it.

**4. `4_span_alignment.py`** is where the pairs get prepped for judging. The judge is only shown a limited amount of text per document (6000 characters, matching the judge's context window), and rather than hand it two long, mostly-identical blobs of text and hope it spots the difference, this step runs a text diff (`difflib`) between each pair and breaks the visible text into labeled spans: `SHARED` (text common to both), `A_ONLY` (only in document A), and `B_ONLY` (only in document B). That span breakdown -- not the raw text -- is what actually gets rendered into the judge's prompt, so the judge's job becomes "look at what's different" instead of "read two documents and find what's different." If a document had to be cut off to fit the 6000-character limit, the pair is flagged `truncated`, since a difference could be hiding in the part the judge never saw. Keep the judge's `max_model_len` in mind when tuning this: the rendered span packet (which repeats shared text and pads each difference with context), the system prompt, and `max_tokens` must all fit in it, and non-Latin or code-heavy text uses far more tokens per character. If the server rejects prompts as exceeding the context length, lower `--max-visible-chars` (and, if needed, `--span-context-chars`, `--max-span-chunk-chars`, or `--max-spans-per-kind`) and rerun this step.

**5. `5_run_llm_judge.py`** is the actual evaluation step, and the main point of this example: it runs `LLMJudgeWorkflow` to have a locally-served LLM judge each pair against the rubric in `judge_config/fuzzy_pair_judge.yaml`. For each pair, the judge scores several rubric fields (e.g. `relation_type` -- is this pair `exact`, `near_surface`, `containment`, `version_related`, `unrelated`, etc. -- plus supporting fields like `material_difference` and `confidence_tier`) and returns its reasoning for each. `relation_type` is the main verdict step 6 evaluates. Before running this step, edit `judge_config/fuzzy_pair_judge.yaml`: set `models[0].model` to a local model path or HF repo id, and size `num_replicas`/`tensor_parallel_size` to the GPUs you have available. `LLMJudgeWorkflow` only supports locally-served models -- there's no hosted-inference-API backend.

**6. `6_analyze_results.py`** is where you actually find out how fuzzy dedup did. It takes the judge's `relation_type` verdicts and buckets them into duplicate / not_duplicate / unresolved, then compares that against each pair's `expected_duplicate` label from step 3 to compute a disagreement rate per pairing strategy -- e.g. what fraction of `keeper_removed` pairs the judge thinks *aren't* actually duplicates (candidate wrong removals), or what fraction of `cross_group_sample` pairs the judge thinks *are* duplicates (candidate missed duplicates). It writes every disagreeing pair to a JSONL file for manual review, since a disagreement is a signal to look at the pair yourself, not a confirmed fuzzy-dedup error.

Before bucketing, this step also overrides the judge's `relation_type` in two cases where it's checkable independently of the judge's opinion, rather than trusting a wrong call: (1) if step 4's span packet is untruncated and `COMPLETE` with zero `A_ONLY`/`B_ONLY` spans, the visible text on both sides is objectively identical, so `relation_type` is forced to `exact` even if the judge said something else (e.g. `near_surface`); (2) if the judge returned `containment` for a pair that violates one of containment's own rubric preconditions (a content-profile mismatch between the two sides, both sides being `non_main_only`, or no verified shared basis), it's reclassified to `related_non_duplicate` instead, since an uncorrected wrong `containment` call would otherwise inflate the duplicate count. Both corrections are logged with a count, and corrected rows are flagged in the disagreements output (`relation_type_corrected` / `relation_type_containment_corrected`) so you can see exactly which ones were overridden.

**Prerequisites:** step 1 needs network access to Common Crawl; steps 2-3 need the GPU stages (RAPIDS/cuGraph) that `FuzzyDeduplicationWorkflow` normally uses for MinHash/LSH/connected components; step 4 is CPU-only; step 5 needs GPU(s) to serve the judge model.

## Quick start

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

# Step 3: build the labeled pair dataset. Choose one --pair-strategy (see
# above), and make sure --input-path/--input-filetype/--input-blocksize
# match step 2 exactly.
python tutorials/eval/dedup/3_build_pair_dataset.py \
  --input-path output/dedup_eval/raw_corpus \
  --input-filetype jsonl \
  --cache-dir output/dedup_eval/fuzzy_cache \
  --fuzzy-output-dir output/dedup_eval/fuzzy_ids \
  --output-path output/dedup_eval/keeper_removed_pairs_raw \
  --pair-strategy keeper_removed

# Step 4: add span-alignment evidence to each pair.
python tutorials/eval/dedup/4_span_alignment.py \
  --input-path output/dedup_eval/keeper_removed_pairs_raw \
  --output-path output/dedup_eval/keeper_removed_pairs

# Step 5: judge each pair. First edit judge_config/fuzzy_pair_judge.yaml --
# set models[0].model to a local model path or HF repo id, and size
# num_replicas/tensor_parallel_size to your GPUs.
python tutorials/eval/dedup/5_run_llm_judge.py \
  --input-path output/dedup_eval/keeper_removed_pairs \
  --output-path output/dedup_eval/judged_pairs

# Step 6: summarize the results, and write disagreeing pairs out for
# manual review.
python tutorials/eval/dedup/6_analyze_results.py \
  --judge-output-path output/dedup_eval/judged_pairs \
  --disagreements-output output/dedup_eval/disagreements.jsonl
```

## Files

- `1_data_prep.py` -- downloads and extracts a Common Crawl sample with jusText.
- `2_run_fuzzy_dedup.py` -- runs `FuzzyDeduplicationWorkflow(perform_removal=False)`.
- `3_build_pair_dataset.py` -- builds labeled document pairs from fuzzy dedup's groups and removal decisions, using the per-strategy code in `build_pairs/`.
- `4_span_alignment.py` -- diffs each pair's visible text into `SHARED`/`A_ONLY`/`B_ONLY` spans for the judge.
- `judge_config/fuzzy_pair_judge.yaml`, `judge_config/system.jinja`, `judge_config/pair.jinja` -- the LLM judge config for step 5, a semantic-retention rubric.
- `5_run_llm_judge.py` -- runs `LLMJudgeWorkflow` over the span-aligned pairs using this example's judge config.
- `6_analyze_results.py` -- buckets the judge's verdicts and reports disagreement rates against fuzzy dedup's decisions.
