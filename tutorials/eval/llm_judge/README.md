# LLM judge runner

Use this example to add LLM-based evaluations to JSONL or Parquet records. The YAML configuration defines the served judge model, Jinja prompt files, rubric scores, and optional output filters. The runner starts a local Curator Dynamo/vLLM server, executes NeMo Data Designer judge columns, and writes the original records with the judge results added.

The included example compares jusText and Trafilatura web-text extractions. The same runner can judge parser output, extraction quality, or any task whose inputs can be rendered into a Jinja prompt.

## Quick start

Start by copying and editing the files in `cc_extract_example/`. The YAML refers to adjacent Jinja files by relative path, so keep them together.

This is a minimal integration example, not a calibrated production evaluation. Its prompts, rubrics, model settings, thresholds, and concurrency values are illustrative and have not been optimized. Validate and adapt them on a manually reviewed sample before relying on results.

As bundled, [text_extraction_qwen_judge.yaml](cc_extract_example/text_extraction_qwen_judge.yaml) uses **4 GPUs** (`dynamo_model.num_replicas: 4` × `engine_kwargs.tensor_parallel_size: 1`, one model). If you have a different GPU count, edit `num_replicas` (more/fewer independent replicas) and/or `tensor_parallel_size` (GPUs per replica) to match before running.

1. Set `models[0].model` in [text_extraction_qwen_judge.yaml](cc_extract_example/text_extraction_qwen_judge.yaml) to a model identifier or local model path.
2. Update [text_extraction_prompt.jinja](cc_extract_example/text_extraction_prompt.jinja) with the field names from your input rows.
3. Define the rubric outputs under each judge's `scores:` list.
4. Run a small input first.

```bash
python tutorials/eval/llm_judge/run_llm_judge.py \
  --judge-config tutorials/eval/llm_judge/cc_extract_example/text_extraction_qwen_judge.yaml \
  --input-path data/cc_extractions \
  --input-format jsonl \
  --output-path data/qwen_judgements \
  --output-format jsonl
```

The bundled [text_extraction_qwen_gemma_judges.yaml](cc_extract_example/text_extraction_qwen_gemma_judges.yaml) runs the same extraction rubrics with both Qwen and Gemma, and as bundled uses **8 GPUs** (two models, each `num_replicas: 4` × `tensor_parallel_size: 1`). Use it when you want to compare model agreement; update the model paths, `num_replicas`/`tensor_parallel_size` per model, and other serving settings for your hardware before running it.

Use `--checkpoint-path output/judge_checkpoint` to write Curator checkpoint metadata to a durable location. It is useful for normal pipeline recovery, but you should still inspect input and output counts after a run.

`run_llm_judge.py` is a thin CLI over `LLMJudgeWorkflow` (`eval/llm_judge/llm_judge_workflow.py`, importable as `from eval.llm_judge import LLMJudgeWorkflow`). Each `--flag` above maps to a same-named constructor argument, so call it directly when you want to run a judge pass from your own script instead of the CLI (for example, as one step alongside other Curator workflows):

```python
from nemo_curator.core.client import RayClient

from eval.llm_judge import LLMJudgeWorkflow

workflow = LLMJudgeWorkflow(
    judge_config="tutorials/eval/llm_judge/cc_extract_example/text_extraction_qwen_judge.yaml",
    input_path="data/cc_extractions",
    input_format="jsonl",
    output_path="data/qwen_judgements",
    output_format="jsonl",
)

ray_client = RayClient()
ray_client.start()
try:
    result = workflow.run()  # WorkflowRunResult
finally:
    ray_client.stop()
```

`LLMJudgeWorkflow` does not start or stop Ray itself — start a `RayClient` before calling `run()` and stop it after, as shown above.

## Input and output

The runner does not require a fixed text schema. A prompt can reference any fields present in an input JSONL or Parquet row. Keep a stable identifier such as `document_id` when you need to join results to another dataset.

```json
{
  "document_id": "article-42",
  "raw_text": "Example site | Subscribe | The article begins here ...",
  "justext_text": "The article begins here ...",
  "trafilatura_text": "The article begins here, with an extra footer ..."
}
```

Data Designer adds one top-level column for each judge. A judge named `extraction_quality` with a `quality` score produces a result shaped like this:

```json
{
  "extraction_quality": {
    "quality": {
      "reasoning": "The candidate retains the article body and drops the navigation.",
      "score": 4
    }
  }
}
```

An input field can be `null`. Make optional Jinja fields null-safe, for example `{{ (trafilatura_text or "")[:8000] }}` rather than `{{ trafilatura_text[:8000] }}`.

## Scaling with Slurm job arrays

For a large evaluation, split the input into many JSONL or Parquet files and submit one single-node job per Slurm array element. Curator automatically detects the Slurm array environment and deterministically assigns source-file tasks to array elements, so each job reads and judges only its assigned files. Each job starts its own local judge server on the GPUs allocated to that job.

A single input file is one source task and cannot be divided across array elements. Use multiple input files (typically with `--files-per-partition 1`) to create enough work to distribute. Array elements can write part files to a shared `--output-path`; use one shared, durable `--checkpoint-path` for the logical run and reuse it only when retrying that same run.

See the [Slurm tutorial](../../slurm/README.md) for submission, runtime configuration, and retry patterns.

## Prompts

Jinja inserts values from the current row: `{{ field_name }}` becomes the value of `field_name` for that record. Use clear delimiters around untrusted source content and tell the model to treat it as evidence rather than instructions.

```jinja
<candidate_a>
{{ justext_text }}
</candidate_a>

<candidate_b>
{{ trafilatura_text }}
</candidate_b>
```

The bundled extraction prompts use character caps as conservative protection against unusually large Common Crawl pages. Those caps are task-specific starting points, not a general truncation policy. Choose limits from representative input lengths and the context window of the judge model. Reserve enough context for the system prompt, rendered prompt, Data Designer's structured-output instructions, and the requested completion.

If one judge needs an earlier judge's result, reference the nested score in a later prompt:

```jinja
The first judge gave content fidelity: {{ extraction_quality.content_fidelity.score }}
```

Within the same execution stage, Data Designer detects that dependency automatically and runs the producing judge first. Across stages, each stage is its own Curator/Data Designer boundary, so ordering is not auto-detected — put the judge that produces the column in an earlier `execution.stages` entry than the judge that consumes it. Omitting `.score` inserts the complete structured result, including its reasoning.

## YAML configuration

The example YAML has two main sections: `models` describes what Dynamo/vLLM serves, and `execution.stages` describes the judge columns to run.

```yaml
models:
  - alias: judge
    model: YOUR_JUDGE_MODEL
    served_model_name: YOUR_JUDGE_MODEL
    dynamo_model:
      num_replicas: 1
      mode: aggregated
      engine_kwargs:
        tensor_parallel_size: 1
        max_model_len: 32768
        max_num_seqs: 16
        gpu_memory_utilization: 0.85
    inference_parameters:
      temperature: 0.0
      max_tokens: 4096
      timeout: 600
      max_parallel_requests: 64

execution:
  stages:
    - name: extraction_quality
      judges:
        - name: extraction_quality
          model_alias: judge
          system_prompt_path: text_extraction_system.jinja
          prompt_path: text_extraction_prompt.jinja
          scores:
            - name: quality
              description: Rate the candidate's usefulness as clean document text.
              options:
                1: Unusable.
                2: Major problems.
                3: Usable with noticeable problems.
                4: Good, with minor problems.
                5: Excellent.
```

Each entry under a stage's `judges:` list is one LLM call per input row, regardless of how many `scores:` it defines — all scores for a judge are returned together in that single call's structured response. Call count scales with the number of judge entries (summed across every stage) and the number of input rows; stage names and `scores:` count do not affect it. For example, `cc_extract_example/text_extraction_qwen_judge.yaml` has 2 judges (2 calls/row), and `cc_extract_example/text_extraction_qwen_gemma_judges.yaml` runs the same rubrics through two models via YAML anchors, giving 4 judges (4 calls/row).

`alias` is the name judges use to select a served model. `model` is the model identifier or local weights path. `served_model_name` is the API name exposed by Dynamo/vLLM and is useful when it differs from the local path.

Each judge needs a unique `name`, a `prompt_path`, and one or more rubric scores. Score option keys may be numeric or string labels, such as `unclear`. Use bare keys for intentional numeric outputs. Quote string labels that YAML would otherwise coerce to another type, such as `"yes"`, `"no"`, `"true"`, `"false"`, `"on"`, `"off"`, and `"null"`. A judge may omit `model_alias` to use the first configured model.

The bundled Qwen example disables thinking through `inference_parameters.extra_body.chat_template_kwargs.enable_thinking`. Keep that setting for Qwen structured judging; remove it for providers that do not support it.

## Model support

Use a HuggingFace-format model vLLM can load, either local weights or a repo id. Some architectures require `trust_remote_code: true` in `engine_kwargs`, and architecture support varies by the installed Dynamo/vLLM version, so a serving failure can mean the version needs updating rather than that the model is unusable. The model has to fit the GPUs it's given (`tensor_parallel_size` per replica × `num_replicas`), and `max_model_len` has to cover the full rendered prompt plus `max_tokens`.

To download the Qwen judge model to a local path referenced by `models[].model`, for example:

```bash
hf download Qwen/Qwen3.8-27B \
  --local-dir /path/to/Qwen3.8-27B
```

A model that serves fine can still be a bad judge, and the only way to know is to run it. The prompt asks the model in plain language to answer with one of a fixed set of values, like "answer with exactly one of: yes, no, unclear." The pipeline checks that answer against the schema afterward. It doesn't stop the model from answering wrong in the first place — it just drops any row where the answer doesn't match. So a model that doesn't follow instructions well shows up as missing rows, not wrong scores.

Two common causes of dropped rows: `max_tokens` set too small for the rubric (more judges, more scores, and longer reasoning all need more room, and a cut-off answer fails the check), and a reasoning model whose thinking is eating the completion budget before it reaches the answer. If that's what's happening, disabling thinking (as in the Qwen example above) is one fix, though it trades away whatever the reasoning might otherwise have added to the judgment.

Smoke-test any new model against your rubric on a small sample before a full run.

## Execution stages and multiple models

Each entry under `execution.stages` becomes its own Curator/Data Designer stage, run in the order listed:

```text
reader, optional language filter, Data Designer stage, filters, Data Designer stage, filters, ..., writer
```

Group judges into one stage when they should share a Data Designer dependency graph (e.g. one judge's prompt references another's result — see [Prompts](#prompts)) or don't need independent tuning. Split judges into separate stages when they need explicit Curator boundaries, separate stage runtime environments, filters between groups, or independent `num_workers`. In particular, avoid grouping judges with very different generation costs (e.g. a multi-field structured judgment alongside a single-field one) into the same stage — mixing heterogeneous request latencies in one stage's shared concurrency pool can push the slower judge's requests past `inference_parameters.timeout` under load, even though the aggregate concurrency ceiling is unchanged. Giving each judge its own stage (or grouping only similarly-sized judges together) avoids that failure mode.

Set `num_workers` on an execution stage to pass a fixed worker count directly to that `DataDesignerStage` through `.with_(num_workers=...)`. This can stop an earlier Data Designer stage from taking every available Ray worker before downstream stages can run against their own served models.

```yaml
execution:
  stages:
    - name: qwen_judges
      num_workers: 1
      judges: [ ... ]
    - name: gemma_judges
      num_workers: 2
      judges: [ ... ]
```

This setting does not limit requests by itself; each worker can still submit up to its model's `max_parallel_requests`.

Splitting judges into more stages creates useful pipeline overlap only when the reader produces multiple Curator tasks. Shard a large input into multiple files; a single JSONL file is one input task and cannot flow into the next Data Designer stage until its first stage finishes.

Multiple models are supported by adding entries with distinct aliases under `models` and selecting `model_alias` per judge. Start every model through the same Dynamo server only when their worker environment requirements are compatible.

## Traces and filters

Every rubric result already includes a short `reasoning` field. Use `with_trace: last_message` for occasional structured-output debugging, or `with_trace: all_messages` while developing prompts and inspecting rendered input. Full traces duplicate prompt content in the output, so turn them off for large production runs unless they are needed. Set `extract_reasoning_content: true` only when you specifically need a provider's separate reasoning-content field.

Add `filters:` at the top level to retain only records that satisfy judge scores. The filter's `judge` is the top-level output column, and `score` is the nested rubric name.

```yaml
filters:
  - judge: extraction_quality
    score: quality
    operator: gte
    value: 4
```

The example supports `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, and `not_in`. Multiple filters use AND semantics. Top-level filters are placed immediately after the stage that produces their judge column; a filter can also be placed under a specific execution stage when you deliberately need it later. Before Ray or the model server starts, the runner checks that every filter refers to a configured judge column and score.

## Analyzing results

Running the same rubric through multiple LLMs turns judge agreement into a signal, not just a sanity check: where the models agree, the record is likely easy and the score can be trusted with less scrutiny; where they disagree, look into why before trusting the rubric or filter at scale. A disagreement can mean the record is genuinely ambiguous or hard to score — evidence for a human-in-the-loop or an `unresolved`-style rubric option — or it can mean the prompt or rubric wording is too vague or underspecified for a model to apply consistently, which calls for tightening the prompt rather than trusting either score.

The writer emits one or more JSONL/Parquet part files under `--output-path`; load the whole directory with your preferred JSON/Parquet tooling (for example `pandas.read_json(..., lines=True)` per part file, concatenated). Running `text_extraction_qwen_gemma_judges.yaml` writes a column per judge — `qwen3_8_27b_text_extraction_judgment` and `gemma_3_27b_text_extraction_judgment` — each holding the nested `{score_name: {"score": ..., "reasoning": ...}}` structure from [Output shape](#output-shape). Pull each judge's `quality.score` into its own column, subtract the two to get a per-row agreement diff, and sort by the absolute difference to surface the largest disagreements.

Read both `reasoning` fields on a disagreement to tell the two causes apart: differing-but-reasonable justifications point to a genuinely hard record, while justifications that latch onto different aspects of the same instructions point to a vague prompt. The same pattern extends to comparing two rubrics on one model, or checking a filter threshold before committing to it.

### Position bias / order-swap check

A judge that scores consistently can still be biased toward whichever candidate appears first (or under a particular field name) in the prompt, independent of content. Multi-model agreement won't catch this, since both models see the same candidate order. To check for it, add a second judge that renders the same comparison with the candidate fields swapped, and compare its verdict to the original on the same rows.

Add a second Jinja template and a second judge entry in the YAML, same pattern as running a rubric through a second model:

```jinja
{# text_extraction_prompt_swapped.jinja #}
<raw_text>
{{ (raw_text or "")[:12000] }}
</raw_text>

<candidate_a>
{{ (trafilatura_text or "")[:8000] }}
</candidate_a>

<candidate_b>
{{ (justext_text or "")[:8000] }}
</candidate_b>
```

```yaml
- name: qwen3_8_27b_text_extraction_judgment_swapped
  model_alias: judge
  system_prompt_path: text_extraction_system.jinja
  prompt_path: text_extraction_prompt_swapped.jinja
  scores: [ ... ]  # same rubric as the original judge
```

Pull each judge's `best_extraction.score` into its own column, then remap the swapped judge's answer back to the original candidate labels before comparing — since candidates were swapped in the prompt, `candidate_a` and `candidate_b` need to be swapped back in the result (`raw`/`none` map to themselves). Compare the original judge's verdict to the remapped swapped-judge verdict row by row: matches mean the verdict held under the order swap, mismatches mean it didn't.

Rows where the normalized verdict flips are evidence of position bias rather than a genuine judgment; read both `reasoning` fields the same way as a model-agreement disagreement to decide whether the rubric or prompt needs tightening. This costs one extra LLM call per row for the swapped judge, so run it on a calibration sample before deciding whether to keep it in a full production run.

## Operating guidance

Start with a manually reviewed calibration sample. Confirm rendered prompts, structured results, and context lengths before increasing concurrency. For a model that fits on one GPU, begin with one replica and modest `max_parallel_requests`; increase requests gradually only after checking for context-length errors, malformed outputs, and GPU memory pressure. Add replicas when additional GPUs are available and the workload is large enough to use them.

Before trusting the rubric, poke it with counterfactuals: take a row with a score you already trust, swap the candidates, rename a source field, reformat without changing content, or splice in a sentence that reads like an instruction. Change one variable per fixture. A score that moves when the thing it's supposed to measure didn't change is a prompt or rubric weakness — cheaper to catch on one hand-picked row than to find as noise across a full run.

`max_model_len` limits total request context, while `max_tokens` limits only the completion. Increasing `max_model_len` consumes KV-cache capacity; it does not make oversized raw documents safe. `max_num_seqs` should be at least the intended in-flight load, but increasing it by itself does not improve throughput.

The bundled examples set `max_model_len` to the judge model's actual max context length. vLLM would infer that value on its own if the key were omitted; it's spelled out explicitly as a reminder to size prompt truncation (the Jinja character caps) to fit within it.

For the optional FastText language gate, provide `--language`, `--fasttext-langid-model-path`, and optionally `--min-langid-score` and `--language-text-field`. Omitting `--language` skips the stage and does not require FastText.
