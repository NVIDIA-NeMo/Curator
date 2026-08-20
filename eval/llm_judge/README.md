# Generic LLM judge

Use this example to add LLM-based evaluations to JSONL or Parquet records. The YAML configuration defines the served judge model, Jinja prompt files, rubric scores, and optional output filters. The runner starts a local Curator Dynamo/vLLM server, executes NeMo Data Designer (NDD) judge columns, and writes the original records with the judge results added.

The included example compares jusText and Trafilatura web-text extractions. The same runner can judge parser output, document pairs, extraction quality, semantic duplication, or any task whose inputs can be rendered into a Jinja prompt.

## Quick start

Start by copying and editing the files in `cc_extract_example/`. The YAML refers to adjacent Jinja files by relative path, so keep them together.

1. Set `models[0].model` in [text_extraction_judge.yaml](cc_extract_example/text_extraction_judge.yaml) to a model identifier or local model path.
2. Update [text_extraction_prompt.jinja](cc_extract_example/text_extraction_prompt.jinja) with the field names from your input rows.
3. Define the rubric outputs under each judge's `scores:` list.
4. Run a small input first.

```bash
python eval/llm_judge/generic_text_judge.py \
  --judge-config eval/llm_judge/cc_extract_example/text_extraction_judge.yaml \
  --input-path data/extracted.jsonl \
  --input-format jsonl \
  --output-path output/judged \
  --output-format jsonl
```

Use `--checkpoint-path output/judge_checkpoint` to write Curator checkpoint metadata to a durable location. It is useful for normal pipeline recovery, but you should still inspect input and output counts after a run.

## Input and output

The runner does not require a fixed text schema. A prompt can reference any fields present in an input JSONL or Parquet row. Keep a stable identifier such as `document_id` or `track_id` when you need to join results to another dataset.

```json
{
  "document_id": "article-42",
  "raw_text": "Example site | Subscribe | The article begins here ...",
  "justext_text": "The article begins here ...",
  "trafilatura_text": "The article begins here, with an extra footer ..."
}
```

NDD adds one top-level column for each judge. A judge named `extraction_quality` with a `quality` score produces a result shaped like this:

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

The bundled extraction prompts use character caps as conservative protection against unusually large Common Crawl pages. Those caps are task-specific starting points, not a general truncation policy. Choose limits from representative input lengths and the context window of the judge model. Reserve enough context for the system prompt, rendered prompt, NDD's structured-output instructions, and the requested completion.

If one judge needs an earlier judge's result, reference the nested score in a later prompt:

```jinja
The first judge gave content fidelity: {{ extraction_quality.content_fidelity.score }}
```

NDD detects that dependency and runs the first judge before the dependent one. Omitting `.score` inserts the complete structured result, including its reasoning.

## YAML configuration

The YAML has two main sections: `models` describes what Dynamo/vLLM serves, and `execution.stages` describes the judge columns to run.

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
        max_num_seqs: 32
        gpu_memory_utilization: 0.8
    inference_parameters:
      temperature: 0.0
      max_tokens: 512
      max_parallel_requests: 8

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

`alias` is the name judges use to select a served model. `model` is the model identifier or local weights path. `served_model_name` is the API name exposed by Dynamo/vLLM and is useful when it differs from the local path.

Each judge needs a unique `name`, a `prompt_path`, and one or more rubric scores. Score option keys may be numeric or labels such as `yes`, `no`, and `unclear`. A judge may omit `model_alias` to use the first configured model.

The bundled Qwen example disables thinking through `inference_parameters.extra_body.chat_template_kwargs.enable_thinking`. Keep that setting for Qwen structured judging; remove it for providers that do not support it.

## Execution mode and multiple models

By default, `--execution-mode single_stage` puts every configured judge into one NDD stage. Use it when NDD should schedule the whole dependency graph, including prompt dependencies between judges.

```text
reader, optional language filter, one NDD stage, filters, writer
```

Use `--execution-mode multi_stage` when configured judge groups need explicit Curator boundaries, separate stage runtime environments, or filters between groups.

```text
reader, optional language filter, NDD stage, filters, NDD stage, filters, writer
```

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

## Operating guidance

Start with a manually reviewed calibration sample. Confirm rendered prompts, structured results, and context lengths before increasing concurrency. For a model that fits on one GPU, begin with one replica and modest `max_parallel_requests`; increase requests gradually only after checking for context-length errors, malformed outputs, and GPU memory pressure. Add replicas when additional GPUs are available and the workload is large enough to use them.

`max_model_len` limits total request context, while `max_tokens` limits only the completion. Increasing `max_model_len` consumes KV-cache capacity; it does not make oversized raw documents safe. `max_num_seqs` should be at least the intended in-flight load, but increasing it by itself does not improve throughput.

For the optional FastText language gate, provide `--language`, `--fasttext-langid-model-path`, and optionally `--min-langid-score` and `--language-text-field`. Omitting `--language` skips the stage and does not require FastText.

Press `Ctrl-C` once to cancel a local Dynamo run and allow normal Ray and inference-server cleanup to finish. If cancellation interrupts cleanup, inspect remaining model-server subprocesses before starting another run.

## Common adaptations

| Goal | Prompt fields | Useful scores |
|---|---|---|
| Compare two texts for duplicate content | `{{ left_text }}`, `{{ right_text }}` | `semantic_duplicate: yes/no/unclear` |
| Compare extracted text to raw HTML or text | source plus `{{ candidate_text }}` | fidelity, boilerplate removal, usability |
| Judge PDF parser output | OCR or rendered-page text plus `{{ parsed_text }}` | coverage, reading order, hallucination |
| Route screening to adjudication | an earlier score plus source fields | pass/fail, final decision |

For every adaptation, begin with a small manually reviewed sample and tune the prompt and rubric before processing a large corpus.
