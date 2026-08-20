# Generic text LLM judge

Use this example when you have JSONL or Parquet records and want an LLM to evaluate fields in each record with your own prompt and rubric. You supply the input data, a YAML configuration, and one or more small Jinja prompt files.

The runner starts a local Curator Dynamo/vLLM `InferenceServer`, runs NeMo Data Designer (NDD) judge columns, and writes the original records plus judge output. The bundled example compares `raw_text`, `justext_text`, and `trafilatura_text`. Copy and adapt it to judge a PDF parser result against raw PDF text, assess whether two documents are semantic duplicates, or evaluate another text-processing result.

You do **not** need to write NDD Python code or call NDD yourself. The runner does that. For a first run, you only need to set a model, point the prompt at your input fields, and define the scores you want back.

## Before you start

You need:

- A Curator environment that already includes the NDD integration and Dynamo/vLLM support.
- A model identifier or local model path that Dynamo/vLLM can serve.
- A small JSONL or Parquet file to try first.

You do not need to understand the internal difference between Curator, Dynamo, and NDD to use the example. Think of the script as a program that reads rows, sends each row to a judge model using your instructions, and writes the score back beside the row.

## Five-minute path

1. Start by editing the files in `examples/`; do not copy only the YAML yet, because it refers to the adjacent `.jinja` files by relative path.
2. In [text_extraction_judge.yaml](examples/text_extraction_judge.yaml), replace both `YOUR_JUDGE_MODEL` values with your model identifier or local model path.
3. In [text_extraction_prompt.jinja](examples/text_extraction_prompt.jinja), replace `raw_text`, `justext_text`, and `trafilatura_text` with the actual field names in your input file.
4. In the YAML `scores:` section, replace the supplied extraction-quality rubric with the decisions you want the model to make.
5. Run the script on a small input file.

```bash
python eval/llm_judge/generic_text_judge.py \
  --judge-config eval/llm_judge/examples/text_extraction_judge.yaml \
  --input-path data/extracted.jsonl \
  --input-format jsonl \
  --output-path output/judged \
  --output-format jsonl
```

Use `--checkpoint-path output/judge_checkpoint` if you want Curator’s normal pipeline-resume metadata written to a durable local directory.

### Cancelling a local Dynamo run

Press `Ctrl-C` once and allow the script to finish its normal
`InferenceServer`/Ray cleanup. **TODO:** investigate a general Curator/Dynamo
solution for reliably reaping model-server subprocesses when cancellation or a
Ray shutdown interrupts normal cleanup.

### Optional: retain only selected languages before judging

The judge script does not filter language by default. To run Curator's `FastTextLangId` filter immediately after the reader and before every NDD stage, provide one language label and a local FastText model file:

```bash
python eval/llm_judge/generic_text_judge.py \
  --judge-config eval/llm_judge/examples/text_extraction_judge.yaml \
  --input-path data/extracted.jsonl --input-format jsonl \
  --output-path output/judged \
  --language en \
  --fasttext-langid-model-path /models/language-identification.bin
```

`--language en` retains records FastText identifies as English. The label must match the model you supply. `--min-langid-score` controls the confidence threshold and defaults to `0.3`. Use `--language-text-field` to select the input column used for identification; it defaults to `raw_text` for the Common Crawl comparison data. Omit `--language` entirely to skip the stage and avoid needing FastText or its model file.

### Create a small Common Crawl comparison dataset

If you do not already have extracted text, [cc_extract.py](cc_extract.py) uses Curator’s Common Crawl downloader and WARC iterator, then runs both the built-in jusText and Trafilatura algorithms on each decoded HTML record. It writes the `raw_text`, `justext_text`, and `trafilatura_text` fields expected by the bundled judge prompt.

```bash
python eval/llm_judge/cc_extract.py \
  --download-dir data/cc_warcs \
  --output-path data/cc_extractions
```

It defaults to the CC-MAIN `2026-30` snapshot, one WARC file, and 100 records per WARC for a small trial. A WARC file can still be large; raise `--url-limit` and `--record-limit` only after checking the first output. Override `--start-snapshot` and `--end-snapshot` to use another snapshot.

### The only three things you normally customize

| What you want to change | Where to change it | Example |
|---|---|---|
| Which judge model runs | `models[0].model` in YAML | `nvidia/...` or a local model path |
| Which text the judge reads | `{{ field_name }}` in a `.jinja` prompt | `{{ parser_output }}` |
| Which decisions it returns | `scores:` in YAML | `duplicate: yes/no`, `quality: 1–5` |

Leave `dynamo_model` alone for a first run unless you already know the GPU layout required by your model.

## Input data

The input can be JSONL or Parquet. A JSONL row for the bundled example looks like this:

```json
{
  "document_id": "article-42",
  "raw_text": "Example site | Subscribe | The article begins here ...",
  "justext_text": "The article begins here ...",
  "trafilatura_text": "The article begins here, with an extra footer ..."
}
```

The runner does not impose a text schema. Field names only matter because your Jinja templates refer to them. Keep a stable record identifier such as `document_id` if you need to join output back to another dataset.

An extractor can legitimately produce no text. In JSONL that is often represented as `null`, for example `"trafilatura_text": null`. The bundled extraction prompts handle that case and pass an empty candidate to the judge, allowing the rubric to mark the extraction as poor or unusable. If you slice or otherwise transform an optional field in your own prompt, use a null-safe expression such as `{{ (trafilatura_text or "")[:8000] }}`; `{{ trafilatura_text[:8000] }}` fails when the value is `null`.

## Jinja prompts: insert values from the current row

Jinja is a template language. Here, its main job is simple: `{{ field_name }}` is replaced with that field’s value from the current input record before the LLM sees the prompt. It is not a separate program you need to install or run.

For the row above, this template:

```jinja
<candidate_a>
{{ justext_text }}
</candidate_a>

<candidate_b>
{{ trafilatura_text }}
</candidate_b>
```

becomes:

```text
<candidate_a>
The article begins here ...
</candidate_a>

<candidate_b>
The article begins here, with an extra footer ...
</candidate_b>
```

Start with direct field references. You can use normal prose, headings, and XML-like delimiters freely; Jinja only treats `{{ ... }}` as an insertion. Put untrusted document text inside clear delimiters and tell the judge to treat it as evidence, not instructions.

### Temporary length caps in the bundled extraction prompts

The two bundled extraction prompts deliberately use Jinja string slices:

```jinja
{{ (raw_text or "")[:12000] }}
{{ (justext_text or "")[:8000] }}
{{ (trafilatura_text or "")[:8000] }}
```

These are **temporary character-based safeguards**, not a general recommendation
for how to truncate documents. Common Crawl raw HTML has very large outliers;
passing it through verbatim can exceed the model's context window, fail the
request, or collapse throughput by filling the model's KV cache. The limits
give the included four-replica, TP=1, 32k-context example a safe starting
point.

Character counts are only an approximation to token counts, and truncating
from the beginning can omit relevant later content. A future version of this
runner should offer an upstream token-aware `TokenLengthFilter` stage. Until
then, treat these slices as task-specific calibration values: inspect the
length distribution of your inputs, choose caps for your model and rubric, and
replace or remove them when your data has an appropriate token-length policy.

The bundled prompt files are:

- [text_extraction_system.jinja](examples/text_extraction_system.jinja): stable evaluator instructions.
- [text_extraction_prompt.jinja](examples/text_extraction_prompt.jinja): the main comparison prompt.
- [text_extraction_disagreement_prompt.jinja](examples/text_extraction_disagreement_prompt.jinja): a second judge prompt.

If a later judge is in the same NDD stage, it can reference an earlier judge’s numeric rubric value:

```jinja
The first judge gave content fidelity:
{{ qwen3_8_27b_text_extraction_judgment.content_fidelity.score }}
```

NDD detects that reference and runs the earlier judge first. The bare `{{ qwen3_8_27b_text_extraction_judgment.content_fidelity }}` value is the full structured result, including reasoning and score; append `.score` when you want only the rubric value.

## Write the YAML for your task

YAML is just an indented configuration file. Keep the indentation shown in the example: list entries start with `-`, and nested settings are indented beneath their parent. The YAML has three sections.

### 1. `models`: what to serve

Every entry starts one model behind the local Dynamo endpoint. `alias` is the short name judges use; `model` is the actual model identifier or local model path. `served_model_name` is useful when the local path and API model name must differ.

```yaml
models:
  - alias: judge
    model: YOUR_JUDGE_MODEL
    served_model_name: YOUR_JUDGE_MODEL
    dynamo_model:
      num_replicas: 4
      mode: aggregated
      engine_kwargs:
        tensor_parallel_size: 1
        max_model_len: 32768
        max_num_seqs: 256
        gpu_memory_utilization: 0.8
        enforce_eager: true
    inference_parameters:
      temperature: 0.0
      # Qwen-specific: avoid spending the completion budget on thinking before
      # it emits NDD's required structured JSON.
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
      max_tokens: 512
      max_parallel_requests: 32
```

The bundled YAML is a concrete four-replica, TP=1 baseline: each replica uses
one GPU, so it requires four visible GPUs. `temperature: 0.0` is a good
default for reproducible judging. Reduce `num_replicas` and
`max_parallel_requests` together when running on fewer GPUs.

The `extra_body` block is specific to the bundled Qwen example. Qwen enables
thinking by default; for a structured judge, that can consume the completion
budget before the model returns NDD's required JSON. Keep
`enable_thinking: false` for this task, or remove the block for a different
model that does not support this Qwen option.

### Tune for throughput without hiding prompt problems

There is no useful universal concurrency number: throughput depends mostly on
input-token length, requested output tokens, GPU memory, and model size. Start
with the included values, inspect a representative partition, and change one
knob at a time.

| Setting | What it limits | Practical guidance |
|---|---|---|
| `engine_kwargs.max_model_len` | Maximum total context accepted by vLLM for one request. | It must hold the rendered system prompt, rendered Jinja prompt, structured-output instructions, and completion. Increasing it consumes KV-cache capacity even for short requests. Do not use it as a substitute for truncating multi-megabyte input rows. |
| `inference_parameters.max_tokens` | Maximum completion tokens per judge request. | This is not the input limit. Set it only large enough for the structured judge result and rubric reasoning. The bundled three-score judge uses `512`; a simpler one-score rubric may fit in `256`. Raise it after observing genuinely truncated responses. |
| `inference_parameters.max_parallel_requests` | Maximum NDD requests in flight for this model alias. | A conservative starting point is about `8` per replica for long or highly variable text—therefore `32` for this four-replica example. Increase gradually only if requests succeed and GPU memory has headroom. It is an NDD admission limit, not a promise that the server can execute that many full-context requests at once. |
| `engine_kwargs.max_num_seqs` | vLLM's upper bound on active sequences. | Keep it at least as high as the in-flight load you intend to admit. Raising it alone does not create throughput and can increase memory pressure. The example's `256` is not the first knob to tune. |
| `num_replicas` | Independent model replicas. | Add replicas when you have additional GPUs and enough requests to keep them busy. This is the usual way to scale aggregate throughput after a single replica is healthy. |
| `tensor_parallel_size` | GPUs used by each replica. | Increase TP when the model does not fit on one GPU or requires more memory bandwidth. It consumes GPUs per replica; it is not automatically better than using more TP=1 replicas for independent judge requests. |
| `gpu_memory_utilization` | Fraction of GPU memory vLLM may use. | Keep headroom while validating long inputs. Raise cautiously only after confirming no OOMs; more memory can improve KV-cache capacity, but leaves less margin for model/runtime overhead. |

A practical tuning sequence is:

1. Run a small calibration sample with `with_trace: all_messages` or `last_message` and confirm prompt rendering, scores, and context lengths.
2. Turn full tracing off for the throughput run; it duplicates prompt text in the output and can become a significant I/O and storage cost.
3. Run one representative input partition using `max_tokens: 512` and a concurrency matched to the replica count (`32` for the bundled four-replica configuration).
4. Check for context-length errors, malformed/truncated structured output, GPU OOMs, and GPU utilization. Fix oversized prompts before raising concurrency.
5. Increase `max_parallel_requests` one step at a time. When one replica is genuinely saturated and the results are correct, scale with replicas if GPUs are available.

For one JSONL input file, the reader supplies one `DocumentBatch` to NDD. The
stage writes only after that batch's judge work completes, so output does not
necessarily appear row by row. `--files-per-partition` does not improve
throughput for a command that names exactly one file.

`dynamo_server` contains settings shared by every served model:

```yaml
dynamo_server:
  subprocess_env:
    DYN_SYSTEM_PORT: "0"
```

### 2. `execution`: one NDD stage or several

`execution.stages` groups judges. Choose the mode that matches how Curator should execute those groups:

```yaml
execution:
  mode: single_stage  # or multi_stage
  stages:
    - name: extraction_quality
      judges: [...]
    - name: semantic_disagreement
      judges: [...]
```

`single_stage` builds this pipeline:

```text
reader → one DataDesignerStage containing every judge → writer
```

Use it when NDD should own the whole dependency graph. Independent judges can be scheduled together, and Jinja references between judges define ordering.

`multi_stage` builds one streaming Curator pipeline with separate NDD stages:

```text
reader → NDD stage: extraction_quality → NDD stage: semantic_disagreement → writer
```

It is still one pipeline and one final writer. Different document batches can be in different stages at the same time. Use it when a group needs a distinct Curator `runtime_env`, when you want an explicit stage boundary, or when you want its model calls/configuration isolated from other judge groups. To switch, change only `mode: multi_stage`; the command remains the same.

### 3. `judges`: the evaluation semantics

Each judge needs:

- `name`: the new output column name; it must be unique.
- `model_alias`: an alias from `models`.
- `prompt_path`: a Jinja file, relative to the YAML file unless absolute.
- `scores`: one or more named rubric dimensions.

For example:

```yaml
- name: extraction_quality
  model_alias: judge
  system_prompt_path: text_extraction_system.jinja  # optional
  prompt_path: text_extraction_prompt.jinja
  with_trace: last_message                           # optional
  extract_reasoning_content: false                   # optional
  scores:
    - name: quality
      description: Rate whether the candidate preserves meaningful content and removes boilerplate.
      options:
        1: Unusable.
        2: Major problems.
        3: Usable with noticeable problems.
        4: Good, with minor problems.
        5: Excellent.
```

Write score descriptions as the judge’s grading guide. The option keys can be numbers or labels such as `yes` / `no` / `unclear`. For a choice among several texts, a string-valued score is often clearer:

```yaml
- name: preferred_candidate
  description: Choose the most faithful clean extraction.
  options:
    justext: Best overall extraction.
    trafilatura: Best overall extraction.
    raw: Raw text is preferable to both extractions.
    none: No candidate is usable.
```

### When to capture a trace or reasoning content

The normal judge result already contains a short, rubric-specific `reasoning` value beside each `score`. That is usually enough to inspect why a row received a score.

`with_trace` is a separate debugging artifact. It captures the messages NDD exchanged for that judge column and writes an additional `<judge_name>__trace` output column. Choose one of these values:

- Omit `with_trace` (the default) for a large production run when you only need scores and rubric reasoning.
- `with_trace: last_message` when you want the final model response for occasional audit or structured-output debugging, without duplicating the prompt text.
- `with_trace: all_messages` while developing a prompt, investigating surprising scores, debugging Jinja rendering, or using MCP tools. It captures the full system/user/assistant/tool-message history, so you can see the actual rendered document text and model response for that row.

For example, leave `all_messages` on for a small calibration sample, inspect bad or borderline decisions, then normally turn it off before processing a large Common Crawl slice. Full traces duplicate prompt inputs—including raw HTML in this example—so they can make output much larger and may retain content you would rather not persist.

`extract_reasoning_content: true` is different again. Some reasoning-capable providers send a separate reasoning-content field in the final response; NDD copies it to `<judge_name>__reasoning_content` when present. It does not create that field, is not guaranteed to be available for every model, and is not needed for the normal rubric `reasoning` values. Treat it as an optional provider/model diagnostic, not as something downstream filtering should rely on.

## Keep only records that pass a judge score

Add `filters:` at the top level of the YAML. The runner identifies the NDD stage that creates each filter’s `judge` column and places the Curator `Filter` immediately afterward. The location of `filters:` in the YAML file does not matter.

```yaml
execution:
  mode: multi_stage
  stages:
    - name: extraction_quality
      judges: [...]  # define qwen3_8_27b_text_extraction_judgment here

filters:
  - judge: qwen3_8_27b_text_extraction_judgment
    score: content_fidelity
    operator: gte
    value: 4
  - judge: qwen3_8_27b_text_extraction_judgment
    score: boilerplate_removal
    operator: gte
    value: 4
```

In `multi_stage`, the result is:

```text
reader → NDD stage: extraction_quality → its Filter stages → NDD stage: semantic_disagreement → writer
```

In `single_stage`, every judge runs in the one NDD stage, then every filter runs before the writer. The filters above keep only rows whose two scores are at least 4. Multiple filters are combined with **AND**: a row must pass every listed filter. The supported operators are:

| Operator | Meaning | Example |
|---|---|---|
| `eq`, `ne` | equals / does not equal | `eq` with `value: justext` |
| `gt`, `gte` | greater than / greater than or equal | `gte` with `value: 4` |
| `lt`, `lte` | less than / less than or equal | `lt` with `value: 3` |
| `in`, `not_in` | included in / excluded from a list | `in` with `value: [justext, trafilatura]` |

The runner validates that the named judge and score exist in the YAML. In `multi_stage`, a filter cannot reference a later judge because it is placed after the stage that produces its own judge. A missing or malformed judge result does not pass the filter.

For an unusual case where you deliberately want a filter later than its producer, you may instead put `filters:` inside an execution-stage mapping. That placement is explicit; for ordinary use, top-level filters are simpler.

## Multiple models

Both execution modes can route each judge to a different served model. Add models with distinct aliases, then select the alias per judge:

```yaml
models:
  - alias: fast_judge
    model: YOUR_FAST_MODEL
    served_model_name: fast_judge
    dynamo_model:
      num_replicas: 1
      engine_kwargs: {tensor_parallel_size: 1}
  - alias: strong_judge
    model: YOUR_STRONG_MODEL
    served_model_name: strong_judge
    dynamo_model:
      num_replicas: 1
      engine_kwargs: {tensor_parallel_size: 2}

execution:
  mode: single_stage
  stages:
    - name: review
      judges:
        - name: quick_screen
          model_alias: fast_judge
          # prompt_path and scores go here
        - name: final_adjudication
          model_alias: strong_judge
          # prompt_path and scores go here
```

Model-level `dynamo_model.runtime_env` configures that model’s vLLM workers. With `multi_stage`, `execution.stages[].runtime_env` configures Curator workers for that NDD stage. Models in one Dynamo server still share a frontend, so keep their environment/package requirements compatible.

## What the output contains

NDD adds one nested result per judge. A judge named `extraction_quality` with a score named `quality` produces a value shaped like:

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

If `with_trace` is enabled, NDD also writes `extraction_quality__trace`. If `extract_reasoning_content: true` is enabled, it writes `extraction_quality__reasoning_content` when the model provider returns a separate reasoning-content field. These are different from the rubric’s `reasoning` above; see “When to capture a trace or reasoning content” above for when to use each.

## Common adaptations

| Goal | Input fields in your prompt | Useful rubric |
|---|---|---|
| Compare two texts for duplicate content | `{{ left_text }}`, `{{ right_text }}` | `semantic_duplicate: yes/no/unclear` |
| Compare extracted text to raw HTML/text | `{{ raw_text }}`, `{{ candidate_text }}` | fidelity, boilerplate removal, usability |
| Judge a PDF parser result | rendered-page or OCR fields plus `{{ parsed_text }}` | coverage, reading order, hallucination |
| Route cheap screening to a strong adjudicator | earlier judge output plus source fields | pass/fail, final decision |

For every adaptation, begin with a small manually reviewed sample. Tune the prompt and rubric definitions before spending resources on a large corpus.
