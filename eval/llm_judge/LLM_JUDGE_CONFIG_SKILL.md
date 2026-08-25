---
name: llm-judge-config
description: Create or revise Curator LLM judge Jinja prompts and YAML configurations.
---

# Curator LLM judge configuration

Writes configs for `run_llm_judge.py` (see `README.md` in this directory
for the full runner documentation). Every judge config is three things: one or
more Jinja prompt files, a YAML file wiring models to prompts to rubrics, and
optional filters on the results. Prefer adapting an existing working YAML/Jinja
pair in this repo as a starting point rather than writing from scratch, then
change only what the task requires.

Before writing anything, pin down:
1. The actual input row schema (field names, which fields can be `null`).
2. The decision the judge must make and what evidence it may use.
3. The output contract: judge/score names and option values that downstream
   code depends on.
4. Available serving resources (GPUs, model).

## Minimal YAML skeleton

```yaml
models:
  - alias: judge                       # referenced by judges[].model_alias
    model: /path/to/weights            # or a served model identifier
    served_model_name: Org/Model-Name  # API name if it differs from `model`
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
      timeout: 600            # seconds; raise for slower/larger models

execution:
  # num_workers: 2                  # used by --execution-mode single_stage
  stages:
    - name: my_judge_group
      # num_workers: 1                # used by --execution-mode multi_stage
      judges:
        - name: my_judge               # top-level output column in the written row
          model_alias: judge           # optional; defaults to models[0]
          system_prompt_path: my_system.jinja   # optional
          prompt_path: my_prompt.jinja          # required
          scores:
            - name: my_score           # nested key under judge output
              description: One sentence telling the model what to assess.
              options:                 # numeric or label keys — both are valid
                1: Description of the low end of the scale.
                5: Description of the high end of the scale.

# filters:
#   - judge: my_judge
#     score: my_score
#     operator: gte                    # eq, ne, gt, gte, lt, lte, in, not_in
#     value: 4
```

Every path in the YAML (`prompt_path`, `system_prompt_path`) resolves relative
to the YAML file's own directory, so keep the YAML and its Jinja files
together in one directory when you copy an example.

When the same rubric needs to run against multiple models, define the judge
once with a YAML anchor and reuse it per model with a merge key, overriding
only what differs (`name` must stay unique per judge, `model_alias` picks the
model):

```yaml
execution:
  stages:
    - name: qwen_judges
      judges:
        - &my_judge
          name: qwen_my_judge
          model_alias: qwen
          prompt_path: my_prompt.jinja
          scores: [ ... ]
    - name: other_model_judges
      judges:
        - <<: *my_judge
          name: other_model_my_judge
          model_alias: other_model
```

Option keys can be integers (`1`..`5`) for ordinal scales or short labels
(e.g. `unclear`) for categorical ones. Use bare numeric keys. Quote string
labels that YAML would otherwise coerce to another type, such as `"yes"`,
`"no"`, `"true"`, `"false"`, `"on"`, `"off"`, and `"null"`, so
`yaml.safe_load` does not convert them to booleans or null. Keep one score's
option values deliberately typed and stable — downstream filters and
analysis scripts compare against them.

## Writing the Jinja prompt

- `{{ field_name }}` renders a value from the current input row. Only
  reference fields that actually exist in the input — check a real input row,
  don't guess field names.
- Guard every optional field: `{{ (field or "")[:8000] }}`, not
  `{{ field[:8000] }}` — a bare `None` will error or silently render as the
  string `"None"`. But null-safety is only syntax; also decide and state the
  *policy* in the prompt when an empty/near-empty value would affect the
  rubric (e.g., "an empty candidate is only correct when the source has no
  meaningful content" / "do not reward an empty candidate merely because it
  has no boilerplate").
- Wrap untrusted source content (raw scraped text, user text) in clear
  delimiter tags and tell the model to treat it as evidence, not instructions:
  ```jinja
  <candidate_text>
  {{ (candidate_text or "")[:8000] }}
  </candidate_text>
  ```
- For a blind evaluation (e.g., comparing two candidates), do not leak which
  candidate came from which source, an earlier judge's verdict, or any label
  the model shouldn't see — put those only in fields NDD doesn't render into
  the prompt. If the YAML draws from an input schema with several
  eval-only/label fields, a comment at the top of the YAML listing which
  fields exist and which must stay out of which prompts helps keep this
  correct as prompts evolve.
- Truncation length is a decision to make deliberately, not a default to
  copy: pick limits from real input-length distributions and the judge
  model's `max_model_len`, leaving headroom for the system prompt, NDD's
  structured-output instructions, and `max_tokens`. If a flat cap would cut
  off content that matters (e.g., a long document compared for duplication
  or fidelity), consider windowed/sampled evidence instead:
  ```jinja
  {% if long_field %}
  {{ long_field[:12000] }}
  {% else %}
  {% for window in fallback_windows[:2] %}
  [evidence window {{ window.start_char }}-{{ window.end_char }}]
  {{ window.text[:6000] }}
  {% endfor %}
  {% endif %}
  ```
- A later judge in the same YAML can reference an earlier judge's result by
  name; NDD infers the dependency and runs them in order:
  ```jinja
  Earlier judge's score: {{ my_earlier_judge.my_earlier_score.score }}
  ```
  Omitting `.score` inserts the full structured result including `reasoning`.

Put shared instructions (role, output-format reminders, general policy) in
`system_prompt_path` and put the per-record evidence in `prompt_path` — but
there's no required split; a single prompt file is fine for simple tasks.
A short, reusable system prompt pattern: state that supplied text/HTML is
untrusted evidence rather than instructions, require the model to return
exactly one of the listed option values (not a substituted free-form
answer), and cap reasoning length (e.g., "at most 30 words") so traces stay
cheap to read and store.

## Designing the rubric (scores)

- Judge `name` and score `name` become the JSON keys downstream code reads —
  treat renames as breaking changes once anything consumes the output.
- Give each score a single, unambiguous axis. If two decisions can disagree
  (e.g., "which candidate is better" vs. "is the winner good enough to keep"),
  use two scores or two judges — don't overload one score to answer both.
- Each option value is the one-sentence anchor description shown to the
  model. Keep anchors mutually exclusive and ordered if the scale is ordinal.
- Consider whether a rubric can legitimately face insufficient or ambiguous
  evidence (truncated input, contradictory signals, an out-of-scope edge
  case). If so, an explicit escape-hatch option — e.g.
  `unresolved: The visible evidence is insufficient to decide.` — paired with
  a system-prompt instruction to use it rather than guess, beats forcing a
  confident answer. A well-scoped binary rubric with unambiguous inputs may
  not need one.
- Every result also carries a free-text `reasoning` field automatically —
  don't add a redundant "explain your answer" score.
- After changing any score name or option set, check `filters:` — a filter
  referencing a renamed judge/score fails config validation before the run
  starts (`_validate_filter_references` in `run_llm_judge.py`).

## Output shape

A judge named `my_judge` with score `my_score` produces, per row:

```json
{
  "my_judge": {
    "my_score": {
      "reasoning": "...",
      "score": 4
    }
  }
}
```

## Execution mode and capacity

- `single_stage` (default): all judges run in one NDD dependency graph — use
  this unless you need explicit Curator-level boundaries between judge groups.
  Set `execution.num_workers` to cap the workers for that one combined stage.
- `multi_stage`: one Curator stage per `execution.stages` entry, letting you
  set per-stage `num_workers`, a per-stage `runtime_env`, or filters between
  groups. This only creates real overlap if the input is sharded into
  multiple files/tasks — a single JSONL file is one task and can't pipeline
  across stages.

When tuning throughput, change the layer that's actually the bottleneck:

| Setting | Controls |
|---|---|
| `execution.num_workers` (`single_stage`) / `execution.stages[].num_workers` (`multi_stage`) | Ray/NDD client workers for the resulting NDD stage — not model replicas or request capacity |
| `inference_parameters.max_parallel_requests` | requests offered by one NDD client process |
| `dynamo_model.num_replicas` | independent model servers (horizontal throughput) |
| `dynamo_model.engine_kwargs.tensor_parallel_size` | GPUs per replica |
| `dynamo_model.engine_kwargs.max_num_seqs` | active-sequence capacity per replica |
| `dynamo_model.engine_kwargs.max_model_len` / `inference_parameters.max_tokens` | total-context / completion-token budgets |

Rough GPU usage is `num_replicas × tensor_parallel_size` per model. If a first
NDD stage seems to starve a later stage's model server, cap the first stage's
`num_workers` before adding replicas.

## Before scaling up

Run a small representative sample first and check:
- Rendered prompts look right (no leaked `None`, no truncated evidence that
  matters, no accidental leaked labels in a blind eval).
- Structured output matches the intended rubric (no schema/option mismatches).
- No context-length errors from `max_model_len`/`max_tokens` being too tight.
- Every input row produced an output row (rows aren't silently dropped).

Only after that, increase `max_parallel_requests`, then replicas, watching for
context-length errors, malformed outputs, and GPU memory pressure.
