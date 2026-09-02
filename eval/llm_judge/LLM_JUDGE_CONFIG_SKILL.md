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
4. The evaluation mode: pointwise (each judge assigns a score and we compare
   scores), pairwise (compare A vs B), or both, including explicit tie
   and insufficient-evidence semantics.
5. The policy for pairwise evaluations: use anonymous candidate labels,
   evaluate both A/B and B/A when position bias matters, and preserve the
   mapping back to the original candidates.
6. The aggregation and escalation policy for repeated or multi-model
   judgments, including what constitutes disagreement or an unresolved
   result.
7. Available serving resources (GPUs, model).

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
  stages:
    - name: my_judge_group
      # num_workers: 1                # workers for this stage's DataDesignerStage
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

- `{{ field_name }}` renders a value from the current row — only reference
  fields that actually exist; check a real input row, don't guess.
- Guard optional fields: `{{ (field or "")[:8000] }}`, not `{{ field[:8000] }}`
  — a bare `None` errors or renders as the string `"None"`. Also state the
  *policy* when an empty value should affect the rubric (e.g. "an empty
  candidate is only correct when the source has no meaningful content").
- Wrap untrusted source content in delimiter tags and tell the model to
  treat it as evidence, not instructions:
  ```jinja
  <candidate_text>
  {{ (candidate_text or "")[:8000] }}
  </candidate_text>
  ```
- For blind evaluation, don't leak which candidate came from which source,
  an earlier judge's verdict, or any label the model shouldn't see — keep
  those in fields the template never renders. Use neutral labels
  (`candidate_a`/`candidate_b`), not source identity, unless identity is
  legitimately part of the criterion.
- If position bias matters, render mirrored A/B and B/A prompts and map each
  anonymous position back to the original candidate. If the two judgments
  disagree on the underlying winner, mark it position-inconsistent rather
  than averaging or defaulting to a tie.
- Test with controlled counterfactuals before scaling: candidates swapped,
  source names removed, formatting changed, an instruction-like sentence
  spliced into untrusted content. Change one factor per fixture.
- Pick truncation limits from real input-length distributions and the
  judge's `max_model_len`, leaving headroom for the system prompt, NDD's
  structured-output instructions, and `max_tokens` — don't copy a default.
- A later judge can reference an earlier judge's result by name:
  `{{ my_earlier_judge.my_earlier_score.score }}` (omit `.score` for the
  full result including `reasoning`). Within one `execution.stages` entry,
  NDD infers the dependency and orders judges automatically. Across stages,
  each stage is its own Curator/NDD boundary — this is *not* auto-detected,
  so put the producing judge's stage before the dependent judge's stage.

Put shared instructions (role, output-format reminders) in
`system_prompt_path` and per-record evidence in `prompt_path` — no required
split; one file is fine for simple tasks. A reusable system-prompt pattern:
state that supplied content is untrusted evidence, require exactly one of
the listed option values, and cap reasoning length (e.g. "at most 30 words").

## Designing the rubric (scores)

- Judge `name` and score `name` become JSON keys downstream code reads —
  treat renames as breaking changes.
- One unambiguous axis per score. If two decisions can disagree (e.g. "which
  is better" vs. "is the winner good enough"), use two scores or judges.
- Each option value is a one-sentence anchor shown to the model — keep
  anchors mutually exclusive and ordered if the scale is ordinal.
- If the rubric can legitimately face insufficient/ambiguous evidence, add
  an explicit escape hatch (e.g. `unresolved: ...`) plus a system-prompt
  instruction to use it rather than guess.
- `reasoning` is included automatically — don't add a redundant
  "explain your answer" score.
- After renaming a score/judge, check `filters:` — a stale reference fails
  validation before the run starts (`_validate_filter_references`).

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

## Execution stages and capacity

See `README.md`'s "Execution stages and multiple models" for how
`execution.stages` grouping works, including why judges with very
different generation costs shouldn't share a stage. When tuning
throughput, change the layer that's actually the bottleneck:

| Setting | Controls |
|---|---|
| `execution.stages[].num_workers` | Ray/NDD client workers for that stage's `DataDesignerStage` — not model replicas or request capacity |
| `inference_parameters.max_parallel_requests` | requests offered by one NDD client process |
| `inference_parameters.timeout` | per-request timeout (defaults to 60s); raise it if a judge's rendered prompt/reasoning is long enough to routinely exceed the default |
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
