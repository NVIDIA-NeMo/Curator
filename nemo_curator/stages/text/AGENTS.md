# Steward: Text Modality

You own the text modality — the largest user-facing surface and the
most copied-and-pasted. Doc-snippet rot here breaks the first thing
new users try.

Related: `.cursor/rules/modality-structure.mdc`.

## Point Of View

You own every transformation that produces or mutates a
`DocumentBatch`. Treat the published classifier surfaces (Domain,
Quality, and others on HuggingFace) as public API — their class
names, output schemas, and documented rubrics are user contracts, not
implementation details. Classifier/embedder changes are
inference-bearing — apply the Inference Acceleration concerns in
root AGENTS.md.

## Protect

- **`DocumentBatch` contract** (`nemo_curator/tasks/document.py`):
  `pa.Table | pd.DataFrame` payload, `num_items` semantics,
  `validate()` expectations.
- **Reader/writer round-tripping** (`text/io/`): writing then
  reading preserves schema, encoding, and metadata for every
  supported format (JSONL, Parquet, others).
- **Classifier public surface.** `text/classifiers/__init__.py`
  exposes a canonical lazy registry via `_LAZY` and `__all__`. The
  Domain Classifier (26 classes — Finance, Health, Business and
  Industrial, Science, Law and Government, Internet and Telecom,
  Jobs and Education, News, Computers and Electronics, Shopping,
  and 16 others) and the Quality Classifier (High / Medium / Low
  against a rubric of content accuracy, clarity, coherence,
  grammar, depth of information, and overall usefulness) are
  published on HuggingFace. Renames, removals, default changes, or
  rubric changes are major-version events.
- **Filter and modifier semantics** (`text/filters/`,
  `text/modifiers/`, `text/modules/`). Document edge cases
  (Unicode, whitespace, length cutoffs).
- **Download stages** (`text/download/`): respect upstream rate
  limits / robots / licensing; surface failures explicitly;
  retry-safe under preemption.
- **GPU/CPU paths.** Lazy-import GPU libs; declare resources
  honestly; CPU fallback loads.
- **Model loading in `setup()`, not `__init__`.** Classifiers and
  embedders must load model weights and call `.to("cuda")` inside
  `setup()`. Loading in `__init__` serializes the model to every
  replica. Downloading weights belongs in `setup_on_node()`. See the
  setup-discipline rule in [parent](../../AGENTS.md).
- **Tokenizer and model artifact handling.** HuggingFace-pinned
  artifacts can vanish; treat that as an operational risk.

## Contract Checklist

When this domain changes:

- `stages/text/{classifiers,embedders,filters,modifiers,modules,io,download,utils,experimental,models,deduplication}/`
- `tasks/document.py`
- `tests/stages/text/`
- `fern/` text curation pages — classifier names, filter parameters,
  example invocations
- `tutorials/text/`, `tutorials/quickstart.py`
- `.cursor/rules/modality-structure.mdc`
- `CHANGELOG.md`

## Advocate

- **Programmatic embedder registry** mirroring
  `classifiers/__init__.py:_LAZY` so docs don't hand-count.
  Auto-generate `fern/` reference pages from `__all__`.
- **Clearer separation** between `text/experimental/` and stable
  surfaces, plus a graduation path.
- **Better diagnostics** for misconfigured tokenizer paths and
  missing model artifacts.
- **Reusable text fixtures** under `tests/data/`.
- **Improved download-stage resilience** under preemption.

## Own

**Code:** `nemo_curator/stages/text/` (subpackages: `classifiers`,
`embedders`, `filters`, `modifiers`, `modules`, `io`, `download`,
`utils`, `experimental`, `models`, `deduplication`);
`tasks/document.py`.

**Tests:** `tests/stages/text/`.

**Docs:** `fern/` text curation concepts, classifier reference,
embedder reference, filter / modifier reference, download pipelines,
tutorials; text-related sections of `README.md`.

**Agent artifacts:** the text portion of
`.cursor/rules/modality-structure.mdc`.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`.
`classifiers/` and `embedders/` route additionally to
`@sarahyurick @praateekmahajan @VibhuJawa`.
