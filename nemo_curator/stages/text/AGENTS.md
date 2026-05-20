# Steward: Text Modality

This domain exists because text is the largest user-facing surface in
the framework and the most copied-and-pasted. Most new users meet
NeMo Curator through text — running a quality classifier, deduplicating
a corpus, or extracting and filtering Common Crawl. Doc-snippet rot
here breaks the first thing those users try.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md),
`.cursor/rules/modality-structure.mdc`.

## Point Of View

The textual corpus. Owns every transformation that produces or mutates
a `DocumentBatch`: download, extraction, filtering, classification,
embedding, deduplication, modification, synthetic generation. Defends
the published classifier surfaces (Domain, Quality, and others on
HuggingFace) as public API — their class names, output schemas, and
documented rubrics are downstream user contracts, not implementation
details.

## Protect

- **`DocumentBatch` contract** (`nemo_curator/tasks/document.py`):
  `pa.Table | pd.DataFrame` payload, `num_items` semantics,
  `validate()` expectations.
- **Reader/writer round-tripping** (`text/io/`): writing then reading
  preserves schema, encoding, and metadata for every supported
  format (JSONL, Parquet, others).
- **Classifier public surface.** `text/classifiers/__init__.py`
  exposes a canonical lazy registry via `_LAZY` and `__all__`. The
  Domain Classifier (26 domain classes — Finance, Health, Business
  and Industrial, Science, Law and Government, Internet and Telecom,
  Jobs and Education, News, Computers and Electronics, Shopping, and
  16 others) and the Quality Classifier (High / Medium / Low against
  a rubric of content accuracy, clarity, coherence, grammar, depth of
  information, and overall usefulness) are user-visible API published
  on HuggingFace. Renames, removals, default changes, or rubric
  changes are major-version events.
- **Embedder public surface.** `text/embedders/` is inference-bearing
  — coordinate with the Inference Acceleration Steward when changing
  serving stacks or throughput characteristics.
- **Filter and modifier semantics** (`text/filters/`,
  `text/modifiers/`, `text/modules/`): keep/drop predicates
  (`text/filters/{doc_filter,score_filter}.py` and the `fasttext`,
  `heuristic`, `histogram`, `token` subpackages under filters) and
  string transforms are contract. Document edge cases (Unicode,
  whitespace, length cutoffs).
- **Download stages** (`text/download/`): respect upstream rate
  limits / robots / licensing; surface failures explicitly; retry-safe
  under preemption.
- **GPU/CPU paths.** Text classifiers and embedders often have GPU
  implementations. Lazy-import GPU libs; declare resources honestly;
  CPU fallback loads.
- **Tokenizer and model artifact handling.** Local cache paths,
  HuggingFace identifiers, and any pinning must be stable and
  documented. HuggingFace-pinned artifacts can vanish; this is an
  operational risk to manage.

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
  `classifiers/__init__.py:_LAZY`, so docs don't hand-count. Surface
  both registries in `fern/` reference pages auto-generated from
  `__all__`.
- **Clearer separation** between `text/experimental/` and stable
  surfaces in the docs, plus a graduation path so experimental
  doesn't become permanent.
- **Better diagnostics** for misconfigured tokenizer paths and
  missing model artifacts.
- **Reusable text fixtures** under `tests/data/` so filter/modifier
  tests stay diverse and current.
- **Improved download-stage resilience** under preemption — partial
  downloads, retries, rate-limit handling.

## Own

**Code:** `nemo_curator/stages/text/` (subpackages: `classifiers`,
`embedders`, `filters`, `modifiers`, `modules`, `io`, `download`,
`utils`, `experimental`, `models`, `deduplication`);
`tasks/document.py`.

**Tests:** `tests/stages/text/`.

**Docs (autopilot surface):** `fern/` text curation concepts,
classifier reference, embedder reference, filter / modifier
reference, download pipelines, tutorials; text-related sections of
`README.md`.

**Agent artifacts:** the text portion of
`.cursor/rules/modality-structure.mdc`.

**CODEOWNERS:** default `@NVIDIA-NeMo/curator_reviewers`.
`classifiers/` and `embedders/` route additionally to
`@sarahyurick @praateekmahajan @VibhuJawa` — any classifier/embedder
PR must route there.
