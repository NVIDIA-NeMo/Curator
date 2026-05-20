# Steward: Text Modality

Text is the largest and most diverse modality: readers, writers,
classifiers, embedders, modifiers, modules, and download pipelines.
The most user-touched modality — most copy-paste from the docs lands
here, so doc-snippet rot bites hardest.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md),
`.cursor/rules/modality-structure.mdc`.

## Point Of View

The textual corpus. Owns every transformation that produces or mutates
a `DocumentBatch`.

## Protect

- **`DocumentBatch` contract** (`nemo_curator/tasks/document.py`):
  `pa.Table | pd.DataFrame` payload, `num_items` semantics,
  `validate()` expectations.
- **Reader/writer round-tripping** (`text/io/`): writing then reading
  preserves schema, encoding, and metadata for every supported
  format (JSONL, Parquet, others).
- **Classifier and embedder public surfaces.**
  `text/classifiers/__init__.py` exposes a canonical lazy registry
  via `_LAZY` and `__all__` — class names there are documented API.
  `text/embedders/` lacks an equivalent registry today (see
  Advocate). Renames, removals, and default changes are
  user-visible.
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
  ensure CPU fallback loads.
- **Tokenizer / model artifact handling.** Local cache paths,
  HuggingFace identifiers, and any pinning must be stable and
  documented.

## Contract Checklist

When this domain changes:

- `stages/text/{classifiers,embedders,filters,modifiers,modules,io,download,utils,experimental,models,deduplication}/`
- `tasks/document.py`
- `tests/stages/text/`
- `fern/` text curation pages — classifier names, filter params,
  example invocations
- `tutorials/text/`, `tutorials/quickstart.py`
- `.cursor/rules/modality-structure.mdc`
- `CHANGELOG.md`

## Advocate

- **Programmatic embedder registry** mirroring
  `classifiers/__init__.py:_LAZY`, so docs don't hand-count.
  Auto-generate `fern/` reference pages from `__all__`.
- Clearer separation between `text/experimental/` and stable
  surfaces in the docs.
- Better diagnostics for misconfigured tokenizer paths and missing
  model artifacts.
- Reusable text fixtures under `tests/data/` so filter/modifier
  tests stay diverse and current.
- Improved download-stage resilience under Xenna preemption.

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
