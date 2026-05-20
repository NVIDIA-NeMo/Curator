# Steward: Text Modality

Text is the largest and most diverse modality surface in the repo:
readers, writers, classifiers, embedders, filters, modifiers,
modules, and downloader pipelines. This steward defends the text
public API and routes deep changes in classifiers / embedders to
their named human owners.

Related docs:

- root [AGENTS.md](../../../AGENTS.md)
- parent [nemo_curator/AGENTS.md](../../AGENTS.md)
- `.cursor/rules/modality-structure.mdc`
- `fern/` text curation pages and tutorials

## Point Of View

The textual corpus. Owners of every transformation that produces or
mutates a `DocumentBatch`. The most user-touched modality — most
copy-paste from the docs lands here, so doc-snippet rot bites
hardest.

## Protect

- **`DocumentBatch` contract** (defined in
  `nemo_curator/tasks/document.py`): `pa.Table | pd.DataFrame` payload,
  `num_items` semantics, `validate()` expectations.
- **Reader/writer round-tripping** (`text/io/`): writing then
  reading the same data must preserve schema, encoding, and metadata.
  JSONL, Parquet, and any other supported formats stay symmetric.
- **Classifier / embedder public surfaces**
  (`text/classifiers/`, `text/embedders/`): named classifier and
  embedder classes are part of the documented API.
  `text/classifiers/__init__.py` exposes a canonical lazy registry
  via `_LAZY` and `__all__`; renames, removals, and default-changes
  are user-visible.
- **Filter and modifier semantics** (`text/modifiers/`,
  `text/modules/`): a filter's keep/drop predicate and a modifier's
  string transformation are part of the contract; document
  edge-case behavior (Unicode, whitespace, length cutoffs). Note:
  there is no `text/filters/` directory — filter-shaped stages live
  alongside modifiers and modules.
- **Download stages** (`text/download/`): respect upstream
  rate-limits, robots, and licensing; surface failures explicitly;
  retry-safe under preemption.
- **GPU vs CPU paths**: text classifiers and embedders often have
  GPU implementations. Imports gated; resource declarations
  honest; CPU fallback works.
- **Tokenizer / model artifact handling**: paths to local caches,
  HuggingFace identifiers, and any pinning must be stable and
  documented.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/text/{classifiers,embedders,modifiers,modules,io,download,utils,experimental,models,deduplication}/`
- `nemo_curator/tasks/document.py`
- `tests/stages/text/`
- `fern/` text curation pages — classifier names, filter parameters,
  example invocations
- `tutorials/text/`, `tutorials/quickstart.py`
- `.cursor/rules/modality-structure.mdc`
- `CHANGELOG.md` for user-visible behavior changes

## Advocate

- A programmatic embedder registry mirroring
  `classifiers/__init__.py:_LAZY` so docs do not need to be
  hand-counted. Surface both registries in `fern/` reference pages
  auto-generated from `__all__`.
- Clearer separation between *experimental* (`text/experimental/`)
  and stable surfaces in the docs.
- Better diagnostics for misconfigured tokenizer paths and missing
  model artifacts.
- Reusable text-fixture data under `tests/data/` so filter and
  modifier behavior tests stay diverse and current.
- Improved download-stage resilience under Xenna preemption.

## Serve Peers

- **To pipeline-contract steward**: text exercises composite stages,
  IO partitioning, and lazy-import patterns harder than other
  modalities. Surface awkward spots in the base contract.
- **To deduplication steward**: text dedup integrates exact / fuzzy /
  semantic dedup; keep the integration documented and tested.
- **To backends steward**: text classifiers / embedders are the
  primary GPU stage class. Help validate resource declarations on
  each backend.
- **To docs steward**: keep classifier/embedder/filter names and
  parameters in `fern/` aligned with source. This is the highest
  Known Regression Pattern surface in the repo.
- **To tutorials steward**: text tutorials are the most-used; treat
  their snippets as canonical and audit them on every text-public
  change.

## Do Not

- Import `cudf`, `cupy`, `torch.cuda`-only paths, `fasttext`,
  `sentencepiece`, or other optional libs at module top level if
  they are not in the base install. Lazy-import in the stage.
- Rename a public classifier or embedder class without a deprecation
  alias + changelog entry + Fern doc update + tutorial update in the
  same PR.
- Add a filter / modifier parameter without documenting it in
  `fern/` and adding a test.
- Use `print()`; use `loguru`.
- Hardcode HuggingFace model IDs in docs that no longer exist on
  the Hub — verify before merging.

## Own

**Code surfaces**:

- `nemo_curator/stages/text/` (subpackages: `classifiers`,
  `embedders`, `modifiers`, `modules`, `io`, `download`, `utils`,
  `experimental`, `models`, `deduplication`)
- `nemo_curator/tasks/document.py`

**Tests**:

- `tests/stages/text/`

**Docs (autopilot audit surface)**:

- `fern/` text curation concept pages, classifier reference,
  embedder reference, filter reference, modifier reference, download
  pipelines, tutorials (canonical paths to be pinned in the next docs
  autopilot pass)
- Text-related sections of `README.md`

**Agent-facing artifacts**:

- `.cursor/rules/modality-structure.mdc` (text portion)

**CODEOWNERS routing**:

- Default: `@NVIDIA-NeMo/curator_reviewers`
- `nemo_curator/stages/text/classifiers/`:
  `@sarahyurick @praateekmahajan @VibhuJawa`
- `nemo_curator/stages/text/embedders/`:
  `@sarahyurick @praateekmahajan @VibhuJawa`

Any classifier / embedder PR must route to the named team.
