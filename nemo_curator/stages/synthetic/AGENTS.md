# Steward: Synthetic Data Generation

Synthetic data generation pipelines and prompt content. Outputs seed
downstream training corpora — quality and prompt fidelity matter more
than raw throughput.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md).

## Point Of View

Prompts in, generated samples out. Owns every prompt the framework
ships, every model integration that produces synthetic text, and the
post-filters that decide which generated samples survive.

## Protect

- **Prompt content.** SDG prompts today ship as Python string
  constants in `nemotron_cc/prompts.py` (no YAML/JSON/text template
  files exist under `stages/synthetic/`). These constants are
  effectively public API — edits change downstream model behavior, so
  version, document, and changelog them.
- **Output reproducibility.** Given a fixed seed, model, and prompt,
  outputs must be reproducible to the extent the underlying model
  allows.
- **Filter semantics.** Any quality / safety / dedup filter applied
  to generated samples must be documented and tested.
- **Model integration boundaries.** External LLM APIs (NIM,
  OpenAI-compatible) must surface failures and rate limits clearly,
  retry safely, respect quotas.
- **Packaging (if/when file-based templates land).** `MANIFEST.in`
  already does `recursive-include nemo_curator *.csv *.json *.yaml
  *.yml *.txt *.md`, so file-based templates under
  `stages/synthetic/` will be packaged by default. For
  modality-specific prompt directories elsewhere, mirror the
  translation-prompts precedent in `pyproject.toml`'s
  `[tool.setuptools.package-data]` for
  `nemo_curator.stages.text.experimental.translation`.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/synthetic/`
- Prompt strings (paths and content)
- `MANIFEST.in` and `pyproject.toml` package-data declarations if
  file-based templates are added
- `tests/stages/synthetic/`
- `fern/` synthetic curation pages
- `tutorials/synthetic/`
- `CHANGELOG.md`

## Advocate

- A programmatic registry for the constants in `nemotron_cc/prompts.py`
  (module-level constants today, no `__all__` / loader) so docs and
  tutorials don't hand-enumerate.
- A test that imports the package and asserts each prompt constant
  loads — catches packaging regressions early.
- Documentation of each prompt's intended use, expected output shape,
  and known failure modes.
- A consistent seed / determinism story across generators.

## Own

**Code:**

- `nemo_curator/stages/synthetic/nemo_data_designer/data_designer.py`
- `nemo_curator/stages/synthetic/nemotron_cc/` — top-level
  (`base.py`, `nemotron_cc.py`, `prompts.py`) plus nested
  `nemotron_cc/nemo_data_designer/{base,nemotron_cc}.py`
- `nemo_curator/stages/synthetic/qa_multilingual_synthetic.py`
- Any future file-based prompt templates under `stages/synthetic/`

**Tests:** `tests/stages/synthetic/`.

**Docs (autopilot surface):** `fern/` synthetic curation concepts,
prompt reference, model integration guides, tutorials.

**CODEOWNERS:** `@huvunvidia`.
