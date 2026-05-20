# Steward: Synthetic Data Generation

Synthetic data generation pipelines and prompt templates. Outputs from
this domain seed downstream training corpora — quality and prompt
fidelity matter more than raw throughput.

Related docs:

- root [AGENTS.md](../../../AGENTS.md)
- parent [nemo_curator/AGENTS.md](../../AGENTS.md)
- `fern/` synthetic data curation pages

## Point Of View

Prompts in, generated samples out. Owners of every prompt template
the framework ships, every model integration that produces synthetic
text, and the post-filters that decide which generated samples
survive.

## Protect

- **Prompt template integrity**: today SDG prompts ship as Python
  string constants in `nemotron_cc/prompts.py` (no YAML/JSON/text
  template files exist under `stages/synthetic/`). The constants are
  effectively public API; edits change downstream model behavior —
  version, document, and changelog them.
- **Template packaging (if/when file-based templates are
  introduced)**: `MANIFEST.in` already does `recursive-include
  nemo_curator *.csv *.json *.yaml *.yml *.txt *.md`, so file-based
  templates under `nemo_curator/stages/synthetic/` will be packaged
  by default. For modality-specific prompt directories under other
  trees, mirror the precedent set by the translation-prompts
  packaging (see `pyproject.toml`'s `[tool.setuptools.package-data]`
  section for `nemo_curator.stages.text.experimental.translation`).
- **Output reproducibility**: given a fixed seed, model, and prompt,
  outputs must be reproducible to the extent the underlying model
  allows.
- **Filter semantics**: any quality / safety / dedup filter applied
  to generated samples must be documented and tested.
- **Model integration boundaries**: any external LLM API (NIM,
  OpenAI-compatible, etc.) must surface failures and rate-limits
  clearly, retry safely, and respect quotas.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/synthetic/`
- Prompt template files (paths and packaging)
- `MANIFEST.in` and `pyproject.toml` package-data declarations if
  templates moved
- `tests/stages/synthetic/`
- `fern/` synthetic data curation pages
- `tutorials/synthetic/`
- `CHANGELOG.md`

## Advocate

- A programmatic registry for the prompt constants currently in
  `nemotron_cc/prompts.py` (today they are module-level constants
  with no `__all__` / loader), so docs and tutorials do not
  hand-enumerate.
- A test that imports the package and asserts each template loads —
  catches packaging regressions early.
- Documentation of each template's intended use, expected output
  shape, and known failure modes.
- A consistent seed / determinism story across generators.

## Serve Peers

- **To text steward**: SDG outputs are usually `DocumentBatch`. Keep
  the integration documented.
- **To pipeline-contract steward**: SDG often uses long-running
  external-API stages — surface where fault-tolerance contracts are
  awkward.
- **To docs steward**: prompt templates are easy to drift from docs;
  prioritize accuracy of any "templates we ship" claim.

## Do Not

- Edit a shipped prompt template without a changelog entry and a
  version bump if downstream behavior changes.
- Ship a new template without verifying it loads from an installed
  wheel (not just from source checkout).
- Hard-code API keys or model endpoints; use environment / config.
- Document a template that does not exist on disk.

## Own

**Code surfaces**:

- `nemo_curator/stages/synthetic/nemo_data_designer/data_designer.py`
- `nemo_curator/stages/synthetic/nemotron_cc/{base.py, nemotron_cc.py, prompts.py}`
- `nemo_curator/stages/synthetic/qa_multilingual_synthetic.py`
- Any future file-based prompt templates under
  `nemo_curator/stages/synthetic/`

**Tests**:

- `tests/stages/synthetic/`

**Docs (autopilot audit surface)**:

- `fern/` synthetic data curation concept pages, templates reference,
  model integration guides, tutorials (canonical paths to be pinned
  in the next docs autopilot pass)

**Agent-facing artifacts**: none scoped to SDG yet.

**CODEOWNERS routing**: `@huvunvidia`. Every SDG PR routes here.
