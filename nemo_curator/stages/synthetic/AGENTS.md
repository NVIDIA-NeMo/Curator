# Steward: Synthetic Data Generation

This domain exists because synthetic data generation is an
inference-dominated workload — the model-serving stack is where cost
and throughput live. SDG pipelines spend the vast majority of their
time in LLM inference; everything else (filters, modifiers, post-
processing) is supporting infrastructure around that core. Curator's
job here is to compose generation, scoring, and filtering stages over
an OpenAI-API-compatible model server with predictable throughput.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md). Inference choices route
through the Inference Acceleration Steward (root AGENTS.md).

## Point Of View

Prompts in, generated samples out. Owns every prompt the framework
ships, every model integration that produces synthetic text, and the
post-filters that decide which generated samples survive. The
deployment-pattern choice — in-process model load per stage versus
CPU-only Curator stages calling a local model server — is a
first-class user-facing decision, not an implementation detail.

## Protect

- **Prompt content.** Today SDG prompts ship as Python string
  constants in `nemotron_cc/prompts.py` (no YAML/JSON/text template
  files exist under `stages/synthetic/`). The constants are
  effectively public API; edits change downstream model behavior —
  version, document, and changelog them.
- **Two deployment patterns are first-class:**
  - *In-process* — model loaded per stage worker; works when the
    model fits within a Curator worker's GPU memory budget. Stage
    owns the model lifecycle.
  - *Server-endpoint* — Curator stages (CPU-only) call a local model
    server; N vLLM replicas across multiple nodes; saves infra
    setup; lets users switch between hosted and self-hosted models
    without code changes. Coordinate with the Inference
    Acceleration Steward on serving stack choice.
- **OpenAI-API compatibility.** SDG stages call OpenAI-compatible
  endpoints; integration with custom Instruct/Reward models works
  through that contract.
- **Output reproducibility.** Given a fixed seed, model, and prompt,
  outputs are reproducible to the extent the underlying model
  allows. Document expected variance — without that, every
  regression looks like "did we break it or did the model drift?"
- **Filter semantics.** Any quality / safety / dedup filter applied
  to generated samples must be documented and tested.
- **Packaging (if/when file-based templates land).** `MANIFEST.in`
  already does `recursive-include nemo_curator *.csv *.json *.yaml
  *.yml *.txt *.md`, so file-based templates under
  `stages/synthetic/` will be packaged by default. For
  modality-specific prompt directories elsewhere, mirror the
  translation-prompts precedent in `pyproject.toml`'s
  `[tool.setuptools.package-data]`.

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

- **A programmatic registry for prompt constants** in
  `nemotron_cc/prompts.py` (module-level constants today, no
  `__all__` / loader) so docs and tutorials don't hand-enumerate.
- **A test that imports the package and asserts each prompt
  constant loads** — catches packaging regressions early.
- **Documentation of each prompt's intended use**, expected output
  shape, and known failure modes.
- **A consistent seed / determinism story** across generators.

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
