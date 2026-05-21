# Steward: Synthetic Data Generation

You own SDG. Inference is the dominant cost in these pipelines, so
the model-serving deployment pattern (in-process vs server-endpoint)
is a first-class user choice, not an implementation detail. Apply the
Inference Acceleration concerns in root AGENTS.md.

## Point Of View

Prompts in, generated samples out. You own every prompt the framework
ships, every model integration that produces synthetic text, and the
post-filters that decide which samples survive. Treat shipped prompts
as public API — edits change downstream model behavior.

## Protect

- **Prompt content.** SDG prompts ship as Python string constants in
  `nemotron_cc/prompts.py` today (no YAML/JSON/text template files
  exist under `stages/synthetic/`). Version, document, and changelog
  edits.
- **Two deployment patterns are first-class:**
  - *In-process* — model loaded per stage worker; works when the
    model fits within a Curator worker's GPU memory budget.
  - *Server-endpoint* — Curator stages (CPU-only) call a local
    model server with N vLLM replicas across nodes; saves infra
    setup; lets users switch between hosted and self-hosted models
    without code changes.
- **OpenAI-API compatibility.** SDG stages call OpenAI-compatible
  endpoints; custom Instruct/Reward models work through that
  contract.
- **Client setup in `setup()`, not `__init__`.** SDG stages that
  hold a connection to a model server (vLLM, NIM) initialize the
  client inside `setup()`. Auth tokens, endpoint URLs, and other
  configuration are stored in `__init__` (runtime validation only).
  See the setup-discipline rule in
  [parent](../../AGENTS.md).
- **Output reproducibility.** Given a fixed seed, model, and prompt,
  outputs are reproducible to the extent the underlying model
  allows. Document expected variance.
- **Filter semantics.** Quality / safety / dedup filters applied to
  generated samples are documented and tested.
- **Packaging (if/when file-based templates land).** `MANIFEST.in`
  does `recursive-include nemo_curator *.csv *.json *.yaml *.yml
  *.txt *.md`, so file-based templates under `stages/synthetic/`
  will be packaged by default.

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
  `nemotron_cc/prompts.py` so docs don't hand-enumerate.
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

**Docs:** `fern/` synthetic curation concepts, prompt reference,
model integration guides, tutorials.

**CODEOWNERS:** `@huvunvidia`.
