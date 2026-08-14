<!-- generated steward map; edit source layers, not this file -->
<!-- source layers: .stewards/layers/synthetic.toml -->

> **Guidance owners:** @huvunvidia. Update `.stewards/layers/synthetic.toml`.
> Regenerate with `python .stewards/project.py`, then run `python .stewards/verify.py --coverage`.

# Steward: synthetic

Protect prompt, model-processing, data-designer, and sample I/O contracts for synthetic generation.

Ordinary work: use this map directly with the root map and run only affected checks.
Open `.stewards/` only for explicit review, audit, or steward maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| ModelProcessingStage preserves the setup and sample-processing behavior covered by its focused unit tests. | P1 | machine-backed | `uv run pytest tests/stages/synthetic/omni/test_base.py -q -m 'not gpu'` (`synthetic-base`) |
| Nemotron-CC system prompts are behavior-bearing inputs and changes receive output-quality review. | P2 | manual | nemo_curator/stages/synthetic/nemotron_cc/prompts.py · `NEMOTRON_CC_SYSTEM_PROMPT` |
| DataDesignerStage remains a DocumentBatch-to-DocumentBatch processing stage. | P2 | manual | nemo_curator/stages/synthetic/nemo_data_designer/data_designer.py · `class DataDesignerStage` |

## Guardrails

- Credentialed or network behavior must remain explicitly isolated from focused local tests.

## Edges

- depends-on → **pipeline** (processing-stage and DocumentBatch contracts)
- depends-on → **backends** (execution and resource behavior)

## Owns

- **code:** `nemo_curator/stages/synthetic`
- **tests:** `tests/stages/synthetic`
