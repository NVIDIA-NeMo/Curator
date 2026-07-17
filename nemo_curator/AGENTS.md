<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->

# Steward: pipeline

Protect the stage/task/workflow substrate and the contracts every modality and executor consumes.

Ordinary work: use this map directly with the root map and run only affected checks.
Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.

## Protects

| Invariant | Sev | Backing | Proof / anchor |
| --- | --- | --- | --- |
| ProcessingStage subclasses expose the descriptor properties exercised by the common stage contract tests. | P1 | machine-backed | `uv run pytest tests/stages/common/test_base.py -q -m 'not gpu'` (`stage-descriptors`) |
| Backend task-ID postprocessing preserves the tested root and derived task identity rules. | P1 | machine-backed | `uv run pytest tests/backends/test_task_id_postprocess.py -q -m 'not gpu'` (`task-postprocess`) |
| Per-node setup remains distinct from worker setup and per-task processing. | P1 | manual | nemo_curator/stages/base.py · `def setup_on_node` |
| Tasks retain per-stage performance history and free-form metadata across pipeline boundaries. | P2 | manual | nemo_curator/tasks/tasks.py · `_stage_perf: list[StagePerfStats]` |

## Guardrails

- Treat descriptor, lifecycle, task lineage, and resource changes as cross-backend work.

## Edges

- serves → **backends** (backend-neutral stage and task contracts)
- routes → **deduplication** (dedup workflows)
- routes → **synthetic** (synthetic generation stages)
- routes → **text** (text curation stages)
- routes → **video** (video curation stages)

## Owns

- **code:** `nemo_curator/pipeline`, `nemo_curator/tasks`, `nemo_curator/stages/base.py`, `nemo_curator/stages/resources.py`
- **tests:** `tests/pipelines`, `tests/tasks`, `tests/stages/common`

## Advocate

- Prefer a small composable stage over backend-specific branching in shared pipeline code.

## Do Not

- Do not change shared descriptors to accommodate one executor without checking the others.

## Serves

- Modality stages, backend adapters, workflow authors, and downstream library users.
