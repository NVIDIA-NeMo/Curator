# Audio performance telemetry

Curator can write one raw performance record for each successful generic stage-adapter attempt. The audio `ManifestWriterStage` is the first report consumer, but collection is shared by all `ProcessingStage` implementations executed through the integrated Ray Data or Xenna adapters.

## Enable the report

Set a non-empty `performance_report_path` on the terminal writer:

```yaml
performance_report_path: /output/performance.json

stages:
  - _target_: nemo_curator.stages.audio.common.ManifestWriterStage
    output_path: /output/results.jsonl
    performance_report_path: ${performance_report_path}
```

`null` disables collection, which is the Qwen-Omni tutorial default. An empty or whitespace-only string is invalid.

For the bundled Qwen-Omni configuration, opt in with a Hydra override:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/output/qwen_omni_results.jsonl \
  performance_report_path=/output/qwen_omni_performance.json
```

The manifest and performance paths must resolve to different destinations. Slurm-array runs add a deterministic shard suffix to the report filename.

## Accounting behavior

The report uses an adapter attempt, not an output task, as its accounting unit:

| Adapter result | Output tasks | New raw records |
|---|---:|---:|
| Every item is filtered | 0 | 1 |
| One-to-one processing | 1 | 1 |
| One input fans out | More than 1 | 1 |

Publication occurs after Curator's task-ID, filtering, Slurm source-filtering, and resumability bookkeeping. The worker waits for the collector acknowledgement before the adapter call returns. This prevents a successful backend completion from overtaking an unacknowledged record, but it adds an actor round trip to every enabled adapter attempt.

When reporting is enabled, the new invocation record is not attached to output tasks. When it is disabled, the existing main-branch behavior remains: each output task receives its `StagePerfStats`, so zero-output calls remain absent and fan-out calls retain the existing copies.

Records describe successful attempts. They do not provide logical exactly-once accounting across backend retries.

## Current raw report shape

The following example is illustrative. Timestamps, durations, UUIDs, stage types, and record count depend on the actual run.

```json
{
  "schema_version": 1,
  "pipeline_name": "audio-example",
  "run_id": "73006a419f89498fa4243238095ba950",
  "executor": "RayDataExecutor",
  "pipeline": {
    "pipeline_name": "audio-example",
    "pipeline_description": "",
    "stages": [
      {
        "stage_id": "000:audio_transform",
        "name": "audio_transform",
        "type": "example.AudioTransformStage",
        "batch_size": 8,
        "num_workers": null
      },
      {
        "stage_id": "001:manifest_writer",
        "name": "manifest_writer",
        "type": "nemo_curator.stages.audio.common.ManifestWriterStage",
        "batch_size": 1,
        "num_workers": 1
      }
    ]
  },
  "slurm_array": null,
  "wall_time_s": 45.2,
  "record_count": 2,
  "records": [
    {
      "stage_name": "audio_transform",
      "stage_id": "000:audio_transform",
      "invocation_id": "77de2081ec534fbda4df8d755889f311",
      "process_time": 1.21,
      "actor_idle_time": 0.0,
      "num_items_processed": 8,
      "custom_metrics": {},
      "window_start_s": 1786400000.1,
      "window_end_s": 1786400001.31
    },
    {
      "stage_name": "manifest_writer",
      "stage_id": "001:manifest_writer",
      "invocation_id": "34ea8bcc4a314fa7aa80a834792a1cf9",
      "process_time": 0.01,
      "actor_idle_time": 0.08,
      "num_items_processed": 1,
      "custom_metrics": {},
      "window_start_s": 1786400001.4,
      "window_end_s": 1786400001.41
    }
  ]
}
```

The header includes the Curator-owned run ID, executor wall time, optional Slurm identity, and every concrete stage in built-plan order. `records` remains in collector arrival order, which can differ from plan order when workers run concurrently.

Each core record currently contains:

- `stage_name` and plan-order `stage_id`;
- a unique attempt-level `invocation_id`;
- stage-call `process_time` and epoch `window_start_s`/`window_end_s`;
- the existing `StageTimer` actor-idle and input-item values; and
- stage-defined `custom_metrics`.

The raw writer preserves additional JSON-serializable fields added by later producers. This foundation does not itself add input byte size, output cardinality, stage aggregation, GPU identity or utilization, hardware telemetry, or actor/worker identity.

## Inspect a report

Count records by stage without assuming arrival order:

```bash
jq '.records | group_by(.stage_id) | map({stage_id: .[0].stage_id, attempts: length})' \
  /output/qwen_omni_performance.json
```

List the built plan separately:

```bash
jq '.pipeline.stages | map({stage_id, name, type, batch_size, num_workers})' \
  /output/qwen_omni_performance.json
```

Do not infer serialized execution from array position. Use the plan metadata for stage order and the record windows for observed activity.

## Storage behavior

The collector keeps raw records in a driver-node temporary JSONL spool and the writer streams them into the final `records` array. Collection and finalization therefore do not load the complete record set into memory.

Local report output uses an atomic temporary-file replacement. Remote fsspec destinations stream directly and have the guarantees of their filesystem implementation. Install and configure the appropriate fsspec backend separately for any remote URL; this feature does not introduce a general cloud-filesystem dependency extra.

## Historical 16-row ASR parity evidence

The earlier GPU parity run below established transcript parity while this feature still emitted a compact, stage-grouped report. It is useful functional evidence for the accounting change, but its JSON is **not** the current raw `records` schema and should not be copied as a current output example.

| Item | Historical value |
|---|---|
| Target commit | `ab7081b325dfabc702a3c6642184c20c13633ac4` |
| Main reference commit | `a4470c6fe9b20ec98eb0839939c5e89de8aca3e5` |
| Input rows | 16 |
| ASR batch size | 8 |
| GPU | NVIDIA GeForce RTX 3080 Ti |
| Transcript comparison | 16 exact matches |
| Captured invocation count | 18: two ASR calls and sixteen writer calls |
| Historical public layout | `stage_performance` compact groups |
| Current public layout | raw top-level `records` array |

Because the implementation and schema changed after that run, this document does not present the historical JSON as output from the current branch. A new end-to-end run is required before publishing exact current-commit timings or UUIDs.

## Follow-up feature ownership

- PR #2223 can interpret these raw records into audio/Qwen summaries.
- PR #2262 can add backend, actor, worker, GPU, and hardware fields.

Those features are not emitted by this foundation alone.
