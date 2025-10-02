## NeMo-Curator Nightly Benchmarking

Run a matrix of benchmark scripts with per-entry Ray isolation, timeouts, environment capture, and pluggable metric sinks.

### Quick start

```bash
# Minimal
python -m nightly_benchmarking.run \
  --matrix nightly_benchmarking/matrix.yaml \
  --datasets nightly_benchmarking/dataset_paths.json

# With MLflow and a custom session name
MLFLOW_TRACKING_URI=http://your-mlflow:8265 \
python -m nightly_benchmarking.run \
  --matrix nightly_benchmarking/matrix.yaml \
  --datasets nightly_benchmarking/dataset_paths.json \
  --sink mlflow \
  --session-name dedup_removal
```

### How it works

- **Isolated runs**: For each matrix entry the driver starts a fresh Ray head, runs the command, then stops Ray.
- **Placeholders**: Command args can reference `{dataset:name,format}` and `{session}/...` paths.
- **Environment capture**: Saves `pip-freeze.txt`, `conda-explicit.txt` (if available), and `sys-env.json` per entry.
- **Results**: Scripts write standardized results; the driver aggregates and logs params/metrics to the chosen sink.

### Matrix YAML

Required top-level fields and supported options:

```yaml
results_dir: /path/to/results
default_timeout_s: 7200           # Optional; per-entry timeout overrides this
delete_scratch: true              # Optional; auto-delete {session}/scratch after run

# Optional sinks (passed through to the sink factory)
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}
  experiment: ray-curator-xyz
wandb: {}
slack: {}

entries:
  - name: xyz_id_xenna
    script: some_benchmark.py   # Resolved under nightly_benchmarking/scripts by default
    script_base_dir: nightly_benchmarking/scripts  # Optional override
    args: >-
      --input-path {dataset:dataset_name,jsonl}
      --output-path {session}/scratch
      --executor xenna
    timeout_s: 1800                # Optional per-entry timeout
    delete_scratch: true           # Optional per-entry override
    ray:
      num_cpus: 128
      num_gpus: 0
      enable_object_spilling: false
```

Notes:
- `${ENV_VAR}` in the YAML is resolved at load time and must be set; otherwise YAML loading fails.
- Only `script` is supported (no custom `cmd` field). The driver appends `--benchmark-results-path {session}/benchmark_results` to the command.

### Placeholders and datasets

- **`{dataset:name,format}`**: Resolved via `dataset_paths.json`.
- **`{session}/...`**: Expanded to the entry directory path.

Example `dataset_paths.json`:

```json
{
  "dataset_name": {
    "jsonl": "/path/to/dataset_json/",
    "parquet": "/path/to/dataset_parquet/"
  }
}
```

### Script contract (what your benchmark must do)

- Accept `--benchmark-results-path` (required).
- Write the following files to that directory:
  - `params.json`: your input parameters.
  - `metrics.json`: include at least `is_success: true|false` plus any metrics.
  - `tasks.pkl`: pickled task objects; the driver aggregates per-stage metrics (sum/mean/std).
- Exit with code `0` on success, non-zero on failure.

Minimal skeleton:

```python
import argparse, json, pickle
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-results-path", required=True)
    args = p.parse_args()
    out = Path(args.benchmark_results_path); out.mkdir(parents=True, exist_ok=True)
    Path(out/"params.json").write_text(json.dumps({}))
    Path(out/"metrics.json").write_text(json.dumps({"is_success": true}))
    with open(out/"tasks.pkl", "wb") as f: pickle.dump([], f)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

### CLI

```bash
python -m nightly_benchmarking.run \
  --matrix PATH_TO_YAML \
  --datasets PATH_TO_dataset_paths.json \
  [--sink {mlflow,wandb,none}] \
  [--session-name NAME]
```

### Output layout (per entry)

```
<results_dir>/<session_name>/<entry_name>/
├── scratch/                       # Temporary workspace (deleted if delete_scratch=true)
├── ray_cluster/                   # Ray debug artifacts
├── logs/                          # stdout/stderr and ray startup logs
│   ├── stdout.log
│   ├── stderr.log
│   └── ray_startup.log
├── artifacts/                     # Environment snapshots
│   ├── pip-freeze.txt
│   ├── conda-explicit.txt
│   └── sys-env.json
├── benchmark_results/             # Script outputs (required by contract)
│   ├── params.json
│   ├── metrics.json
│   └── tasks.pkl
└── results.json                   # Normalized driver result
```

### Troubleshooting

- Set appropriate `timeout_s`/`default_timeout_s` for long runs.
- Ensure `${ENV_VAR}` used in YAML is exported; otherwise config loading fails.
- If datasets fail to resolve, check names/formats in `dataset_paths.json`.
- Inspect `logs/stderr.log` and `logs/ray_startup.log` for failures.
