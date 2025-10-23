# NeMo Curator Benchmarking Framework

A comprehensive benchmarking framework for measuring and tracking the performance of NeMo Curator. This tool enables developers to ensure quality and performance by running standardized benchmark scripts in reproducible environments.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Writing Benchmark Scripts](#writing-benchmark-scripts)
- [Sinks: Custom Reporting & Actions](#sinks-custom-reporting--actions)
- [Docker Usage](#docker-usage)
- [Debugging](#debugging)
- [Examples](#examples)

---

## Overview

The NeMo Curator Benchmarking Framework helps developers:

- **Measure Performance**: Run benchmarks to measure execution time, throughput, and resource utilization
- **Track Quality**: Monitor performance trends over time through, receive notifications, and more through configurable sinks (MLflow, Slack, etc.)
- **Ensure Reproducibility**: Use Docker containers for consistent environments across runs
- **Automate Testing**: Integrate benchmarks into CI/CD pipelines
- **Compare Implementations**: Test different configurations, algorithms, or data processing strategies

The framework orchestrates benchmark scripts, collects metrics, manages environments, and delivers results through multiple reporting channels.

---

## Quick Start

### Using Docker (Recommended)

**1. Build the Docker image:**

```bash
cd benchmarking/tools
./build_docker.sh
```

**2. Create a simple configuration file** (e.g., `my_benchmark.yaml`):

```yaml
results_dir: "/benchmarking/results"

entries:
  - name: test_benchmark
    script: test_benchmark.py
    args: --iterations 100
    timeout_s: 300
```

**3. Run the benchmark:**

```bash
# Using the provided run.sh script
cd benchmarking/tools
./run.sh
```

Or manually:

```bash
docker run --rm \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/my_benchmark.yaml
```

**4. View results:**

Results are written to the `results_dir` specified in your configuration, organized by session timestamp.

---

## Core Concepts

### Session

A **session** represents a single invocation of the benchmarking framework. Each session:
- Has a unique name with timestamp (e.g., `nightly-run__2025-01-23__14-30-00`)
- Contains one or more benchmark entries
- Produces a session directory with results and artifacts
- Captures environment metadata (system info, package versions, etc.)

### Entry

An **entry** is a single benchmark task within a session. Each entry:
- Runs a specific benchmark script with defined arguments
- Has its own timeout, Ray configuration, and scratch directory
- Produces metrics, parameters, and task performance data
- Can reference datasets using template syntax

Example entry:
```yaml
- name: cc_benchmark
  script: common_crawl_benchmark.py
  args: --url_limit 100 --executor ray_data
  timeout_s: 3600
  ray:
    num_cpus: 64
    num_gpus: 1
```

### Scripts

**Benchmark scripts** are Python programs that:
- Reside in the `scripts/` directory
- Receive arguments from the framework (paths, parameters, etc.)
- Execute Curator operations and collect metrics
- Write standardized output files (params.json, metrics.json, tasks.pkl)

See [Writing Benchmark Scripts](#writing-benchmark-scripts) for details.

### Sinks

**Sinks** are pluggable modules that process benchmark results at various stages:
- Initialize at session start
- Process each entry's results
- Finalize at session end

Built-in sinks include:
- **MLflow**: Track experiments and metrics
- **Slack**: Post results to Slack channels
- **Google Drive**: Upload results to cloud storage (extensible)

See [Sinks: Custom Reporting & Actions](#sinks-custom-reporting--actions) for details.

### Tools

The `tools/` directory contains helper scripts:
- `build_docker.sh`: Build the benchmarking Docker image
- `run.sh`: Example script for running benchmarks in Docker with volume mounts

---

## Installation & Setup

### Option 1: Docker Container (Recommended)

**Build the image:**

```bash
cd benchmarking/tools
./build_docker.sh
```

This builds the `curator_benchmarking` image with:
- CUDA support
- All NeMo Curator dependencies
- Benchmarking framework and scripts
- Python 3.12 environment

**Verify the build:**

```bash
docker images | grep curator_benchmarking
```

### Option 2: Bare Metal (Local Python Environment)

**Requirements:**
- Python 3.10+
- CUDA 12.x (for GPU operations)
- NeMo Curator installed with all extras

**Setup:**

```bash
# Install NeMo Curator with all dependencies
cd <curator_repo_root>
uv sync --extra all --all-groups

# Install additional benchmarking dependencies
uv add rich loguru

# Run directly
python benchmarking/run.py --config benchmarking/config.yaml
```

**Note**: Bare metal requires manually managing dependencies and environment consistency.

---

## Configuration

### YAML Configuration Files

The framework uses one or more YAML files to configure benchmark sessions. Multiple configuration files are merged, allowing separation of concerns (e.g., machine-specific paths vs. benchmark definitions).

### Configuration Structure

```yaml
# Required: Where to write results
results_dir: "/path/to/results"

# Optional: Where to write large artifacts (logs, scratch data)
artifacts_dir: "/path/to/artifacts"  # Defaults to results_dir/artifacts

# Optional: Global timeout for all entries (seconds)
default_timeout_s: 7200

# Optional: Delete scratch directories after each entry completes
delete_scratch: true

# Optional: Configure sinks for result processing
sinks:
  - name: mlflow
    tracking_uri: ${MLFLOW_TRACKING_URI}
    experiment: my-experiment
    enabled: true
  - name: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
    enabled: true

# Optional: Define datasets for template substitution
datasets:
  - name: common_crawl
    formats:
      - type: json
        path: /data/cc_sample.jsonl
      - type: parquet
        path: /data/cc_sample.parquet

# Required: List of benchmark entries to run
entries:
  - name: my_benchmark
    script: my_script.py
    args: >-
      --input {dataset:common_crawl,parquet}
      --output {session_entry_dir}/output
    timeout_s: 1800
    ray:
      num_cpus: 32
      num_gpus: 1
    delete_scratch: false  # Override global setting
```

### Passing Configuration Files

**Multiple config files:**

```bash
python benchmarking/run.py \
  --config config.yaml \
  --config paths.yaml \
  --config machine_specific.yaml
```

Files are merged in order. Later files override earlier ones for conflicting keys.

**Session naming:**

```bash
python benchmarking/run.py \
  --config config.yaml \
  --session-name my-experiment-v2
```

### Environment Variables

Configuration values can reference environment variables:

```yaml
results_dir: "${HOME}/benchmarks/results"
sinks:
  - name: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
```

### Template Substitution

**Dataset references** in entry arguments:

```yaml
args: --input {dataset:common_crawl,parquet}
```

Resolves to the path defined in the `datasets` section.

**Session entry directory** for output paths:

```yaml
args: --output {session_entry_dir}/results
```

Resolves to the entry's unique directory within the session (e.g., `/results/session-name/entry-name/results`).

---

## Writing Benchmark Scripts

### Script Location

Benchmark scripts should be placed in the `benchmarking/scripts/` directory or specify a custom location:

```yaml
entries:
  - name: my_benchmark
    script: my_benchmark.py  # Looks in benchmarking/scripts/
    script_base_dir: /custom/path  # Optional override
```

### Required Script Interface

#### 1. Accept Framework Arguments

Your script must accept the `--benchmark-results-path` argument (automatically passed by the framework):

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    # Add your custom arguments
    parser.add_argument("--input", type=str)
    parser.add_argument("--iterations", type=int, default=100)
    
    args = parser.parse_args()
    
    # Your benchmark logic here
    run_benchmark(args)

if __name__ == "__main__":
    main()
```

#### 2. Generate Required Output Files

Your script **must** write three files to `--benchmark-results-path`:

**a) `params.json`** - Parameters used in the benchmark:

```python
import json

params = {
    "input_path": str(args.input),
    "iterations": args.iterations,
    "executor": "ray_data",
    "num_workers": 64
}

with open(args.benchmark_results_path / "params.json", "w") as f:
    json.dump(params, f, indent=2)
```

**b) `metrics.json`** - Measured metrics:

```python
metrics = {
    "execution_time_s": 123.45,
    "throughput_mb_s": 456.78,
    "rows_processed": 1000000,
    "peak_memory_gb": 32.5
}

with open(args.benchmark_results_path / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

**c) `tasks.pkl`** - Task performance data (using NeMo Curator's TaskPerfUtils):

```python
import pickle
from nemo_curator.tasks import Task
from nemo_curator.tasks.utils import TaskPerfUtils

# Wrap your operations in Task objects
with Task("data_loading", TaskPerfUtils()):
    # Your data loading code
    df = load_data(args.input)

with Task("processing", TaskPerfUtils()):
    # Your processing code
    result = process_data(df)

# Save all tasks
tasks = Task.get_all_tasks()
with open(args.benchmark_results_path / "tasks.pkl", "wb") as f:
    pickle.dump(tasks, f)
```

### Script Template

```python
#!/usr/bin/env python3
"""My benchmark script for NeMo Curator."""

import argparse
import json
import pickle
import time
from pathlib import Path

from nemo_curator.tasks import Task
from nemo_curator.tasks.utils import TaskPerfUtils


def run_benchmark(args):
    """Main benchmark logic."""
    start_time = time.time()
    
    # Your benchmark code here
    with Task("my_operation", TaskPerfUtils()):
        result = perform_operation(args.input)
    
    execution_time = time.time() - start_time
    
    # Write required output files
    params = {
        "input": str(args.input),
        "parameter1": args.param1,
    }
    with open(args.benchmark_results_path / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    metrics = {
        "execution_time_s": execution_time,
        "items_processed": len(result),
    }
    with open(args.benchmark_results_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    tasks = Task.get_all_tasks()
    with open(args.benchmark_results_path / "tasks.pkl", "wb") as f:
        pickle.dump(tasks, f)


def main():
    parser = argparse.ArgumentParser(description="My benchmark")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--param1", type=str, default="default")
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
```

### Example Scripts

See existing scripts in `scripts/`:
- `test_benchmark.py` - Simple demo benchmark
- `common_crawl_benchmark.py` - Common Crawl processing
- `removal_benchmark.py` - Data removal logic benchmark

---

## Sinks: Custom Reporting & Actions

### Overview

Sinks extend the framework to perform custom actions at various stages of the benchmark lifecycle:

1. **Initialize**: Called once at session start with session metadata
2. **Process Result**: Called after each entry completes with that entry's results
3. **Finalize**: Called once at session end to perform final actions

### Built-in Sinks

#### MLflow Sink

Tracks experiments and metrics in MLflow:

```yaml
sinks:
  - name: mlflow
    tracking_uri: http://mlflow-server:5000
    experiment: my-experiment
    enabled: true
```

#### Slack Sink

Posts results to Slack channels:

```yaml
sinks:
  - name: slack
    webhook_url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
    enabled: true
```

Results are formatted as interactive Slack messages with environment info and metrics.

#### Google Drive Sink

Placeholder for uploading results to Google Drive:

```yaml
sinks:
  - name: gdrive
    enabled: false
```

### Writing a Custom Sink

**1. Create a new sink class** in `runner/sinks/`:

```python
# runner/sinks/my_custom_sink.py
from typing import Any
from loguru import logger
from runner.sinks.sink import Sink


class MyCustomSink(Sink):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.enabled = config.get("enabled", True)
        self.api_endpoint = config.get("api_endpoint")
        
        # Initialize any resources
        if not self.api_endpoint:
            raise ValueError("MyCustomSink: api_endpoint is required")
    
    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None:
        """Called at session start."""
        self.session_name = session_name
        self.env_data = env_data
        
        if self.enabled:
            logger.info(f"MyCustomSink: Starting session {session_name}")
            # Perform initialization (e.g., create remote session)
    
    def process_result(self, result: dict[str, Any]) -> None:
        """Called after each entry completes."""
        if self.enabled:
            logger.info(f"MyCustomSink: Processing {result['name']}")
            # Send result to your API, database, etc.
            self._send_to_api(result)
    
    def finalize(self) -> None:
        """Called at session end."""
        if self.enabled:
            logger.info("MyCustomSink: Finalizing session")
            # Perform cleanup, send summary, etc.
    
    def _send_to_api(self, data: dict) -> None:
        """Helper method for API calls."""
        # Your implementation
        pass
```

**2. Register your sink** in `runner/matrix.py`:

```python
@classmethod
def load_sinks(cls, sink_configs: list[dict]) -> list[Sink]:
    sinks = []
    for sink_config in sink_configs:
        sink_name = sink_config["name"]
        if sink_name == "my_custom":
            from runner.sinks.my_custom_sink import MyCustomSink
            sinks.append(MyCustomSink(config=sink_config))
        # ... other sinks ...
    return sinks
```

**3. Use in configuration:**

```yaml
sinks:
  - name: my_custom
    api_endpoint: https://api.example.com/benchmarks
    enabled: true
```

### Result Data Structure

Results passed to `process_result()` contain:

```python
{
    "name": "entry_name",
    "success": True,
    "exec_time_s": 123.45,
    "timeout": False,
    "script_params": { ... },  # From params.json
    "script_metrics": { ... },  # From metrics.json
    "tasks": [ ... ],  # From tasks.pkl
    "command": "python script.py ...",
    "returncode": 0,
    "stdouterr_file": "/path/to/log.txt"
}
```

---

## Docker Usage

### Building the Image

Use the provided build script:

```bash
cd benchmarking/tools
./build_docker.sh
```

Or manually:

```bash
docker build \
  -f benchmarking/Dockerfile \
  --target curator_benchmarking \
  -t curator_benchmarking \
  /path/to/curator/repo
```

### Running with Docker

#### Basic Usage

```bash
docker run --rm \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml
```

#### GPU Support

```bash
docker run --rm \
  --gpus all \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml
```

Or specify specific GPUs:

```bash
docker run --rm \
  --gpus '"device=0,1"' \
  ...
```

### Volume Mounts for Development

#### Mount Local Curator Source

Test changes to Curator code without rebuilding the image:

```bash
docker run --rm \
  --volume /host/path/to/curator:/opt/Curator \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml
```

This allows you to:
- Edit Curator source code on the host
- Run benchmarks in the container using your modified code
- Iterate quickly without rebuilding

#### Mount Datasets

Provide access to datasets on the host:

```bash
docker run --rm \
  --volume /host/datasets:/datasets:ro \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml
```

Use `:ro` for read-only access to prevent accidental modification.

#### Mount Configuration Files

Keep configuration on the host:

```bash
docker run --rm \
  --volume $(pwd)/my_config.yaml:/config/my_config.yaml:ro \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking \
    --config /config/my_config.yaml
```

#### Mount Results and Artifacts

Write results back to the host:

```bash
docker run --rm \
  --volume $(pwd)/results:/benchmarking/results \
  --volume $(pwd)/artifacts:/benchmarking/artifacts \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml
```

Configuration should reference these paths:

```yaml
results_dir: /benchmarking/results
artifacts_dir: /benchmarking/artifacts
```

### Complete Example with Multiple Mounts

The `tools/run.sh` script demonstrates a full setup:

```bash
docker run \
  --gpus='"device=1"' \
  --rm \
  -it \
  --volume /host/datasets:/datasets:ro \
  --volume /host/results:/benchmarking/results \
  --volume /host/artifacts:/benchmarking/artifacts \
  --volume /host/curator:/opt/Curator \
  --env=MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
  --env=SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
  curator_benchmarking \
    --config=/opt/Curator/benchmarking/config.yaml \
    --config=/opt/Curator/benchmarking/paths.yaml
```

### Environment Variables

Pass environment variables for configuration:

```bash
docker run --rm \
  --env MLFLOW_TRACKING_URI=http://mlflow:5000 \
  --env SLACK_WEBHOOK_URL=https://hooks.slack.com/... \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml
```

---

## Debugging

### Interactive Shell in Container

Override the entrypoint to get a bash shell:

```bash
docker run --rm -it \
  --entrypoint /bin/bash \
  --volume /host/curator:/opt/Curator \
  --volume $(pwd)/results:/benchmarking/results \
  curator_benchmarking
```

Inside the container, you can:

```bash
# Explore the environment
ls /opt/Curator/benchmarking/

# Run benchmarks manually
python /opt/Curator/benchmarking/run.py \
  --config /opt/Curator/benchmarking/config.yaml

# Run individual scripts
python /opt/Curator/benchmarking/scripts/test_benchmark.py \
  --benchmark-results-path /tmp/test

# Check Python environment
python -c "import nemo_curator; print(nemo_curator.__version__)"

# Debug with interactive Python
ipython
```

### Debugging Benchmark Scripts

**Run a script standalone:**

```bash
# Create a temporary results directory
mkdir -p /tmp/benchmark_results

# Run the script directly
python benchmarking/scripts/my_script.py \
  --benchmark-results-path /tmp/benchmark_results \
  --input /data/test.parquet \
  --iterations 10

# Check outputs
ls -la /tmp/benchmark_results/
cat /tmp/benchmark_results/params.json
cat /tmp/benchmark_results/metrics.json
```

**Use Python debugger:**

Add breakpoints to your script:

```python
import pdb; pdb.set_trace()
```

Or use `ipdb` for a better experience:

```python
import ipdb; ipdb.set_trace()
```

### Viewing Logs

**Framework logs** are written to stdout/stderr and captured in the session directory.

**Entry logs** (stdout/stderr from benchmark scripts) are saved to:

```
{results_dir}/{session_name}/{entry_name}/stdouterr.log
```

**View logs in real-time:**

```bash
tail -f results/my-session__2025-01-23__14-30-00/my_entry/stdouterr.log
```

### Common Issues

**Issue: "Module not found" errors**

- Ensure `PYTHONPATH` includes the benchmarking directory
- In Docker, verify the Curator installation with `uv sync`

**Issue: Permission denied when writing results**

- Check volume mount permissions
- Ensure the results directory is writable by the container user

**Issue: Timeout errors**

- Increase `timeout_s` in entry configuration
- Check `default_timeout_s` in global configuration
- Monitor resource usage (CPU, memory, GPU)

**Issue: Out of memory**

- Adjust Ray configuration (`num_cpus`, `object_spilling`)
- Reduce dataset size or batch size
- Monitor memory usage with `nvidia-smi` (GPU) or `top` (CPU)

---

## Examples

### Example 1: Simple Benchmark

**config.yaml:**

```yaml
results_dir: "./results"
entries:
  - name: quick_test
    script: test_benchmark.py
    args: --iterations 50
    timeout_s: 300
```

**Run:**

```bash
python benchmarking/run.py --config config.yaml
```

### Example 2: Multiple Benchmarks with Datasets

**config.yaml:**

```yaml
results_dir: "./results"
datasets:
  - name: sample_data
    formats:
      - type: parquet
        path: /data/sample.parquet

entries:
  - name: benchmark_v1
    script: my_benchmark.py
    args: --input {dataset:sample_data,parquet} --algorithm v1
    
  - name: benchmark_v2
    script: my_benchmark.py
    args: --input {dataset:sample_data,parquet} --algorithm v2
```

### Example 3: Production Setup with Sinks

**config.yaml:**

```yaml
results_dir: "/benchmarks/results"
default_timeout_s: 7200

sinks:
  - name: mlflow
    tracking_uri: http://mlflow.example.com:5000
    experiment: nightly-benchmarks
    enabled: true
  - name: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
    enabled: true

datasets:
  - name: cc_large
    formats:
      - type: parquet
        path: /data/common_crawl_large.parquet

entries:
  - name: cc_extraction
    script: common_crawl_benchmark.py
    args: >-
      --download_path {session_entry_dir}/scratch/downloads
      --output_path {session_entry_dir}/scratch/output
      --output_format parquet
      --url_limit 10000
    timeout_s: 14400
    ray:
      num_cpus: 128
      num_gpus: 8
```

**Run with Docker:**

```bash
docker run --rm \
  --gpus all \
  --volume /data:/data:ro \
  --volume /benchmarks:/benchmarks \
  --env SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
  curator_benchmarking \
    --config /opt/Curator/benchmarking/config.yaml \
    --session-name nightly-run
```

---

## Additional Resources

### Directory Structure

```
benchmarking/
├── README.md              # This file
├── run.py                 # Main framework entry point
├── config.yaml            # Example configuration
├── paths.yaml             # Example paths configuration
├── Dockerfile             # Container definition
├── scripts/               # Benchmark scripts
│   ├── test_benchmark.py
│   ├── common_crawl_benchmark.py
│   └── removal_benchmark.py
├── runner/                # Framework modules
│   ├── matrix.py          # Configuration and entry management
│   ├── datasets.py        # Dataset resolution
│   ├── process.py         # Process execution with timeout
│   ├── env_capture.py     # Environment metadata capture
│   ├── utils.py           # Utility functions
│   └── sinks/             # Result processing sinks
│       ├── sink.py        # Base sink class
│       ├── mlflow_sink.py
│       ├── slack_sink.py
│       └── gdrive_sink.py
└── tools/                 # Helper scripts
    ├── build_docker.sh    # Build Docker image
    └── run.sh             # Example run script
```

### Tips & Best Practices

1. **Use Version Control**: Track configuration files and scripts in git
2. **Parameterize Paths**: Use environment variables and separate path configs
3. **Start Small**: Test with small datasets before running full benchmarks
4. **Monitor Resources**: Watch CPU, memory, and GPU usage during runs
5. **Clean Up Scratch**: Enable `delete_scratch` to avoid filling disk
6. **Tag Docker Images**: Use tags for reproducibility (`curator_benchmarking:v1.0`)
7. **Document Changes**: Add comments to configuration explaining non-obvious settings
8. **Test Locally First**: Verify scripts work before containerizing

### Getting Help

- Check logs in `{results_dir}/{session_name}/`
- Review script outputs in `{session_name}/{entry_name}/stdouterr.log`
- Use `--help` flag: `python run.py --help`
- Inspect results: `cat results/session/entry/params.json`

---

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0. See the main repository LICENSE file for details.

