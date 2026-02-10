# NeMo Curator Benchmarking Framework

A comprehensive benchmarking framework for measuring and tracking the performance of NeMo Curator. This tool enables developers to ensure quality and performance by running standardized benchmark scripts in reproducible environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [Running benchmarks and using the container](#running-benchmarks-and-using-the-container)
- [Writing Benchmark Scripts](#writing-benchmark-scripts)
- [Sinks: Custom Reporting & Actions](#sinks-custom-reporting--actions)

---

## Quick Start

**1. Build the Docker image:**

Assuming the working directory is the NeMo Curator repo root dir:
```bash
./benchmarking/tools/build_docker.sh
```

This builds the `curator_benchmarking` image with:
- CUDA support
- Python 3.12 environment
- NeMo Curator from source in repo root dir
- All NeMo Curator dependencies
- Benchmarking framework and scripts

Note: you may only need to do this periodically when the environment needs to be updated. See the `--use-host-curator` example below.

**2. Update config:**

Update `results_path` and `datasets_path` in the YAML config file based on your preferences. In this example, we'll edit the YAML config `./benchmarking/nightly-benchmark.yaml`

```yaml
results_path: /path/where/results/are/stored
datasets_path: /path/to/datasets
```

**3. Run benchmarks:**

```bash
./benchmarking/tools/run.sh --config ./benchmarking/nightly-benchmark.yaml
```

To run using the Curator sources on the host instead of those in the image, pass the `--use-host-curator` option:
```bash
./benchmarking/tools/run.sh --config ./benchmarking/nightly-benchmark.yaml --use-host-curator
```
This is especially useful during active development and debugging since it avoids a costly rebuild step.


**4. View results:**

Results are written to the `results_path` specified in your configuration, organized by session timestamp.

---

## Concepts

### Session

A **session** represents a single invocation of the benchmarking framework. Each session:
- Has a unique name with timestamp (e.g., `benchmark-run__2025-01-23__14-30-00`)
- Contains one or more benchmark entries
- Produces a session directory with results
- Captures environment metadata (system info, package versions, etc.)

### Scripts

**Benchmark scripts** are Python programs that:
- Reside in the `scripts/` directory
- Receive arguments from the framework (paths, parameters, etc.)
- Execute Curator operations and collect metrics
- Write standardized output files (params.json, metrics.json, tasks.pkl)
- Can be run standalone outside of the benchmark framework to debug problems, perform useful work, or be used as examples.
- Can be written by users to benchmark specific use cases.
- Are referenced in the YAML configuration as "entries" to be included in benchmark runs with specific options.

See [Writing Benchmark Scripts](#writing-benchmark-scripts) for details.

### Entry

An **entry** is a single benchmark run within a session. Each entry:
- Runs a specific benchmark script with defined arguments
- Has its own timeout, Ray configuration, sink configuration, pass/fail requirments, or can inherit from session-wide defaults
- Produces metrics, parameters, and run performance data
- Can reference datasets using template syntax
- Can pass additional data to sinks to provide for customized operations unique to the entry. For example, the `slack_sink` can accept additional metrics to report for an entry that other entries may not have.
- Can specify specific requirements that must be met in order to return a passing status. For example, an entry can require that a specific throughput metric meet or exceed a minimum value.

### Sinks

**Sinks** are pluggable modules that are called by the framework at various stages to allow for custom processing of benchmark data:
- Initialize at session start
- Process each entry's individual benchmark results
- Finalize at session end

Built-in sinks include:
- **Slack**: Post results to Slack channels
- **Google Drive**: Upload results to cloud storage (extensible)
- **MLflow**: Track experiments and metrics

See [Sinks: Custom Reporting & Actions](#sinks-custom-reporting--actions) for details.

## Configuration

### YAML Configuration Files

The framework uses one or more YAML files to configure benchmark sessions. Multiple configuration files are merged, allowing separation of concerns (e.g., machine-specific paths vs. benchmark definitions).

A useful pattern is to use multiple YAML files, where configuration that does not typically change is in one or more files, and user or machine-specific configuration is others.  For example, `my_paths_and_reports.yaml` could have results / datasets paths and personal sink settings (individual slack channel, etc.), and `release-benchmarks.yaml` could have the team-wide configuration containing the individual benchmark entries and performance requirements.

This can be especially useful during development. During development you'll not only want to use your own paths and report settings, you'll also want to use the standard benchmarking environment (i.e. a container), but cannot afford to rebuild the Docker image for each code change you're evaluating. The `--use-host-curator` flag is intended for this case. This flag will use your Curator source dir on host inside the container via a volume mount (this works because the container has curator installed in editable mode), and no image rebuild step is needed.

An example of a development scenario using this pattern looks like this:
```bash
./benchmarking/tools/run.sh --use-host-curator --config ~/curator_benchmarking/my_paths_and_reports.yaml --config ./benchmarking/release-benchmarks.yaml
```

### Configuration Structure

```yaml
# Required: Base paths for results and datasets
# These paths must exist on the host machine
# When running in Docker with tools/run.sh, paths are automatically mapped to container volumes
# These base paths can be referenced in other configuration values using {results_path}, {datasets_path}
results_path: /path/to/results
datasets_path: /path/to/datasets

# Optional: Global timeout for all entries (seconds)
default_timeout_s: 7200

# Optional: Delete scratch directories after each entry completes
# The path {session_entry_dir}/scratch is automatically created when an entry starts and can be used by benchmark
#scripts for writing temp files. This directory is automatically cleaned up on completion of the entry if
# delete_scratch is true.
delete_scratch: true

# Optional: Configure sinks for result processing
sinks:
  - name: mlflow
    enabled: true
    tracking_uri: ${MLFLOW_TRACKING_URI}
    experiment: my-experiment
  - name: slack
    enabled: true
    channel_id: ${SLACK_CHANNEL_ID}
    default_metrics: ["exec_time_s"]  # Metrics to report by default for all entries
  - name: gdrive
    enabled: false
    drive_folder_id: ${GDRIVE_FOLDER_ID}
    service_account_file: ${GDRIVE_SERVICE_ACCOUNT_FILE}

# Optional: Define datasets for template substitution
datasets:
  - name: common_crawl
    formats:
      - type: json
        path: "{datasets_path}/cc_sample"  # Can reference base paths
      - type: parquet
        path: "{datasets_path}/cc_sample"

# Required: List of benchmark entries to run
entries:
  - name: my_benchmark
    enabled: true  # Optional: Whether to run this entry (default: true)
    script: my_script.py
    args: >-
      --input {dataset:common_crawl,parquet}
      --output {session_entry_dir}/output
    timeout_s: 1800  # Optional: Override global timeout

    # Optional: Per-entry sink configuration
    sink_data:
      - name: slack
        additional_metrics: ["throughput_docs_per_sec", "num_documents_processed"]

    # Optional: Ray configuration for this entry
    ray:
      num_cpus: 32
      num_gpus: 1
      enable_object_spilling: false

    # Optional: Requirements for the benchmark to pass
    requirements:
      - metric: throughput_docs_per_sec
        min_value: 100

    # Optional: Override global delete_scratch setting
    delete_scratch: false
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

Configuration values can reference environment variables using `${VAR_NAME}` syntax:

```yaml
results_path: "${HOME}/benchmarks/results"
sinks:
  - name: slack
    channel_id: ${SLACK_CHANNEL_ID}
  - name: mlflow
    tracking_uri: ${MLFLOW_TRACKING_URI}
```

### Template Substitution and Path Resolution

The framework supports several types of placeholders in configuration values:

**Base path references** - Reference the configured base paths:

```yaml
datasets:
  - name: my_dataset
    formats:
      - type: parquet
        path: "{datasets_path}/subdir/data.parquet"
```

Available base path placeholders:
- `{results_path}` - Resolves to the configured `results_path`
- `{datasets_path}` - Resolves to the configured `datasets_path`

**Dataset references** - Reference datasets in entry arguments:

```yaml
args: --input {dataset:common_crawl,parquet}
```

Resolves to the path defined in the `datasets` section for that dataset and format.

**Session entry directory** - Reference the entry's runtime directory:

```yaml
args: --output {session_entry_dir}/results
```

Resolves to the entry's unique directory within the session (e.g., `/results/session-name__timestamp/entry-name/results`).

### Entry Configuration Details

**enabled**: Controls whether an entry is run (default: `true`). Useful for temporarily disabling entries without removing them from the configuration.

**sink_data**: Provides entry-specific configuration for sinks. For example, the Slack sink can accept `additional_metrics` to report metrics beyond the default set:

```yaml
sink_data:
  - name: slack
    additional_metrics: ["num_documents_processed", "throughput_docs_per_sec"]
```

**requirements**: Defines pass/fail criteria for the benchmark. If any requirement is not met, the entry is marked as failed:

```yaml
requirements:
  - metric: throughput_docs_per_sec
    min_value: 100
  - metric: peak_memory_gb
    max_value: 64
```

**ray**: Configures Ray resources for the entry:

```yaml
ray:
  num_cpus: 64
  num_gpus: 4
  enable_object_spilling: false  # Disable object spilling to local disk
```

---

## Running benchmarks and using the container

The `benchmarking/tools/run.sh` script provides a convenient way to run benchmarks in a Docker container with proper volume mounts, GPU access, and environment configuration.

### Basic Usage

Run benchmarks using a configuration file:

```bash
./benchmarking/tools/run.sh --config benchmarking/my-benchmark.yaml
```

This command:
- Reads the configuration file and extracts `results_path` and `datasets_path`
- Automatically creates volume mounts to map these paths into the container
- Runs the benchmarking framework with the Curator code built into the Docker image
- Passes environment variables like `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`, and `MLFLOW_TRACKING_URI` to the container

### Using Host Curator Sources

To run benchmarks using Curator source code from your local repository instead of the version built into the image:

```bash
./benchmarking/tools/run.sh --use-host-curator --config benchmarking/my-benchmark.yaml
```

This mounts your local Curator repository (from `$HOST_CURATOR_DIR`) into the container at `/opt/Curator`, allowing you to:
- Test local changes without rebuilding the Docker image
- Quickly iterate on Curator development
- Debug issues with modified source code

The `HOST_CURATOR_DIR` environment variable defaults to the repository root but can be overridden:

```bash
HOST_CURATOR_DIR=/path/to/my/curator/fork ./benchmarking/tools/run.sh --use-host-curator --config my-benchmark.yaml
```

### Interactive Shell

Get an interactive bash shell in the container environment:

```bash
./benchmarking/tools/run.sh --shell
```

This is useful for:
- Exploring the container environment
- Running benchmarks manually for debugging
- Checking installed packages and versions
- Testing commands before adding them to scripts

### Running Commands in the Container

Execute a specific command in the container without an interactive shell:

```bash
./benchmarking/tools/run.sh --shell "uv pip list"
```

This runs the command and exits. Examples:

```bash
# Check installed packages
./benchmarking/tools/run.sh --shell "uv pip list | grep curator"

# Verify Python environment
./benchmarking/tools/run.sh --shell "python -c 'import nemo_curator; print(nemo_curator.__version__)'"

# List available benchmark scripts
./benchmarking/tools/run.sh --shell "ls -l /opt/Curator/benchmarking/scripts/"
```

### Controlling GPU Access

Use the `GPUS` environment variable to control which GPUs are visible to the container:

```bash
# Use all GPUs (default)
./benchmarking/tools/run.sh --config my-benchmark.yaml

# Use specific GPUs
GPUS="device=0,1" ./benchmarking/tools/run.sh --config my-benchmark.yaml

# Use only GPU 2
GPUS="device=2" ./benchmarking/tools/run.sh --config my-benchmark.yaml

# Run without GPU access
GPUS="none" ./benchmarking/tools/run.sh --config my-benchmark.yaml
```

The `GPUS` value is passed directly to Docker's `--gpus` flag.

### More details
For more details, refer to the `--help` output for `run.sh`
```bash
./benchmarking/tools/run.sh --help
```

---

## Writing Benchmark Scripts

### Script Location

Benchmark scripts should be placed in the `benchmarking/scripts/` directory. Scripts are referenced by filename in the YAML configuration.

### Required Script Interface

Benchmark scripts must follow these requirements:

#### 1. Accept Framework Arguments

Your script must accept the `--benchmark-results-path` argument. This is automatically passed by the framework and specifies the directory where output files should be written. You can add any additional custom arguments your benchmark needs.

#### 2. Generate Required Output Files

Your script **must** write three JSON/pickle files to the `--benchmark-results-path` directory:

**`params.json`** - A JSON file containing all parameters used in the benchmark run (input paths, configuration options, etc.). This allows for reproducibility and tracking of what settings were used.

**`metrics.json`** - A JSON file containing all measured metrics from the benchmark (execution time, throughput, memory usage, etc.). Metric names used here can be referenced in entry requirements and sink configurations.

**`tasks.pkl`** - A pickle file containing NeMo Curator `Task` objects that capture detailed performance data. Use `nemo_curator.tasks.Task` with `TaskPerfUtils()` to wrap operations in your script, then save all tasks using `Task.get_all_tasks()`.

### Reference Implementations

See existing scripts in `scripts/` for complete examples:
- `domain_classification_benchmark.py` - Domain classification with model inference
- `embedding_generation_benchmark.py` - Embedding generation benchmark
- `removal_benchmark.py` - Data removal operations benchmark

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
    channel_id: C1234567890  # Your Slack channel ID
    enabled: true
```

Results are posted as interactive Slack messages with environment info and metrics. Requires:
- `SLACK_BOT_TOKEN` environment variable set to your Slack Bot User OAuth Token
- `SLACK_CHANNEL_ID` in config or environment variable for the target channel

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

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0. See the main repository LICENSE file for details.
