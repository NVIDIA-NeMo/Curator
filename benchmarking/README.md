# NeMo Curator Benchmarking Framework

A comprehensive benchmarking framework for measuring and tracking the performance of NeMo Curator. This tool enables developers to ensure quality and performance by running standardized benchmark scripts in reproducible environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Nightly Benchmark Ownership](#nightly-benchmark-ownership)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [Running benchmarks and using the container](#running-benchmarks-and-using-the-container)
- [Audio Benchmark Data Setup](#audio-benchmark-data-setup)
- [Writing Benchmark Scripts](#writing-benchmark-scripts)
- [Sinks: Custom Reporting & Actions](#sinks-custom-reporting--actions)

---

## Quick Start

**1. Build the Docker image:**

Assuming the working directory is the NeMo Curator repo root dir:
```bash
./benchmarking/tools/build_docker.sh --tag-as-latest
```

This builds the `curator_benchmarking` image with:
- CUDA support
- Python 3.12 environment
- NeMo Curator from source in repo root dir
- All NeMo Curator dependencies
- Benchmarking framework and scripts

Note: you may only need to do this periodically when the environment needs to be updated. See the `--use-host-curator` example below.

**2. Update config:**

Update the `host_path` values in the `paths` section of the YAML config file based on your preferences. In this example, we'll edit the YAML config `./benchmarking/benchmarks.yaml`

```yaml
paths:
  - name: results_path
    host_path: /path/where/results/are/stored
  - name: datasets_path
    host_path: /path/to/datasets
    container_path: /datasets
  - name: model_weights_path
    host_path: /path/to/model_weights
    container_path: /model_weights
```

Keep `model_weights_path` configured when running benchmarks that consume
pre-staged model snapshots or caches, such as audio tagging.

**3. Run benchmarks:**

```bash
./benchmarking/tools/run.sh \
  --config ./benchmarking/benchmarks.yaml \
  --config ./benchmarking/nightly-data-setup.yaml
```

For a 4-GPU, 64-CPU GB200 environment, layer the SKU override after the full-suite config:

```bash
./benchmarking/tools/run.sh \
  --config ./benchmarking/benchmarks.yaml \
  --config ./benchmarking/4xGB200-64CPU.yaml \
  --config ./benchmarking/nightly-data-setup.yaml
```

The 4xGB200-64CPU override updates resource counts, timeout values, known 4-GPU video
throughput thresholds, and workload-specific scaling settings. Other performance
requirements are inherited from `benchmarks.yaml` until 4xGB200-64CPU-specific
baselines are measured.

To run using the Curator sources on the host instead of those in the image, pass the `--use-host-curator` option:
```bash
./benchmarking/tools/run.sh \
  --config ./benchmarking/benchmarks.yaml \
  --config ./benchmarking/nightly-data-setup.yaml \
  --use-host-curator
```
This is especially useful during active development and debugging since it avoids a costly rebuild step.


**4. View results:**

Results are written to the `results_path` specified in your configuration, organized by session timestamp.

---

## Nightly Benchmark Ownership

Curator owns the benchmark workload: `benchmarking/benchmarks.yaml`,
the benchmark runner, benchmark scripts, data setup scripts, and local developer
tools such as `benchmarking/tools/run.sh`.

The scheduled nightly run is orchestrated outside of the Curator repository by
CI infrastructure. That pipeline reads Curator's
`benchmarking/benchmarks.yaml` plus any selected SKU override config, generates
one scheduler job per enabled entry from the merged config, and starts each job
in a benchmark runtime environment.

Each generated job invokes Curator's `benchmarking/run.py` for its assigned
entry. The jobs share a session name and results root so their per-entry outputs
are collected as one logical nightly benchmark session. The CI layer also
provides environment-specific path overrides, such as mapping the public
benchmark config's logical dataset and results paths to the storage locations
available in that runtime environment.

CI-only files that control job generation, path mapping, and runtime launch
behavior live in a CI orchestration repository outside Curator. Keeping those
files out of Curator lets benchmark infrastructure change independently of the
Curator source ref or prebuilt Curator image being benchmarked, which is
important for release-candidate and historical-image runs.

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
# Required: Paths to files and directories used by the benchmarks.
# Each entry must have a "name" and a "host_path". The name can be referenced elsewhere
# in the config using {name} placeholders (e.g. {datasets_path}).
# When running in Docker with tools/run.sh, each path is automatically mounted into the
# container. An optional "container_path" overrides the default mount point
# (which is the host_path prefixed with "/MOUNT").
# An entry with name "results_path" is required.
paths:
  - name: results_path
    host_path: /path/to/results
  - name: datasets_path
    host_path: /path/to/datasets
    container_path: /datasets  # optional override
  - name: model_weights_path
    host_path: /path/to/model_weights
    container_path: /model_weights  # optional override

# Optional: Global timeout for entries that omit timeout_s (seconds)
default_timeout_s: 7200

# Optional: Maximum allowed effective timeout for any entry (seconds).
# Defaults to 14340 (3h59m).
max_timeout_s: 14340

# Optional: Free-text reason for the run, persisted in env.json and surfaced to sinks.
run_reason: "26.06 RC7 benchmarks"

# Optional: Resolved benchmark viewer URL, persisted in env.json and surfaced to sinks.
# Set either viewer_url or viewer_url_template, not both.
viewer_url: "http://viewer.example.com/run-viewer?dir=/path/to/results/session"

# Optional: Benchmark viewer URL template. Used when viewer_url is not set, and
# rendered after the session name/path are known. Supported placeholders are:
# {results_path}, {results_path_url}, {session_name}, {session_name_url},
# {session_path}, and {session_path_url}. The *_url forms are URL-encoded.
viewer_url_template: "http://viewer.example.com/run-viewer?dir={results_path_url}&run={session_name_url}"

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

# Optional: Global Ray settings inherited by all entries; per-entry ray sections override these values
ray:
  num_cpus: 64
  num_gpus: 8
  enable_object_spilling: false

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

Files are merged in order using a deep recursive merge, so later files can override or extend specific nested values without replacing entire top-level keys. `benchmarking/benchmarks.yaml` is the complete full-suite reference config and is calibrated for the default 8-GPU H100 nightly environment. SKU-specific files such as `benchmarking/4xGB200-64CPU.yaml` should be passed after it to override only the values that differ for that environment.

**Merge behavior:**
- **Scalar values** (strings, numbers, booleans): later file wins.
- **Nested dicts**: merged recursively — only the keys present in the later file are updated.
- **Lists of dicts** (e.g. `entries`, `paths`, `requirements`, `sinks`): items are matched by their `name` key when present (the canonical identifier for most list items), falling back to the first key otherwise. If a matching item is found, it is merged recursively; if not, the item is appended. Use `name` in override files whenever possible to ensure reliable matching.

This makes it practical to write small override files that change only specific entries or requirements without duplicating the full configuration.

**Example — overriding a single entry's timeout and requirements:**

Base config (`benchmarks.yaml`) defines many entries including:
```yaml
entries:
  - name: domain_classification_xenna
    timeout_s: 1400
    requirements:
      - metric: throughput_docs_per_sec
        min_value: 3000
```

Override file (`my_overrides.yaml`) changes only that entry's timeout and requirement minimum:
```yaml
entries:
  - name: domain_classification_xenna
    timeout_s: 2000
    requirements:
      - metric: throughput_docs_per_sec
        min_value: 2000
```

Running with both files:
```bash
python benchmarking/run.py \
  --config benchmarks.yaml \
  --config my_overrides.yaml
```

Results in `domain_classification_xenna` using `timeout_s: 2000` and `min_value: 2000`, while all other entries remain unchanged.

**Session naming:**

```bash
python benchmarking/run.py \
  --config config.yaml \
  --session-name my-experiment-v2
```

**Benchmark viewer URL:**

To include a link to a benchmark run viewer in sinks such as Slack, pass a resolved URL with `--viewer-url`:

```bash
python benchmarking/run.py \
  --config config.yaml \
  --viewer-url "http://viewer.example.com/run-viewer?dir=/path/to/results/&run=my-session"
```

If part of the URL depends on the selected results path or session name, use `--viewer-url-template`. The template is rendered after the final session name and session path are known. When benchmarks run in a container with configured `host_path` / `container_path` mounts, path placeholders use the host-visible path so links work outside the container:

```bash
python benchmarking/run.py \
  --config config.yaml \
  --session-name my-session \
  --viewer-url-template "http://viewer.example.com/run-viewer?dir={results_path_url}&run={session_name_url}"
```

For a viewer that reads results from a remote host path, include the host in the template:

```bash
python benchmarking/run.py \
  --config config.yaml \
  --viewer-url-template "http://rratzel-ws1:5050/run-viewer?dir=dgx-a100-01%3A{results_path_url}%2F&run={session_name_url}"
```

Supported `--viewer-url-template` placeholders:

| Placeholder | Value |
| --- | --- |
| `{results_path}` | The configured results root directory, unmapped to the host-visible path when running in a container. |
| `{results_path_url}` | URL-encoded `results_path`. |
| `{session_name}` | The resolved session name, either from `--session-name` or the generated default. |
| `{session_name_url}` | URL-encoded `session_name`. |
| `{session_path}` | The full session result directory, equivalent to `{results_path}/{session_name}`, unmapped to the host-visible path when running in a container. |
| `{session_path_url}` | URL-encoded `session_path`. |

Use `results_path` when the viewer expects the results root and a separate `run` parameter. Use `session_path` when the viewer expects a single path directly to the session directory. Set either `viewer_url` or `viewer_url_template`, not both.

### Environment Variables

Configuration values can reference environment variables using `${VAR_NAME}` syntax:

```yaml
paths:
  - name: results_path
    host_path: "${HOME}/benchmarks/results"
sinks:
  - name: slack
    channel_id: ${SLACK_CHANNEL_ID}
  - name: mlflow
    tracking_uri: ${MLFLOW_TRACKING_URI}
```

### Template Substitution and Path Resolution

The framework supports several types of placeholders in configuration values:

**Path references** - Reference paths by their `name` from the `paths` section:

```yaml
datasets:
  - name: my_dataset
    formats:
      - type: parquet
        path: "{datasets_path}/subdir/data.parquet"
```

Any name defined in the `paths` section can be used as a placeholder. For example, if your `paths` section defines entries named `datasets_path` and `model_weights_path`, both `{datasets_path}` and `{model_weights_path}` are valid placeholders.

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

**ray**: Configures Ray resources. A global `ray` section can be defined at the top level of the configuration to set defaults inherited by all entries. Per-entry `ray` sections override individual keys from the global defaults.

Global defaults (applies to all entries unless overridden):
```yaml
ray:
  num_cpus: 64
  num_gpus: 8
  enable_object_spilling: false
```

Per-entry override (only the differing keys need to be specified):
```yaml
entries:
  - name: my_benchmark
    ray:
      num_gpus: 0  # overrides global num_gpus; num_cpus and enable_object_spilling inherit global values
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

## Audio Benchmark Data Setup

Audio benchmarks that depend on external corpora use the same two-layer setup:

1. Run a `benchmarking/data_prep/prepare_*_data.py` script once on the benchmark
   machine to populate persistent paths under `{datasets_path}` and, when
   needed, `{model_weights_path}`.
2. Run nightly entries from the staged data and local model paths so the
   benchmark itself never downloads inputs during the scheduled run. Entries
   that support a standalone download fallback also pass `--no-auto-download`.

Benchmarks that expose a standalone auto-download path keep it for ad hoc local
debugging only. That fallback stages into `{session_entry_dir}/scratch` or a
local scratch path and uses a stable Hugging Face cache to avoid re-fetching
blobs across reruns, but it is not the nightly path.

To run the checked-in audio setup before the benchmark session, pass
`--config benchmarking/nightly-data-setup.yaml` alongside the main benchmark
config to `benchmarking/tools/run.sh`. All supplied config files are merged
before the setup entries reuse an existing versioned manifest, or download and
stage it into the configured paths before the nightly benchmark entries start.

Current audio setup commands:

```bash
python benchmarking/data_prep/prepare_librispeech_data.py \
  --output-path {datasets_path}/librispeech_all_train_750h_71cacbfb \
  --cache-dir {datasets_path}/_hf_cache/librispeech \
  --hf-repo-id openslr/librispeech_asr \
  --hf-revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 \
  --hf-config all \
  --hf-split train.clean.100+train.clean.360+train.other.500 \
  --target-audio-hours 750.0

python benchmarking/data_prep/prepare_audio_tagging_data.py \
  --output-path {datasets_path}/audio_tagging_ami_sdm_8cdaae2_30h_max60m \
  --min-audio-hours 30 --max-meeting-duration-minutes 60 \
  --model-output-path {model_weights_path}/audio_tagging/pyannote-speaker-diarization-community-1_8a52737

python benchmarking/data_prep/prepare_alm_data.py \
  --output-path {datasets_path}/alm_ami_sdm_8cdaae2

python benchmarking/data_prep/prepare_audio_sortformer_data.py \
  --output-path {datasets_path}/audio_sortformer_librispeech_450h_1800x15m_71cacbfb \
  --model-output-path {model_weights_path}/audio_sortformer/diar_streaming_sortformer_4spk-v2.1.nemo
```

The setup pins each Hugging Face revision and selects the configured workload
scale in one pass. Timed entries consume these versioned paths and validate
pipeline outputs without rescanning or downloading the staged corpus.

| Workload | Before | Current result and target decision |
| --- | --- | --- |
| LibriSpeech ASR | Full English FLEURS, 7.4908h: Xenna 92.45s, Ray Data 143.92s | Shared 750h `openslr/librispeech_asr` manifest (CC BY 4.0), 217,974 unique clips with no repeated rows. |
| Audio tagging | Three AMI meetings: 100s; synthetic 8× repeat entry: 243s | 56 unique AMI SDM meetings / 30.2032h: 12m02s wall / 11m45s processing. Target achieved with real data; the repeat entry and repeat-factor support were removed |
| ALM | Ticket baselines: Ray Data 65s, Xenna 187s | Full AMI metadata (168 meetings / 82,063 segments / 96.41 timeline hours): Ray Data 32.37s, Xenna 38.72s. CPU-only, so the 8-GPU target does not apply |
| ReadSpeech | Ticket baselines: Xenna 315s; Ray Data did not finish when checked | Unchanged from `main`. The experimental HiFi-TTS calibration was discarded, so neither workload nor timeout is changed in this PR |

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
- `alm_pipeline_benchmark.py` - ALM audio pipeline benchmark
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
