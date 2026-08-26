# NeMo Curator Benchmarks

NeMo Curator benchmarks are a separate benchmark suite for measuring Curator
runtime behavior, validating release candidates, comparing historical releases,
and tracking performance over time.

The benchmark suite is intentionally separate from the released Curator runtime.
Some benchmark dependencies are large, license-gated, or otherwise unsuitable
for redistribution in the standard Curator image. Benchmark users install those
dependencies explicitly by installing the separate, manually-installed
`nemo-curator-benchmarking` package from this repository.

## Table of Contents

- [Benchmark Package Model](#benchmark-package-model)
- [Quick Start](#quick-start)
- [Running Benchmarks in Containers](#running-benchmarks-in-containers)
- [Benchmarking Older Curator Images](#benchmarking-older-curator-images)
- [Prepared Images and Running Containers](#prepared-images-and-running-containers)
- [CI Orchestration](#ci-orchestration)
- [Configuration](#configuration)
- [Benchmark Data Setup](#benchmark-data-setup)
- [Authoring Benchmarks](#authoring-benchmarks)
- [Reporting Sinks](#reporting-sinks)
- [Environment Checks](#environment-checks)

---

## Benchmark Package Model

The benchmark package provides the benchmark runner, benchmark scripts, data
setup scripts, YAML configs, reporting sinks, and benchmark-only Python
dependencies. It is installed into an environment where Curator is already
available.

This separation keeps two versions explicit:

- **Curator under test**: the `nemo-curator` package installed in the active
  environment or prebuilt Curator image.
- **Benchmark suite**: the `nemo-curator-benchmarking` package installed from a
  selected Curator checkout.

The benchmark package must not replace the Curator package under test. This is
especially important when benchmarking a released Curator image or comparing an
older release against a newer benchmark suite.

The package installs a console command:

```bash
curator-benchmark run --config benchmarking/nightly-benchmark.yaml
```

For local source checkouts, install the benchmark package with the desired
extras:

```bash
uv pip install ./benchmarking[all]
```

Use narrower extras when possible:

```bash
uv pip install ./benchmarking[sinks]
```

Available extras include `sinks`, `audio`, and `nemotron_parse`. These extras
are limited to benchmark-package-owned dependencies; Curator feature stacks such
as audio, video, CUDA/vLLM, and interleaved data processing remain owned by the
main Curator package. The `all` extra installs every benchmark-package Python
dependency used by the standard benchmark suite, including dependencies that CI
previously installed directly at benchmark runtime.

If the environment is a released Curator image, use constraints or a setup mode
that prevents the benchmark install from upgrading or replacing the packages
that define the runtime being measured.

---

## Quick Start

Create a small path override file for machine-specific storage:

```yaml
paths:
  - name: results_path
    host_path: /path/to/results
  - name: datasets_path
    host_path: /path/to/datasets
    container_path: /datasets
  - name: model_weights_path
    host_path: /path/to/model_weights
    container_path: /model_weights
```

When the active environment already has Curator and benchmark dependencies
installed, run directly:

```bash
curator-benchmark run \
  --config ./benchmarking/nightly-benchmark.yaml \
  --config ./my-paths.yaml
```

To run in a standard Curator container from the host, add `--image`:

```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<tag> \
  --benchmark-setup auto \
  --config ./benchmarking/nightly-benchmark.yaml \
  --config ./my-paths.yaml
```

Add data setup when benchmarks need inputs that are prepared once and reused:

```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<tag> \
  --benchmark-setup auto \
  --config ./benchmarking/nightly-benchmark.yaml \
  --config ./benchmarking/nightly-data-setup.yaml \
  --config ./my-paths.yaml
```

Results are written under the configured `results_path`, grouped by session
name.

---

## Running Benchmarks in Containers

`curator-benchmark` is the primary entrypoint for bare-metal, image, and
running-container use cases. Without `--image` or `--container`, it runs in the
current environment. With `--image`, it starts a new Docker container. With
`--container`, it execs into an existing running container.

For image-based runs, the command reads the YAML configs on the host, creates
Docker volume mounts for configured paths, forwards GPUs and reporting
environment variables, mounts the selected benchmark-suite checkout, and starts
the selected Curator image.

For container-only use from a restricted host environment, the host Python
environment needs only the package code and the small set of dependencies used
before Docker starts. A minimal host install can be done with:

```bash
uv pip install pyyaml
uv pip install --no-deps ./benchmarking
```

Use this only for `--image` or `--container` workflows. Running benchmarks in
the current host environment requires the normal benchmark package dependencies
and the Curator feature dependencies required by the selected benchmarks.

The default flow is:

1. Start from the standard Curator image.
2. Mount the selected benchmark suite from the host.
3. Check whether the benchmark package and requested extras are already
   installed.
4. Install missing benchmark dependencies when setup mode allows it.
5. Re-run `curator-benchmark run` inside the container with the supplied config
   files and runner args.

Setup mode controls whether installation is attempted:

| Mode | Behavior |
| --- | --- |
| `auto` | Check for the benchmark package first. Install only when missing or incomplete. |
| `always` | Install or refresh the benchmark package from the mounted suite before running. |
| `never` | Do not install anything. Fail fast if the benchmark package or dependencies are missing. |

Run with an explicit image:

```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<tag> \
  --benchmark-setup auto \
  --config ./benchmarking/nightly-benchmark.yaml \
  --config ./my-paths.yaml
```

If `--image` is provided without a value, the default image comes from
`CURATOR_BENCHMARK_IMAGE`, then `CURATOR_BENCHMARKING_IMAGE`, then
`nemo_curator:latest`.

Use `--use-host-curator` only when the host Curator checkout itself is the
Curator version being tested. Do not use it when the Curator-under-test is the
package already installed in a release image.

`benchmarking/tools/run.sh` remains as a compatibility wrapper for older
workflows. It defaults to `curator-benchmark run --image <default-image>` and
passes all runner arguments through.

### Shell Access

Use shell mode to inspect the exact container environment:

```bash
curator-benchmark shell --image nvcr.io/nvidia/nemo-curator:<tag>
```

Run a single command in the container:

```bash
curator-benchmark shell \
  --image nvcr.io/nvidia/nemo-curator:<tag> \
  "curator-benchmark check"
```

### GPU Selection

Use `--gpus` to control Docker GPU visibility:

```bash
curator-benchmark run --image nvcr.io/nvidia/nemo-curator:<tag> --gpus all
curator-benchmark run --image nvcr.io/nvidia/nemo-curator:<tag> --gpus "device=0,1"
curator-benchmark run --image nvcr.io/nvidia/nemo-curator:<tag> --gpus none
```

---

## Benchmarking Older Curator Images

A common release-validation workflow is to benchmark an older Curator image with
the latest benchmark suite. This avoids using stale benchmark scripts and YAML
files that were baked into the old image.

In this mode:

- The old image provides the Curator package under test.
- The current checkout provides the benchmark package and configs.
- The benchmark package install must not replace the Curator package in the old
  image.

Example:

```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<old-release-tag> \
  --benchmark-suite-dir /path/to/latest/Curator \
  --benchmark-setup auto \
  --config /path/to/latest/Curator/benchmarking/nightly-benchmark.yaml \
  --config ./my-paths.yaml
```

Benchmark results should record both the Curator version under test and the
benchmark suite version or git SHA. This makes historical comparisons
interpretable when the benchmark suite evolves between releases.

---

## Prepared Images and Running Containers

Some users maintain images or running containers that already include benchmark
dependencies. Use setup mode `never` for those environments:

```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<tag-with-benchmark-deps> \
  --benchmark-setup never \
  --config ./benchmarking/nightly-benchmark.yaml \
  --config ./my-paths.yaml
```

Use `--container` to run inside an existing container:

```bash
curator-benchmark run \
  --container curator-benchmark-dev \
  --benchmark-setup never \
  --config /opt/curator-benchmark-suite/benchmarking/nightly-benchmark.yaml
```

When running against an existing container, the caller is responsible for
starting the container with appropriate benchmark-suite, dataset, results,
GPU, shared-memory, and network configuration. The benchmark runner should
validate the environment and fail clearly if required paths, tools, or Python
packages are missing.

---

## CI Orchestration

External CI and scheduler systems should treat Curator benchmarks as a selected
benchmark suite plus a selected Curator runtime. The Curator repository owns the
benchmark workload: YAML configs, runner code, benchmark scripts, data setup
scripts, package metadata, and local container launcher.

CI orchestration should remain responsible for:

- selecting the Curator image or environment under test
- selecting the benchmark-suite checkout
- installing `nemo-curator-benchmarking` when needed
- applying machine-specific path overrides
- launching one or more benchmark entries
- collecting results and logs

Keeping dependency metadata in the benchmark package lets benchmark authors
update scripts and dependencies in a single Curator PR. CI no longer needs
benchmark-specific dependency installation logic for each new benchmark.

## Configuration

### YAML Configuration Files

The benchmark runner uses one or more YAML files to configure benchmark
sessions. Multiple configuration files are merged, allowing separation of
concerns such as machine-specific paths, reporting settings, benchmark
definitions, and environment-specific overrides.

A useful pattern is to keep stable benchmark definitions in one config and layer
local or machine-specific settings on top. For example,
`my_paths_and_reports.yaml` can define results and dataset paths plus personal
sink settings, while `nightly-benchmark.yaml` defines the benchmark entries and
requirements.

This is especially useful during development. Use local path and report settings
while running the benchmark suite from the current checkout. Use
`--use-host-curator` only when the Curator source checkout on the host is the
Curator version being tested. When benchmarking a released Curator image, leave
the image's installed Curator package in place and install or reuse only the
separate benchmark package.

An example of a development scenario using this pattern looks like this:
```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<tag> \
  --use-host-curator \
  --config ~/curator_benchmarking/my_paths_and_reports.yaml \
  --config ./benchmarking/nightly-benchmark.yaml
```

### Configuration Structure

```yaml
# Required: Paths to files and directories used by the benchmarks.
# Each entry must have a "name" and a "host_path". The name can be referenced elsewhere
# in the config using {name} placeholders (e.g. {datasets_path}).
# When running with --image, each path is automatically mounted into the container.
# An optional "container_path" overrides the default mount point, which is the
# host_path prefixed with "/MOUNT".
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
# The path {session_entry_dir}/scratch is automatically created when an entry
# starts and can be used by benchmark scripts for writing temp files.
# This directory is automatically cleaned up on completion of the entry if
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
curator-benchmark run \
  --config config.yaml \
  --config paths.yaml \
  --config machine_specific.yaml
```

Files are merged in order using a deep recursive merge, so later files can override or extend specific nested values without replacing entire top-level keys.

**Merge behavior:**
- **Scalar values** (strings, numbers, booleans): later file wins.
- **Nested dicts**: merged recursively — only the keys present in the later file are updated.
- **Lists of dicts** (e.g. `entries`, `paths`, `requirements`, `sinks`): items are matched by their `name` key when present (the canonical identifier for most list items), falling back to the first key otherwise. If a matching item is found, it is merged recursively; if not, the item is appended. Use `name` in override files whenever possible to ensure reliable matching.

This makes it practical to write small override files that change only specific entries or requirements without duplicating the full configuration.

**Example — overriding a single entry's timeout and requirements:**

Base config (`nightly-benchmark.yaml`) defines many entries including:
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
curator-benchmark run \
  --config nightly-benchmark.yaml \
  --config my_overrides.yaml
```

Results in `domain_classification_xenna` using `timeout_s: 2000` and `min_value: 2000`, while all other entries remain unchanged.

**Session naming:**

```bash
curator-benchmark run \
  --config config.yaml \
  --session-name my-experiment-v2
```

**Benchmark viewer URL:**

To include a link to a benchmark run viewer in sinks such as Slack, pass a resolved URL with `--viewer-url`:

```bash
curator-benchmark run \
  --config config.yaml \
  --viewer-url "http://viewer.example.com/run-viewer?dir=/path/to/results/&run=my-session"
```

If part of the URL depends on the selected results path or session name, use `--viewer-url-template`. The template is rendered after the final session name and session path are known. When benchmarks run in a container with configured `host_path` / `container_path` mounts, path placeholders use the host-visible path so links work outside the container:

```bash
curator-benchmark run \
  --config config.yaml \
  --session-name my-session \
  --viewer-url-template "http://viewer.example.com/run-viewer?dir={results_path_url}&run={session_name_url}"
```

For a viewer that reads results from a remote host path, include the host in the template:

```bash
curator-benchmark run \
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

## Benchmark Data Setup

Data setup entries prepare reusable input data before benchmark entries run.
They are configured separately from benchmark entries so expensive downloads or
conversion steps can be run once and reused by later benchmark sessions.

Run the checked-in setup config with the benchmark config and your local path
overrides:

```bash
curator-benchmark run \
  --image nvcr.io/nvidia/nemo-curator:<tag> \
  --benchmark-setup auto \
  --config ./benchmarking/nightly-benchmark.yaml \
  --config ./benchmarking/nightly-data-setup.yaml \
  --config ./my-paths.yaml
```

All config files are merged before execution. Setup entries can use the same
path and dataset placeholders as benchmark entries. Logs are written under the
session directory at `data_setup/<setup-name>/logs/stdouterr.log`.

For setup-only workflows, use a standalone config that defines only paths,
sinks, and data setup entries. An empty benchmark entry list is written as:

```yaml
entries: []
sinks: []
```

If you layer a setup config on top of a full benchmark config, remember that
configuration files are merged. `entries: []` in a later file does not remove
entries already loaded from an earlier file. In that case, either use a
setup-only base config or pass an `--entries` selector that matches no benchmark
entries.

Data preparation code lives under `benchmarking/data_prep/`. Keep those scripts
idempotent: they should verify existing staged data, reuse it when valid, and
only download or transform data when required. Benchmark scripts should read
prepared data by path and should not modify persistent dataset directories
during scheduled runs.

## Authoring Benchmarks

Benchmark authors should be able to update benchmark code, config, and
benchmark-only dependencies in one Curator PR.

Place benchmark scripts in `benchmarking/scripts/` and reference them by
filename from YAML. Each benchmark script must accept
`--benchmark-results-path`; the runner passes this automatically and expects the
script to write its output files there.

Required output files:

| File | Purpose |
|---|---|
| `params.json` | Parameters and input paths used for the run. |
| `metrics.json` | Metrics used by requirement checks and reporting sinks. |
| `tasks.pkl` | Pickled Curator `Task` objects with detailed timing data. |

Benchmark-only dependency changes belong in the benchmark package metadata
under `benchmarking/pyproject.toml`. Keep the benchmark package separate from
the main Curator package: it should install benchmark tools and optional
benchmark-only dependencies without reinstalling or replacing the Curator
package being tested. Dependencies for Curator feature stacks belong in the
main Curator package extras instead of being duplicated here.

When a benchmark requires data, add or update a data setup script and the
corresponding data setup YAML in the same PR. The benchmark entry should then
refer to the staged dataset using placeholders rather than downloading data at
runtime.

Benchmarks may be run against older Curator release images using the latest
benchmark package and configs. If a script requires a newer Curator API, fail
with a clear error message or document the minimum supported Curator version for
that benchmark.

Reference implementations:

- `benchmarking/scripts/domain_classification_benchmark.py`
- `benchmarking/scripts/embedding_generation_benchmark.py`
- `benchmarking/scripts/removal_benchmark.py`
- `benchmarking/scripts/audio_tagging_benchmark.py`

## Reporting Sinks

Sinks handle reporting and side effects for benchmark lifecycle events. Built-in
sinks include Slack, MLflow, and Google Drive support. Sink failures are logged
by the runner and should not cause a benchmark entry to fail.

Example Slack sink:

```yaml
sinks:
  - name: slack
    channel_id: C1234567890
    enabled: true
```

Slack reporting requires `SLACK_BOT_TOKEN` in the environment. `channel_id` may
be provided in YAML or through the environment expected by the runner.

Example MLflow sink:

```yaml
sinks:
  - name: mlflow
    tracking_uri: http://mlflow-server:5000
    experiment: curator-benchmarks
    enabled: true
```

Entry-specific sink configuration belongs under `sink_data` on the entry. For
example, Slack can be asked to display additional metrics for a specific
benchmark:

```yaml
sink_data:
  - name: slack
    additional_metrics:
      - throughput_docs_per_sec
      - num_documents_processed
```

Custom sinks live under `benchmarking/runner/sinks/` and subclass
`runner.sinks.sink.Sink`. Register new sink names with the runner's sink loading
logic, then enable the sink from YAML.

## Environment Checks

Use `curator-benchmark check` to check whether the active environment is ready
to run benchmarks. This command is intended for both containers and bare-metal
environments.

Expected checks include:

- Curator import and version detection.
- Benchmark package version or git revision.
- Python executable and environment details.
- GPU visibility when GPU benchmarks are requested.
- Ray availability and object-store configuration.
- Required system tools such as `ffmpeg` and `ffprobe` when relevant.
- Required sink environment variables when sinks are enabled.
- Whether setup was skipped even though benchmark dependencies appear missing.

Examples:

```bash
curator-benchmark check --config ./benchmarking/nightly-benchmark.yaml

curator-benchmark check \
  --container curator-benchmark-dev \
  --benchmark-setup never \
  --config /opt/curator-benchmark-suite/benchmarking/nightly-benchmark.yaml
```

The check command should be advisory by default: it should explain missing or
risky environment pieces clearly and return a nonzero exit code only for checks
that make the requested benchmark run impossible.

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0. See the main repository LICENSE
file for details.
