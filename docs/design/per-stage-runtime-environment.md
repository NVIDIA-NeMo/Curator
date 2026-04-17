# Per-Stage Runtime Environment Design Doc

**Author:** Ao Tang

## Overview

This document describes the design and implementation for enabling users to specify a distinct runtime environment for each stage in a data processing pipeline. This feature is essential for scenarios where different stages require incompatible library versions (e.g., `transformers==4.4` in one stage and `transformers==4.8` in another), or for services like vLLM that mandate specific dependency versions.

## Goals

| Goal | Description |
|------|-------------|
| Generic | Any pipeline stage can declare a specific runtime environment. |
| Backend agnostic | The solution supports Xenna, RayData, and RayActorPool execution backends. |

## Use Cases

| ID | Description |
|----|-------------|
| 1 | Execute a pipeline where the first stage requires `transformers==4.4` and the second requires `transformers==4.8`. |

## Implementation

### Stage Specification

A stage declares its required runtime environment via the `runtime_env` class variable on `ProcessingStage`. The value is a standard Ray [`runtime_env`](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html) dict:

```python
class VersionStage1(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "version_stage_1"
    resources = Resources(cpus=0.5)
    runtime_env: ClassVar[dict] = {"pip": ["transformers==4.4.0"]}  # only line needed

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import transformers  # will be 4.4.0
        ...
```

`runtime_env` is a `ClassVar` so it is part of the stage's class definition, not per-instance state. Stages without `runtime_env` (or with `runtime_env = None`) run in the default environment.

It can also be set per-instance via `with_()`:

```python
stage = MyStage().with_(runtime_env={"pip": ["transformers==4.4.0"]})
```

### Mechanism: Ray Native `runtime_env`

All backends pass `runtime_env` directly to Ray's actor/task options. Ray handles all virtualenv creation and caching — no driver-side venv creation, no `PYTHONPATH` manipulation.

Ray creates an isolated virtualenv per unique `runtime_env` under `/tmp/ray/session_latest/runtime_resources/pip/<hash>/virtualenv` and **caches it within the session**, so subsequent pipeline runs reuse the same env without reinstalling.

### Ray Data Backend

`RayDataStageAdapter.process_dataset()` forwards `runtime_env` into `ray_remote_args` before calling `map_batches`:

```python
# nemo_curator/backends/ray_data/adapter.py
if self.stage.runtime_env:
    ray_remote_args["runtime_env"] = self.stage.runtime_env
```

### Xenna Backend

Xenna's `Stage.env_info` property returns a `pipelines_v1.RuntimeEnv`, which Xenna converts to a Ray `runtime_env` via `to_ray_runtime_env()`. The built-in `pipelines_v1.RuntimeEnv` only supports `conda` and `env_vars`.

To bridge this, a `CuratorRuntimeEnv` class accepts the full Ray runtime_env dict and implements the two methods Xenna calls:

```python
# nemo_curator/backends/xenna/adapter.py
class CuratorRuntimeEnv:
    def __init__(self, runtime_env: dict[str, Any]) -> None:
        self._runtime_env = runtime_env

    def to_ray_runtime_env(self) -> ray.runtime_env.RuntimeEnv:
        return ray.runtime_env.RuntimeEnv(**self._runtime_env)

    def format(self) -> str:
        return f"runtime_env_keys: {', '.join(self._runtime_env.keys())}"
```

`XennaStageAdapter.env_info` returns a `CuratorRuntimeEnv` when the stage has `runtime_env`:

```python
@property
def env_info(self) -> pipelines_v1.RuntimeEnv | None:
    if not self.processing_stage.runtime_env:
        return None
    return CuratorRuntimeEnv(self.processing_stage.runtime_env)
```

### RayActorPool Backend

`RayActorPoolExecutor._create_actor_pool()` passes `runtime_env` to `.options()` when spawning actors:

```python
actor_options: dict = {"num_cpus": stage.resources.cpus, "num_gpus": stage.resources.gpus}
if stage.runtime_env:
    actor_options["runtime_env"] = stage.runtime_env
actor = AdapterClass.options(**actor_options, name=...).remote(stage)
```

### Call Flow

```
Stage.runtime_env = {"pip": ["pkg==1.0"]}
        │
        ▼
┌───────────────────────────────────────┐
│ Ray Data                              │
│  adapter → ray_remote_args            │
│            ["runtime_env"]            │
│            → map_batches(...)         │
└───────────────────────────────────────┘
        │
        ▼                    OR
┌───────────────────────────────────────┐
│ Xenna                                 │
│  adapter.env_info                     │
│    → CuratorRuntimeEnv(runtime_env)   │
│    → to_ray_runtime_env()             │
│    → actor.remote(runtime_env=...)    │
└───────────────────────────────────────┘
        │
        ▼                    OR
┌───────────────────────────────────────┐
│ RayActorPool                          │
│  _create_actor_pool                   │
│    → actor_options["runtime_env"]     │
│    → .options(**actor_options)        │
└───────────────────────────────────────┘
        │
        ▼
Ray creates & caches virtualenv
/tmp/ray/session_latest/runtime_resources/pip/<hash>/virtualenv
```

## Files Changed

| File | Change |
|------|--------|
| `nemo_curator/stages/base.py` | Added `runtime_env: ClassVar[dict \| None] = None` to `ProcessingStage`; added to `with_()` |
| `nemo_curator/backends/ray_data/adapter.py` | Passes `runtime_env` to `ray_remote_args` |
| `nemo_curator/backends/xenna/adapter.py` | Added `CuratorRuntimeEnv`; `env_info` returns it when `runtime_env` is set |
| `nemo_curator/backends/experimental/ray_actor_pool/executor.py` | Passes `runtime_env` to actor `.options()` |
| `tests/pipelines/test_per_stage_runtime_env.py` | Integration test for RayData and Xenna backends using `packaging==23.2` vs `packaging==24.0` |

## Alternatives Considered

### Driver-Side Venv Resolution (Original Design)

The original proposal created venvs on the driver process using the `uv` CLI before execution, then injected the resulting `site-packages` path via `PYTHONPATH` into worker environments.

**Why it was dropped:** Ray already provides exactly this mechanism natively — it creates virtualenvs per unique runtime_env, caches them, and handles injection automatically. The driver-side approach added ~140 lines of code (`stage_pip_env.py`), a `uv` CLI dependency, a `_resolved_site_packages_path` attribute on every stage, and PYTHONPATH string manipulation in both adapters, with no benefit over Ray's built-in handling.

### Ray's `uv` Plugin

Ray also supports a `uv` field in `runtime_env` (instead of `pip`). Under the hood, Ray's `uv` plugin still bootstraps by running `python -m pip install uv` inside the worker virtualenv. The `pip` field avoids this by using pip directly, which is available as a seed package in the container's `/opt/venv`.

## Notes

- The Ray `pip` plugin creates worker environments by cloning the current virtualenv. The NeMo Curator container's `/opt/venv` is created with `uv venv --seed`, which installs pip as a seed package. The clone therefore inherits pip and `python -m pip install` succeeds in worker virtualenvs.
- Virtualenvs are cached **per Ray session**. Restarting the Ray cluster clears the cache; the first run after a restart will reinstall packages.
- Stages with identical `runtime_env` share the same cached virtualenv.
- **`uv.lock` is excluded from the working_dir upload** (via `.rayignore`). When Ray installs `nemo-curator` from the uploaded working_dir into a per-stage pip virtualenv, uv reads `uv.lock` and enforces the locked transitive dependency versions (e.g. `transformers==4.55.2`), overwriting the runtime_env version the stage declared (e.g. `transformers==4.40.0`). Excluding the lockfile forces uv to resolve from `pyproject.toml` metadata only, leaving already-installed packages untouched.

## PR

https://github.com/NVIDIA-NeMo/Curator/pull/1623
