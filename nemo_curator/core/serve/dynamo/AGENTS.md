# Dynamo Backend — Agent Guide

Use this guide when a Dynamo/vLLM inference server fails to start or serve
correctly under Curator, or when a model needs dependencies beyond what
Dynamo's own install resolves. Diagnose from where the failure actually
occurs (driver, Ray actor venv, or worker subprocess) before changing
configuration.

## Files

| File | Role |
|---|---|
| `backend.py` | `DynamoBackend` lifecycle: infra placement group, etcd/nats, router, per-model launch, readiness |
| `vllm.py` | Runtime-env construction (`dynamo_runtime_env`), actor-venv override file, worker subprocess env, engine kwargs |
| `config.py` | `DynamoVLLMModelConfig`, `DynamoServerConfig`, `DynamoRouterConfig` |
| `infra.py` | Actor naming, endpoint URLs, CLI-flag translation |
| `constants.py` | Default ports, namespace, event/request plane names |

## Base venv vs. actor venv

By default, Ray's `uv` runtime environment clones the driver's virtualenv
and installs additional packages for the actor. `dynamo_runtime_env()` pins
`ai-dynamo[vllm]` to the driver's installed Dynamo version and merges model
extras with the required wheel indexes and overrides. The driver's own vLLM
version need not match the one resolved by Dynamo's extra. Inspect installed
package metadata and the current helpers in `vllm.py`; do not copy version
pins or CUDA wheel indexes from an older image.

### Preinstall a serving venv to avoid startup installs

Keep the pipeline's required dependencies available in the final image. When
extending a full Curator image, preserve its driver environment and add a
separate serving venv. A smaller backend-based image can also host a CPU pipeline
with Curator's base and client dependencies installed; validate the complete
read → inference → write flow, not just server startup. Use separate environments
when dependencies conflict, matching the serving actors' Python minor version
and Ray version to the driver even when their backend stacks differ.

First check the model's architecture, upstream serving recipe and Dynamo's
backend compatibility. A model can require an unreleased backend even when
its weights are cached. The helper below follows the driver's Dynamo release;
it does not discover the newest backend that supports a model. For a different
release, align the base image, Python, Torch/CUDA wheels and backend together.
Check `torch.version.cuda` and import the backend with a GPU visible: dependency
resolution and CPU-only imports can both pass with incompatible CUDA libraries.

Install **Curator without extras** together with `ai-dynamo[vllm]` and the
model's additional packages. Curator's base dependencies supply Ray, Xenna,
pandas and PyArrow; a bare Dynamo venv misses Curator's bootstrap imports.
Let Curator's package metadata maintain that dependency list. A normal
installation also makes its source importable without a custom `.pth` file.

Add this build step to a Curator Docker image with its driver environment
at `/opt/venv` and Curator checkout at `/opt/Curator` (adjust paths to match
the image). Add the model's extra packages to the config before resolving
its runtime environment; do not set `py_executable` until after the build.

```dockerfile
RUN /opt/venv/bin/python - <<'PY'
import subprocess
import sys
from importlib.metadata import version

from nemo_curator.core.serve import DynamoVLLMModelConfig
from nemo_curator.core.serve.dynamo import vllm

model = DynamoVLLMModelConfig(
    model_identifier="your-model",
    runtime_env={"uv": {"packages": []}},  # Add model-specific dependencies here.
)
uv = vllm.dynamo_runtime_env(model)["uv"]
python = "/opt/dynamo/bin/python"
subprocess.run(["uv", "venv", "--python", sys.executable, "/opt/dynamo"], check=True)
overrides = vllm._ACTOR_VENV_OVERRIDES_PATH
overrides.write_text(f"ray=={version('ray')}\n{vllm._ACTOR_VENV_NIXL_CU13_EXCLUSION}\n")
subprocess.run(
    ["uv", "pip", "install", "--python", python, "--no-sources",
     *uv["uv_pip_install_options"], "/opt/Curator", *uv["packages"]],
    cwd="/tmp", check=True,
)
overrides.unlink()
subprocess.run(
    [python, "-I", "-c", "import cosmos_xenna; import nemo_curator.core.serve.subprocess_mgr"],
    check=True,
)
PY
```

`/opt/Curator` installs the image's Curator revision with no extras. Resolve it
and the serving packages in one install so their shared dependencies agree;
the override keeps Ray matched to the driver. `--no-sources` prevents the
checkout's `tool.uv.sources` from silently selecting development wheel indexes
instead of the serving stack's indexes. Avoid Curator's `vllm` or
`inference_server` extras here: Dynamo selects its own vLLM dependencies.
Build the venv separately for each CPU architecture. The import check covers
actor bootstrap; validate model-specific dependencies with one replica and
a real request before scaling up.

Select the preinstalled venv with:

```python
DynamoVLLMModelConfig(
    model_identifier="your-model",
    runtime_env={"py_executable": "/opt/dynamo/bin/python"},
)
```

This skips automatic package additions for workers and the shared frontend.
Do not combine it with `uv`, `pip`, or `conda` installation settings. All models
sharing a frontend must select the same interpreter, available at the same
path on every participating node.

For multi-model serving, pass multiple model configs to one `InferenceServer`.
They share an HTTP endpoint; the request's `model` selects the corresponding
workers. GPU allocations remain separate. The frontend merges their runtime
environments, not their model engines: packages must resolve together, and
preinstalled configs must use one interpreter containing their combined
dependencies. Separate incompatible environments need separate server endpoints
or an explicit frontend-environment design; merging dicts cannot combine venvs.

### Persist compilation caches across server starts

Cache configuration is independent of whether the venv is managed or baked
into the image. Set these variables on all serving subprocesses:

```python
DynamoServerConfig(subprocess_env={
    "CUDA_CACHE_PATH": "/cache/cuda",
    "TRITON_CACHE_DIR": "/cache/triton",
    "VLLM_CACHE_ROOT": "/cache/vllm",
})
```

| Variable | Reusable artifacts |
|---|---|
| `CUDA_CACHE_PATH` | CUDA driver JIT compilation output |
| `TRITON_CACHE_DIR` | Compiled Triton kernels |
| `VLLM_CACHE_ROOT` | vLLM compilation artifacts and other cached data |

With Docker, mount writable host/shared storage **outside the container** at
`/cache` so the cache survives container deletion and subsequent runs can
reuse compiled artifacts:

```bash
mkdir -p /shared/curator-cache/stack-id/{cuda,triton,vllm}
docker run --mount type=bind,src=/shared/curator-cache/stack-id,dst=/cache IMAGE ...
```

Use a separate `stack-id` directory for each image/GPU/driver combination.
The first run populates the caches; measure warm startup separately. These
caches do not contain model weights: mount the existing model cache and set
`HF_HOME` to its container path too. They reduce repeated compilation, but
model loading and CUDA graph capture can still contribute to startup time.

## Two separate environments, two separate mechanisms

Every Dynamo model runs as a Ray actor in a managed or preinstalled Python
venv, which launches a **worker subprocess** (`python -m dynamo.vllm ...`).
A dependency or environment-variable problem belongs to exactly one of
these, and the fix mechanism differs:

| Need | Mechanism | Where it lands | Config surface |
|---|---|---|---|
| Install/override a Python package before the actor starts (a different `transformers`, an extra loader package, a version pin or exclusion) | Ray `runtime_env` (`uv`/`pip` packages) | Actor venv, cloned from the driver venv outside the project directory, then installed on top additively | `DynamoVLLMModelConfig.runtime_env`, merged via `dynamo_runtime_env()` in `vllm.py` |
| Set an env var scoped to **one model's worker** (an engine feature flag, a per-model cache path) | `runtime_env["env_vars"]` on that model | That model's worker actor's `os.environ`, inherited by its worker subprocess | Same `runtime_env` field as above; `merge_runtime_envs()` unions `env_vars` too, not just packages |
| Set an env var that should reach **every model's** worker plus the frontend (a transport timeout, a compatibility shim path) | `subprocess_env` on the server | `base_env` folded into every worker/frontend subprocess's OS environment, not just one actor's | `DynamoServerConfig.subprocess_env`, applied in `backend.py` (`_deploy_and_healthcheck`) |

A per-actor package install needs `runtime_env` — no `subprocess_env`
equivalent exists. For a plain env var, the choice is **scope**, not
whether a package is involved: `runtime_env["env_vars"]` on one model
doesn't reach other models' workers (right for a model-specific flag);
`subprocess_env` is server-wide, so a model-specific flag there leaks onto
every other model's worker. This isolation isn't absolute — see the
frontend note below. Never `export` an installer/import-relevant var in the
driver shell: it won't propagate into the actor's isolated venv, and if it
reaches Ray itself (not just the worker subprocess) it can make Ray import
something unexpected and stall startup — scope it to
`runtime_env`/`subprocess_env` instead.

### Minimal `runtime_env` example

A model that needs a newer `transformers` than the base install provides,
plus a vLLM feature flag, sets both on its own `DynamoVLLMModelConfig`:

```python
DynamoVLLMModelConfig(
    model_identifier="google/gemma-4-31B-it",
    runtime_env={
        "uv": {"packages": ["transformers>=5"]},
        "env_vars": {"VLLM_USE_DEEP_GEMM": "0"},
    },
)
```

`merge_runtime_envs()` unions `env_vars` and appends to the `uv`/`pip`
package list rather than replacing it, so this model gets the base
`ai-dynamo[vllm]` install *plus* the extra package. Other models without
`runtime_env` are unaffected — each actor gets its own merged env. The
**shared frontend actor** is the exception: `merge_model_runtime_envs()`
unions *every* model's `runtime_env` onto it. `env_vars` merge cleanly (last
model in the list wins on a conflicting key), but `_merge_package_runtime_env()`
concatenates `uv`/`pip` package lists (`[*base, *override]`) rather than
reconciling them — two models pinning incompatible versions of the same
package both land in the frontend's install list and can fail to resolve,
which blocks the frontend (and the whole server) from starting. Keep
model-specific pins mutually compatible, or split conflicting models across
separate `InferenceServer` instances.

### `subprocess_env` examples already in this codebase

A real example, from `tutorials/interleaved/nemotron_parse_pdf/README.md`,
sets `DYN_TCP_REQUEST_TIMEOUT` — a runtime value the frontend/workers read
at launch, not a package:

```python
DynamoServerConfig(
    request_plane="tcp",
    subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": "180"},
)
```

`subprocess_env` isn't a blank slate: Curator's own `ETCD_ENDPOINTS`/
`NATS_SERVER` are added to `base_env` *after* the user's `subprocess_env` in
`backend.py` (`_deploy_and_healthcheck`), so those two keys always win — use
`etcd_endpoint`/`nats_url` instead to redirect workers.

`_worker_subprocess_env()` anchors FlashInfer's cubin cache per run so a
worker doesn't reuse cubins from a since-replaced actor venv:

```python
def _worker_subprocess_env(base_env: dict[str, str], runtime_dir: str) -> dict[str, str]:
    return {**base_env, "FLASHINFER_WORKSPACE_BASE": f"{runtime_dir}/flashinfer"}
```

This internal per-run setting overrides `FLASHINFER_WORKSPACE_BASE` from
user configuration; the persistent cache paths above do not change it.

A compatibility shim every worker needs importable before it imports
vLLM/QuACK/CUTLASS is the same case — server-wide — so it also goes through
`subprocess_env`, via `PYTHONPATH`:

```python
DynamoServerConfig(subprocess_env={"PYTHONPATH": "/abs/path/to/shim/dir"})
```

`PYTHONPATH` changes what's importable via `sys.path`, not package
installation — no `uv`/`pip` resolution or venv mutation involved. If only
one model needed the shim, `runtime_env["env_vars"]` on that model would be
the right scope instead. Reach for `runtime_env`'s `uv`/`pip` keys only
when the fix genuinely requires installing or pinning a package.

## Finding a working vLLM/QuACK/CUTLASS/CUDA combination

Work through this order rather than changing dependency versions by trial
and error:

1. **Confirm which environment is failing.** A traceback during actor
   creation (before any `dynamo.vllm` subprocess log) is a `runtime_env`/
   actor-venv problem; one inside worker subprocess logs (actor already
   exists) is a `subprocess_env`/installed-package problem.
2. **Check the CUDA tag on every newly-resolved wheel** against the current actor wheel-index
   configuration. Also check whether kernel libraries support the target
   GPU architecture; a matching CUDA tag alone does not guarantee that.
3. **The additive `runtime_env` install can disturb a pin already cloned
   into the actor venv** (silently upgrade `ray`, or introduce `nixl-cu13`)
   unless something pins or excludes it. `_ACTOR_VENV_OVERRIDES_PATH` is
   the existing guard: `ensure_actor_overrides_on_all_nodes()` writes a
   `--override` file to a fixed node-local path before any actor using
   `DYNAMO_VLLM_RUNTIME_ENV` lands, pinning `ray==<driver version>` and
   excluding `nixl-cu13`. Reuse this — via `_ACTOR_VENV_UV_OPTIONS`, the
   override file, or a per-model `runtime_env["uv"]["uv_pip_install_options"]`
   — rather than patching an already-built venv.
4. **Rule out GPU memory contention before chasing a compatibility fix.** A
   `gpu_memory_utilization` failure with seemingly-sufficient free memory is
   a common false lead — check `nvidia-smi` for a competing process first.
5. **Re-run with the smallest reproducing case** (one model, one replica,
   `enforce_eager` if graph capture is a suspect) before assuming a
   multi-model or multi-replica interaction is the cause.
6. **Smoke-test with one replica and one request after any `runtime_env`,
   `subprocess_env`, model, or engine-kwarg change** before trusting a full
   run — a clean server-registration log proves registration, not that
   generation works.
