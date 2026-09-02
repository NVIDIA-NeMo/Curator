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
| `vllm.py` | Runtime-env construction (`dynamo_runtime_env`, `merge_model_runtime_envs`), actor-venv override file, worker subprocess env, engine kwargs |
| `config.py` | `DynamoVLLMModelConfig`, `DynamoServerConfig`, `DynamoRouterConfig` |
| `infra.py` | Actor naming, endpoint URLs, CLI-flag translation |
| `constants.py` | Default ports, namespace, event/request plane names |

## Base venv vs. actor venv

Curator's own `pyproject.toml` pins `vllm[flashinfer,runai,otel]==0.22.0+cu129`
directly (the `vllm` extra) and constrains `transformers>=4.56.0,<5.0`.
`ai-dynamo>=1.3.1` is *also* a direct optional dependency (via the
`inference_server` extra, pulled in by `sdg_cuda12`), so the driver/base venv
can already have Dynamo and vLLM installed. `text_cuda12` pulls in the plain
`vllm` extra but not `inference_server`, so it does not install `ai-dynamo`.

That base install doesn't automatically reach a model's actor, though:
**each Ray actor gets its own fresh, isolated venv that does not inherit the
driver's installed packages** ("Ray creates the actor venv outside the
project directory" — comment in `vllm.py`). `_dynamo_runtime_packages()`
reads whatever `ai-dynamo` version is installed on the *driver* and pins the
actor's `runtime_env` to install that exact `ai-dynamo[vllm]==<version>`
fresh, so the isolated actor venv ends up in lockstep with the driver
instead of silently resolving a different Dynamo/vLLM version. A model
needing something beyond what `ai-dynamo[vllm]`'s own resolution pulls in
(a newer `transformers`, an extra loader package) adds it through the same
`runtime_env`, merged on top of that base install via `merge_runtime_envs()`.

**This stack targets CUDA 12.x, not CUDA 13.x, everywhere.** `vllm` is
pinned to `==0.22.0+cu129`; `_ACTOR_VENV_CUDA_TAG` in `vllm.py` builds the
actor venv against that same `cu129` Torch/vLLM wheel index; `nixl-cu13` is
explicitly excluded from actor-venv resolution
(`_ACTOR_VENV_NIXL_CU13_EXCLUSION`) because it has previously been pulled in
as a transitive dependency. A `cu13`-tagged wheel or kernel landing anywhere
in this stack is one checkable cause of a startup or kernel-warmup failure —
confirm the CUDA tag on any newly-resolved wheel before assuming a model,
prompt, or Dynamo config issue. If the tags all check out, the installed
CUTLASS/QuACK build may simply be too old for the GPU architecture in use,
independent of CUDA tagging.

## Two separate environments, two separate mechanisms

Every Dynamo model runs as a Ray actor that (1) has its own **isolated
Python venv** and (2) launches a **worker subprocess**
(`python -m dynamo.vllm ...`) inside that actor. A dependency or
environment-variable problem belongs to exactly one of these, and the fix
mechanism differs:

| Need | Mechanism | Where it lands | Config surface |
|---|---|---|---|
| Install/override a Python package before the actor starts (a different `transformers`, an extra loader package, a version pin or exclusion) | Ray `runtime_env` (`uv`/`pip` packages) | Isolated actor venv, created fresh outside the project directory | `DynamoVLLMModelConfig.runtime_env`, merged via `dynamo_runtime_env()` / `merge_model_runtime_envs()` in `vllm.py` |
| Set an env var scoped to **one model's worker** (an engine feature flag, a per-model cache path) | `runtime_env["env_vars"]` on that model | That model's worker actor's `os.environ`, inherited by its worker subprocess | Same `runtime_env` field as above; `merge_runtime_envs()` unions `env_vars` too, not just packages |
| Set an env var that should reach **every model's** worker plus the frontend (a transport timeout, a compatibility shim path) | `subprocess_env` on the server | `base_env` folded into every worker/frontend subprocess's OS environment, not just one actor's | `DynamoServerConfig.subprocess_env`, applied in `backend.py` (`_deploy_and_healthcheck`) |

A package install always needs `runtime_env` — there's no `subprocess_env`
equivalent for that. For a plain env var, the choice is about **scope**, not
whether a package is involved: `runtime_env["env_vars"]` on one model does
not reach *other models'* worker actors, so it's the right choice for a
model-specific flag; `subprocess_env` is server-wide, so setting a
model-specific flag there would leak it onto every other configured model's
worker too. This isolation is not absolute, though — see the shared frontend
actor note below. Setting an installer- or import-relevant thing as a shell-level
`export` in the *driver* shell is almost always wrong regardless of scope:
the driver's shell environment does not propagate into the actor's isolated
venv, and if it reaches Ray itself (not just the worker subprocess), it can
make the Ray process import something it shouldn't and stall cluster/actor
startup — scope the variable to `runtime_env`/`subprocess_env` instead.

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
`ai-dynamo[vllm]` install *plus* the extra package — it does not need to
redeclare the base packages. Other models in the same server that don't set
`runtime_env` are unaffected; each model's actor gets its own merged env.
For the **shared frontend actor** (not a per-model worker), `vllm.py` unions
every model's `runtime_env` with `merge_model_runtime_envs()` so the
frontend venv is compatible with whatever any configured model needs. This
means a model's `runtime_env["env_vars"]` is not perfectly isolated to that
model's own worker — it also reaches the frontend actor. If two models set
the same key with different values, `merge_model_runtime_envs()` reduces
over the model list in order, so the *last* model in the list wins on the
frontend for that key.

### `subprocess_env` examples already in this codebase

A user-facing example is `tutorials/interleaved/nemotron_parse_pdf/README.md`,
which sets `DYN_TCP_REQUEST_TIMEOUT` — a Dynamo runtime setting the frontend
and worker subprocesses read at launch, not a package:

```python
DynamoServerConfig(
    request_plane="tcp",
    subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": "180"},
)
```

Note that `DynamoServerConfig.subprocess_env` is not a blank slate: Curator's
own resolved `ETCD_ENDPOINTS` and `NATS_SERVER` are added to `base_env`
*after* the user's `subprocess_env` in `backend.py` (`_deploy_and_healthcheck`),
so those two keys always win over anything supplied here — use
`etcd_endpoint`/`nats_url` on `DynamoServerConfig` to point workers at a
different etcd/NATS instance instead.

`_worker_subprocess_env()` in `vllm.py` anchors FlashInfer's cubin cache per
run so a worker doesn't reuse cubins from a since-replaced actor venv:

```python
def _worker_subprocess_env(base_env: dict[str, str], runtime_dir: str) -> dict[str, str]:
    return {**base_env, "FLASHINFER_WORKSPACE_BASE": f"{runtime_dir}/flashinfer"}
```

This applies to every worker regardless of model, so it belongs on
`DynamoServerConfig.subprocess_env` rather than a per-model `runtime_env`.

A compatibility shim that every worker needs importable before it imports
vLLM/QuACK/CUTLASS is the same case — server-wide, not model-specific — so
it also goes through `subprocess_env`, via `PYTHONPATH`:

```python
DynamoServerConfig(subprocess_env={"PYTHONPATH": "/abs/path/to/shim/dir"})
```

`PYTHONPATH` changes what's importable, but through `sys.path` at process
start rather than through package installation — no `uv`/`pip` resolution or
venv mutation is involved. If the same shim were only needed by one model,
`runtime_env["env_vars"]` on that model's `DynamoVLLMModelConfig` would be
the right scope instead. Only reach for `runtime_env`'s `uv`/`pip` keys when
the fix genuinely requires installing or pinning a package.

## Finding a working vLLM/QuACK/CUTLASS/CUDA combination

Work through this order rather than changing dependency versions by trial
and error:

1. **Confirm which environment is failing.** A traceback during Ray actor
   creation (before any `dynamo.vllm` subprocess log appears) is a
   `runtime_env` / actor-venv problem. A traceback inside worker subprocess
   logs (after the actor exists) is a `subprocess_env` / installed-package
   problem inside that already-created venv.
2. **Check the CUDA tag on every newly-resolved wheel** against the `cu129`
   baseline — see "Base venv vs. actor venv" above for what this class of
   failure looks like and its two causes.
3. **Do not assume a fresh Ray actor venv keeps driver-side pins.** Ray
   builds the actor venv outside the project directory, so anything the
   driver venv pins (Ray's own version, an excluded conflicting wheel) does
   not carry over automatically. `_ACTOR_VENV_OVERRIDES_PATH` is the
   existing pattern for this: `ensure_actor_overrides_on_all_nodes()` writes
   a `--override` file to a fixed node-local path before any actor using
   `DYNAMO_VLLM_RUNTIME_ENV` lands, pinning `ray==<driver version>` and
   excluding a conflicting wheel (`nixl-cu13`). Reuse this mechanism — via
   `_ACTOR_VENV_UV_OPTIONS` / the override file, or a per-model
   `runtime_env["uv"]["uv_pip_install_options"]` — rather than patching
   files inside an already-built venv.
4. **If free GPU memory looks sufficient but startup still fails on a
   memory check, check for a competing process first.**
   `gpu_memory_utilization` failures are a common false lead for what looks
   like a compatibility error; rule out contention with `nvidia-smi` before
   changing dependency versions.
5. **Re-run with the smallest reproducing case** (one model, one replica,
   `enforce_eager` if graph capture is a suspect) before assuming a
   multi-model or multi-replica interaction is the cause.
6. **After any `runtime_env`, `subprocess_env`, model, or engine-kwarg
   change, smoke-test with one replica and one request before trusting a
   full run.** A clean server-registration log is the serving checkpoint,
   not proof that generation will work end to end.
