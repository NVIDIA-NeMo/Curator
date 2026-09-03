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

Curator's `pyproject.toml` pins `vllm[flashinfer,runai,otel]==0.22.0+cu129`
(the `vllm` extra) and constrains `transformers>=4.56.0,<5.0`. `ai-dynamo>=1.3.1`
is a separate optional dependency via `inference_server` (pulled in by
`sdg_cuda12`, not `text_cuda12`) — so the driver/base venv can already have
both Dynamo and vLLM installed, but never as `ai-dynamo[vllm]`:
`inference_server` lists plain `ai-dynamo` and Curator's own `vllm` extra
independently, and `ai-dynamo`'s own `vllm` requirement sits behind its
`[vllm]` extra marker, which the driver venv never requests.

That still isn't automatic per-model, though. Ray's `pip`/`uv` `runtime_env`
**clones the driver venv** for the actor, then installs requested packages
*additively* on top (general Ray/Curator behavior, not Dynamo-specific) —
already-cloned packages stay importable unless the additive install replaces
them. `_dynamo_runtime_packages()` reads the driver's installed `ai-dynamo`
version and has the actor install that exact `ai-dynamo[vllm]==<version>` on
top of the clone, keeping it pinned to the driver instead of drifting. A
model needing more (a newer `transformers`, an extra loader package) adds it
through the same `runtime_env`, merged via `merge_runtime_envs()`.

**This stack targets CUDA 12.x, not CUDA 13.x, everywhere.** `vllm` is
pinned to `==0.22.0+cu129`; `_ACTOR_VENV_CUDA_TAG` builds the actor venv
against that same `cu129` wheel index; `nixl-cu13` is explicitly excluded
(`_ACTOR_VENV_NIXL_CU13_EXCLUSION`) because it has previously been pulled in
transitively. A `cu13`-tagged wheel landing anywhere in this stack is one
checkable cause of a startup/kernel-warmup failure — check the CUDA tag on
any newly-resolved wheel before assuming a model/prompt/config issue. If
tags check out, the installed CUTLASS/QuACK build may just be too old for
the GPU architecture, independent of CUDA tagging.

## Two separate environments, two separate mechanisms

Every Dynamo model runs as a Ray actor with its own **isolated Python
venv**, which launches a **worker subprocess** (`python -m dynamo.vllm ...`).
A dependency or environment-variable problem belongs to exactly one of
these, and the fix mechanism differs:

| Need | Mechanism | Where it lands | Config surface |
|---|---|---|---|
| Install/override a Python package before the actor starts (a different `transformers`, an extra loader package, a version pin or exclusion) | Ray `runtime_env` (`uv`/`pip` packages) | Actor venv, cloned from the driver venv outside the project directory, then installed on top additively | `DynamoVLLMModelConfig.runtime_env`, merged via `dynamo_runtime_env()` in `vllm.py` |
| Set an env var scoped to **one model's worker** (an engine feature flag, a per-model cache path) | `runtime_env["env_vars"]` on that model | That model's worker actor's `os.environ`, inherited by its worker subprocess | Same `runtime_env` field as above; `merge_runtime_envs()` unions `env_vars` too, not just packages |
| Set an env var that should reach **every model's** worker plus the frontend (a transport timeout, a compatibility shim path) | `subprocess_env` on the server | `base_env` folded into every worker/frontend subprocess's OS environment, not just one actor's | `DynamoServerConfig.subprocess_env`, applied in `backend.py` (`_deploy_and_healthcheck`) |

A package install always needs `runtime_env` — no `subprocess_env`
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

This applies to every worker regardless of model, so it belongs on
`DynamoServerConfig.subprocess_env` rather than a per-model `runtime_env`.

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
2. **Check the CUDA tag on every newly-resolved wheel** against the `cu129`
   baseline — see "Base venv vs. actor venv" above for what this class of
   failure looks like and its two causes.
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
