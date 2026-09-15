# Environment guide

One place to understand the environment the audio agent needs, check whether *your* machine is
healthy, and fix what isn't. The `doctor` verb is the runnable version of this document — it
probes the machine and prints per-issue fix steps.

## Check your machine

```bash
python -m nemo_curator.audio_agent doctor          # human-readable health report + fixes
python -m nemo_curator.audio_agent doctor --json   # same report as JSON (for tooling/agents)
```

Run this **first** whenever something env-related looks wrong, or **before a heavy GPU run**.
It is the single source of truth for environment health — capability cards and the failure
taxonomy point here instead of restating setup fixes.

`doctor` is machine-wide. Once a recipe exists, `validate`, `smoke`, and `run`
also return `environment_decision`, which filters those facts to the selected
flattened execution stages. A relevant blocker refuses agent-driven execution
before Ray/model setup and returns grounded choices; an irrelevant CUDA warning
does not block a CPU-only recipe. No choice is applied automatically.

To analyze a captured import/CUDA/Ray/model/runtime failure:

```bash
python -m nemo_curator.audio_agent diagnose --error '...' --recipe recipe.yaml
```

The result contains sanitized evidence, a failure classification, fresh
environment facts, applicable options, and a user-decision prompt. Unknown
failures stay unknown and receive diagnostic steps rather than a guessed fix.
Common disk-full, permission/mount, native-library/ABI, CUDA initialization, TLS,
dependency, and Ray-worker failures have distinct classifications; new signatures
remain additive in the versioned failure taxonomy.

Machine facts are scoped to where execution will happen. For an external Ray
cluster or caller-supplied executor, local driver GPU, ffmpeg, credential, Python,
disk, and worker-launch facts are not claimed as remote facts. The packet marks
the target environment unverified and asks for a bounded target-side smoke.
Unknown external GPU VRAM may be tested by that bounded smoke, but a full run
still refuses until target capacity (or equivalent target-bound evidence) is
available. A caller-supplied custom executor owns its scheduling; the agent
never substitutes the driver's CPUs/GPUs as that executor's capacity.

Overall status is the worst of the individual checks:

- **ok** — healthy.
- **warn** — a limitation or a missing optional (no GPU → CPU-only, unsupported interpreter,
  missing extra, low disk). Light pipelines still run.
- **fail** — broken/misconfigured; a common GPU workload *will* fail here (e.g. a GPU-driver vs
  CUDA-toolkit mismatch). Fix the FAILs before GPU model runs.

## What it checks (and how to fix)

### `python` — interpreter vs the project's `requires-python`
Confirms the interpreter is in the supported range. If not, recreate the venv with a supported
interpreter. Note: an *unsupported* interpreter is **not** the same as "CUDA is broken" — those
are separate checks, so a healthy Python here rules it out as the cause of GPU failures.

### `gpu` — GPU present + VRAM
`ok` when a CUDA GPU is visible. When it is not, the report distinguishes a
CPU-only torch build, `CUDA_VISIBLE_DEVICES` masking, visible NVIDIA devices with
failed torch initialization, driver/device-exposure failures, and genuinely
undetected hardware. These are not interchangeable: the grounded choices may be
to repair the host driver, expose/request a GPU allocation, restore the GPU
project environment, use another GPU host, or propose a CPU recipe. CPU is shown
as a conditional candidate only when every selected execution leaf explicitly
supports it; it is not called executable until the new recipe builds, plans,
validates, and passes a bounded smoke.

### `cuda_driver_toolkit` — GPU driver vs the CUDA toolkit torch was built with
The most important GPU check. `torch` bundles a CUDA **runtime** (`torch.version.cuda`, e.g.
`12.9`); the **driver** supports up to some max CUDA (`nvidia-smi`, e.g. `12.6`). Basic ops run
under minor-version compatibility, but anything that **JIT-compiles PTX at runtime** — NVRTC /
CUDA-graph decoders like NeMo's RNNT/TDT ASR — targets the toolkit's PTX ISA, which an **older
driver cannot load** → `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (error **222**).

Fix (any one):

- **Upgrade the NVIDIA driver** to one that supports the CUDA version torch was built for.
- **Install a torch built for the driver's CUDA** (a matching `+cuXXX` wheel, e.g. `+cu126`).
- **Sidestep the JIT path** for ASR *alignment*: set `decoder_type='ctc'` on
  `NeMoASRAlignerStage` / `SplitASRAlignJoinStage` (the hybrid `tdt_ctc` checkpoint has a CTC
  head, so no CUDA graphs and no model change). Note: plain `InferenceAsrNemoStage` with a
  pure-TDT checkpoint has no CTC head — there the env fix (or a CTC-capable model) is the only
  option.

The recipe-aware packet filters these options: CTC is never offered for an
unrelated metric such as UTMOS or for pure-TDT transcription, and a CPU variant
is unavailable when even one selected leaf is GPU-only.
The known mismatch is a hard preflight blocker only for a selected path known
to use runtime PTX/JIT/CUDA graphs (for example RNNT/TDT decode). Other GPU
stages receive a warning and bounded-smoke recommendation because precompiled
kernels may still work; the agent does not stop an existing working pipeline
without stage-specific evidence.

### `ffmpeg` — audio I/O
Needed for resample/convert and compressed formats (mp3/opus/…). Install with
`apt-get install ffmpeg`, `conda install -c conda-forge ffmpeg`, or `brew install ffmpeg`.

### `audio_extras` — importable audio dependencies
Install an audio dependency profile: `audio_cuda12` (GPU) or `audio_cpu` (CPU). In a
source checkout that is `uv sync --extra <profile>`; for an installed package the
command differs (and carries release-specific details), so take the current one from
the project's audio setup guide rather than from this file:
<https://docs.nvidia.com/nemo/curator/get-started/audio>.
`doctor` already picks the form that matches how this package is installed.
The fast probe checks package discoverability; a later native-library/ABI import
failure is classified from its actual error and analyzed with `diagnose`.

### `worker_env` — will a Ray **worker** import what the driver can?
Every other check probes *this* process, but pipelines execute in Ray **workers**. Ray's `uv`
integration notices the driver was started by `uv run` and rebuilds the worker environment by
re-running that command line — and `uv run` **without** an extra resolves only the *base*
dependency set. The result is a driver that imports `soundfile`/`nemo` happily next to workers
that die on `ModuleNotFoundError`, surfacing as:

```
Node setup failed for stage Stage 02 - GetAudioDurationStage on node ...
ModuleNotFoundError: No module named 'soundfile'
```

That reads like a broken install, but the install is fine — it is a **launch-flag** problem.
Fix (either one):

- Launch the interpreter directly: `.venv/bin/python -m nemo_curator.audio_agent …`
- Or carry the extra through: `uv run --extra audio_cuda12 python -m nemo_curator.audio_agent …`

### `disk` — free space
Model downloads (hundreds of MB to several GB) and intermediate WAVs need room; point caches at
a larger volume if low.

## Adding a new environment check

`env_health.py` is a small registry: write a function that reads the probed `EnvProfile` and
returns a `HealthCheck(id, status, finding, impact, fix=[...])`, decorate it with `@_check`, and
it appears in `doctor` automatically. Keep generic env concerns here — not in per-stage cards.
