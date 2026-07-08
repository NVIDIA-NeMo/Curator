# Miniconda install hangs on draco (Lustre unpack) — problem & fix

## Symptom

Installing Miniconda into a Lustre path on the draco login node hangs forever at:

```
PREFIX=/lustre/fsw/portfolios/nemotron/users/ttimofeeva/miniconda3
Unpacking bootstrapper...
Unpacking payload...
```

- Directory size stalls (e.g. stuck at ~37–50 MB) and never grows.
- The installer subprocess sits in uninterruptible I/O (process state `I`/`D`):

```
_conda constructor --extract-tarball --prefix .../miniconda3
```

- `kill -9 <pid>` does **not** kill it (wedged in a Lustre syscall). The stuck
  PID only clears when the filesystem call returns or the session/node recycles.
- The same hang happens with `micromamba create` and with `rsync` when writing a
  conda tree onto Lustre — anything that writes tens of thousands of tiny files.

## Root cause

Miniconda/conda envs are **tens of thousands of small files**. The draco login
node's Lustre mounts (`/lustre/fsw`, `/lustre/fs12`) are very slow / stall on
that metadata-heavy small-file workload, so the unpack/extract step never
finishes.

## Fix — install to local disk (`/tmp`), not Lustre

Local disk on the login node (`/dev/sda1` mounted at `/`) handles the small-file
unpack fine. A full install finishes in ~25s.

```bash
MINICONDA_DIR=/tmp/miniconda3_tt
INSTALLER=/tmp/Miniconda3-latest-Linux-x86_64.sh

rm -rf "$MINICONDA_DIR"
curl -fsSL -o "$INSTALLER" \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash "$INSTALLER" -b -p "$MINICONDA_DIR"
"$MINICONDA_DIR/bin/conda" --version   # -> conda 26.3.2
```

### Accept conda Terms of Service (new conda requires this)

`conda create` fails with `CondaToSNonInteractiveError` until you accept the ToS:

```bash
source /tmp/miniconda3_tt/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Create the env (keep caches/temp on local disk too)

```bash
export PIP_CACHE_DIR=/tmp/pip_cache_tt
export TMPDIR=/tmp/tmp_tt
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

conda create -y -n curator_pain_1 python=3.12
conda activate curator_pain_1
conda install -y -c conda-forge montreal-forced-aligner=3.3.9 ffmpeg

python --version    # 3.12.13
mfa version         # 3.3.9
ffmpeg -version | head -1
```

## Curator install caveat: use PYTHONPATH, not `pip install -e`

`pip install -e /lustre/.../Curator` fails during metadata generation:

```
ModuleNotFoundError: No module named 'nemo_curator'
... invalid metadata entry `version`
```

(The dynamic version in `pyproject.toml` imports `nemo_curator`, which isn't
importable during the isolated build.) `--no-build-isolation` did not help
either. Workaround: install only the runtime deps and put the repo on
`PYTHONPATH`.

```bash
pip install lhotse textgrid num2words hydra-core omegaconf \
    soundfile tqdm pyloudnorm praatio

export PYTHONPATH=/lustre/fsw/portfolios/nemotron/users/ttimofeeva/Curator
python -c "import nemo_curator, lhotse, textgrid, num2words; print('deps OK')"
```

## Env file for the pipeline

Written to (on draco):

```
/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/david_ai_mfa_env.sh
```

Contents:

```bash
export MINICONDA_DIR=/tmp/miniconda3_tt
export MFA_ENV=/tmp/miniconda3_tt/envs/curator_pain_1
export MFA_ROOT_DIR=/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/MFA_models
export PIP_CACHE_DIR=/tmp/pip_cache_tt
export TMPDIR=/tmp/tmp_tt
export PYTHONPATH=/lustre/fsw/portfolios/nemotron/users/ttimofeeva/Curator
export CURATOR_ROOT=/lustre/fsw/portfolios/nemotron/users/ttimofeeva/Curator
export MFA_TUTORIAL=$CURATOR_ROOT/tutorials/audio/david_ai_redelivered_mfa
export WORK_DIR=/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/david_ai_mfa_workdir
export DATA_ROOT=$WORK_DIR/data_links
export SRC_DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_2026-05-29_redeliver
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate curator_pain_1
export PATH="$MFA_ENV/bin:$PATH"
```

Use it:

```bash
source /lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/david_ai_mfa_env.sh
cd "$MFA_TUTORIAL"
LOCAL_RUN=1 bash run_david_ai_mfa_cluster.sh
```

## Important caveats / gotchas

- **`/tmp` is local + ephemeral.** It is per-node and typically cleared on
  reboot. Fine for login-node testing. For batch/compute jobs, install conda on
  the compute node at job start, or package the env once and unpack per node.
- **Do not reuse the stuck Lustre paths.** These may still have wedged processes
  attached and should be treated as abandoned:
  - `/lustre/fsw/portfolios/nemotron/users/ttimofeeva/miniconda3`
  - `/lustre/fsw/portfolios/nemotron/users/ttimofeeva/miniconda3_copy`
  - `/lustre/fsw/portfolios/nemotron/users/ttimofeeva/micromamba`
- **Copying an env between machines needs identical absolute paths** (conda
  hardcodes them in shebangs/activation). Copying is not what fixed this — a
  clean local install did.

## If you must keep conda on Lustre (persistent)

Install locally first, then move as **one tarball** (one big file copies fine;
many small files do not):

```bash
# on the node with the working /tmp install
cd /tmp
tar --exclude='miniconda3_tt/pkgs/cache' -czf /lustre/.../miniconda3_tt.tar.gz miniconda3_tt

# later / elsewhere
cd /some/lustre/dir
tar -xzf /lustre/.../miniconda3_tt.tar.gz
```

Note the extracted path differs from `/tmp/miniconda3_tt`, so conda activation
may need `conda init` fix-ups or `--prefix` use because of hardcoded paths.
