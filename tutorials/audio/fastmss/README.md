# FastMSS meeting simulation

Generate multi-speaker training mixtures with an external
[FastMSS checkout](https://github.com/popcornell/FastMSS) and ingest one Curator
`AudioTask` per meeting. Use the resulting audio, reference text and RTTM for
multi-speaker ASR and speaker diarization training workflows.

## Prerequisites

- An existing NeMo Curator audio environment on Linux.
- A separately installed FastMSS checkout and its dependencies on every worker.
- Aligned single-speaker source audio in a Lhotse CutSet. The example expects
  `$FASTMSS_MANIFEST_DIR/source_train.jsonl.gz`, with at least two speakers.
- Shared filesystem paths for configs, source audio, the checkout and outputs.

FastMSS is not vendored or installed by Curator. Use an upstream revision whose
license is approved for your intended distribution. The adapter targets the
Hydra CLI and recording-level manifest interface present in upstream commit
`e481ceac821ef0cb31033bf43b45922e377f528d`; the upstream license change must be
verified before this integration is released. This revision identifies the
interface baseline, not a permissive license approval.

## Run a small simulation

The example loads `recipes/default.yaml` from the external FastMSS checkout, so
the paths and defaults remain owned by FastMSS. Activate your existing Curator
environment, with FastMSS dependencies available in the worker Python, then run:

```bash
export FASTMSS_ROOT=/absolute/path/to/FastMSS
export FASTMSS_MANIFEST_DIR=/absolute/path/to/source/manifests

python tutorials/audio/fastmss/main.py \
  --fastmss-root "$FASTMSS_ROOT" \
  --config tutorials/audio/fastmss/meetings.yaml \
  --output-dir /shared/simulated-meetings/run-001 \
  --jobs-per-config 2
```

The output directory must be new. The example requests two approximately
10-second, two-speaker mixtures without noise or reverberation. FastMSS determines
actual recording lengths from the sampled utterances, so durations need not be
exactly 10 seconds. Source data must already be prepared; this example does not
download a speech corpus or run forced alignment.

Every invocation gets its own `fastmss-*` subdirectory containing generated
audio, `rttm/`, `nemo_manifest.json` and `fastmss.log`. After the pipeline succeeds,
the example writes `meetings.jsonl`, preserving FastMSS's manifest columns.
`audio_filepath`, `duration`, `text` and `rttm_filepath` describe the entire
meeting and its reference annotation. They do not describe isolated utterances:
overlapping turns remain in the mixture, and meeting text alone does not retain
speaker assignment. Use RTTM and aligned source annotations for that purpose.

## Configure and scale

Repeat `--config` to schedule independent simulation shards. Each config uses
`--jobs-per-config` Curator CPUs and the same number of FastMSS workers. Choose
separate source/preprocessing directories for shards when changing FastMSS's
`data_dir` or `rir_dir`; sharing writable caches between concurrent shards is
not supported by this adapter. Heavy preprocessing, RIR generation and audio
simulation remain inside FastMSS.

Additional Hydra overrides can enable FastMSS features:

```bash
python tutorials/audio/fastmss/main.py \
  --fastmss-root "$FASTMSS_ROOT" \
  --config tutorials/audio/fastmss/meetings.yaml \
  --output-dir /shared/simulated-meetings/run-002 \
  --override n_meetings=20 \
  --override reverberate=true \
  --override n_rirs=10 \
  --override save_spk=true \
  --override save_anechoic=true
```

The adapter owns `output_dir`, `n_jobs`, `stage`, `save_rttm`,
`save_nemo_manifest`, `hydra.run.dir` and `hydra.job.chdir`. It always starts
the recipe at stage 0 in a fresh output directory and requests recording-level
NeMo manifests plus RTTM. It never appends worker results to a shared manifest.
The example consolidates completed manifests on the driver.

For a customized checkout, use `--script-path sim_im_masr.py` and provide its
matching YAML configs. That script must honor the managed parameters, write
`<output_dir>/nemo_manifest.json` with one row per full recording, and return a
nonzero exit code on failure. Its dependencies must be installed separately;
use `--python-executable /absolute/path/to/python` if necessary. Customized
sampling, noise metadata and other extensions are not implemented by Curator.

The recipe and YAML configs are executable, trusted inputs. Run only code and
configs you control. No network downloads are added by the adapter; any external
downloads or FFmpeg requirements depend on the chosen FastMSS recipe and inputs.

Failures retain partial output and logs and fail the task. A timeout terminates
the child process group on Linux. Retrying starts a new directory, and Curator
checkpoint resumption is disabled for the simulation stage. There is no promise
of byte-identical regeneration or stable audio filenames across retries.

## Tests

```bash
python -m pytest tests/stages/audio/simulation/test_fastmss.py -m 'not gpu' -q
```

These tests exercise the real subprocess, filesystem, manifest reader and agent
contracts using a small protocol fixture. They do not test the FastMSS acoustic
simulation algorithm or claim quality measurements for generated training data.
