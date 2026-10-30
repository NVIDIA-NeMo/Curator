# FastMSS meeting simulation

Generate multi-speaker ASR and diarization training mixtures with the original
[FastMSS](https://github.com/popcornell/FastMSS), included inside Curator.
No separate FastMSS checkout or installation is required.

## Installation and attribution

Install the optional dependencies in an existing Linux Curator environment:

```bash
pip install lhotse pyroomacoustics numpy scipy soundfile torchaudio tqdm urllib3
```

The bundled implementation is an unmodified source snapshot of upstream commit
[`81fc2c3e4b938cf73343f58df9e013bc106cb77a`](https://github.com/popcornell/FastMSS/tree/81fc2c3e4b938cf73343f58df9e013bc106cb77a),
released under Apache-2.0. It includes the original simulator, turn-taking HMM,
RIR generation, utilities, recipe, defaults, paper configurations and preprocessing
scripts. Its `LICENSE`, `NOTICE`, `CITATION.cff` and original README are retained in
`nemo_curator/_vendor/fastmss/`. `UPSTREAM.json` records the
revision, omitted files and SHA-256 of every copied upstream file.

Source code is copied without local FastMSS modifications. Demo media, the
upstream Git ignore file and the two corpus annotation archives are omitted;
provide any data needed by a selected recipe separately. Maintainer scripts and
paper recipes are retained as upstream references, with their original paths.
The runnable Curator example below uses prepared source speech and upstream defaults.

## Run a small simulation

Provide a Lhotse CutSet containing word-aligned single-speaker speech from at least
two speakers at `$FASTMSS_MANIFEST_DIR/source_train.jsonl.gz`. Source audio,
configs and output paths must be visible to all workers.

```bash
export FASTMSS_MANIFEST_DIR=/absolute/path/to/source/manifests
python tutorials/audio/fastmss/main.py \
  --config tutorials/audio/fastmss/meetings.yaml \
  --output-dir /shared/simulated-meetings/run-001 \
  --jobs-per-config 2
```

Use a new output directory. The example requests two approximately 10-second,
two-speaker mixtures without noise or reverberation. Actual lengths depend on
sampled utterances. This example does not download speech or run alignment.
The stage sets `FASTMSS_ROOT` for Hydra to find the bundled original defaults.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.simulation import FastMSS

pipeline = Pipeline(name="multi_speaker_training_data", stages=[
    FastMSS(
        config_paths=["/shared/configs/two-speaker.yaml"],
        output_dir="/shared/simulated-meetings",
        jobs_per_config=2,
    )
])
```

Repeat `--config` to distribute independent shards. Additional `--override`
arguments configure original FastMSS options, such as `n_meetings=20`,
`reverberate=true` and `save_spk=true`. Keep writable preprocessing/RIR caches
separate between concurrent configs. An optional `--fastmss-root` can override
the bundled source for advanced use; the default always uses this original snapshot.

## Outputs and failure handling

Each invocation owns a fresh `fastmss-*` directory with audio, RTTM,
`nemo_manifest.json` and `fastmss.log`. The pipeline emits one `AudioTask` per
complete meeting. The example consolidates these rows into `meetings.jsonl`.
`audio_filepath`, `duration`, `text` and `rttm_filepath` refer to the whole mixture.
Overlapping speech stays in the waveform; meeting text alone does not encode
speaker assignment. Additional upstream manifest columns are preserved.

Curator builds `text` from generated supervision transcripts in ascending start
time, using supervision ID to break ties. It includes overlapping utterances;
this ordering does not imply that the speakers take turns without overlap.
Missing supervision transcripts fail the shard rather than emit unlabelled rows.

The original engine cannot reliably generate single-speaker meetings or save
anechoic references for dry meetings. Curator validates the resolved YAML and
CLI overrides before preprocessing: `min_max_spk` must have a minimum of at least
2 and a maximum no smaller than that minimum. `save_anechoic=true` requires
`reverberate=true` and `reverb_prob=1`. Retained upstream one-speaker paper recipes
are reference files and are unsupported by this adapter.

The adapter manages `output_dir`, `n_jobs`, `stage`, `save_rttm`,
`save_nemo_manifest`, `hydra.run.dir` and `hydra.job.chdir`. It runs the original
recipe in a child process with a matching CPU reservation. Failures retain logs
and partial outputs; Linux timeouts terminate the child process group. A retry
starts a new directory. Generation does not support Curator checkpoint resumption.

## Verification

```bash
python -m pytest tests/stages/audio/simulation/test_fastmss.py -m 'not gpu' -q
```

Tests verify copied upstream file checksums and the adapter's process/manifest
contract. With the simulation extra installed, a real bundled-engine test generates
aligned tone sources and checks nonzero finite audio, sample rate, duration and
both RTTM speakers. Tones test the integration; they are not training data or an
acoustic-quality benchmark.
