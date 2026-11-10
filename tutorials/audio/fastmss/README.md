# FastMSS meeting simulation

From a Curator source checkout, install the optional toolkit with uv on Linux:

```bash
uv sync --extra audio_simulation
export FASTMSS_MANIFEST_DIR=/absolute/path/to/source/manifests
```

The extra pins FastMSS 0.1.0 to immutable Git revision
[packaging-only revision](https://github.com/timofeeva16tanya/FastMSS/commit/3f0a17393f0fc1642527aea1a057a8a5a1e9420d). Its code is installed as a separate
Apache-2.0 dependency; no FastMSS source is copied into Curator.
The pinned fork changes only package data to include YAML defaults; engine code
is unchanged from upstream 81fc2c3. FastMSS is not on PyPI, so use the source
checkout: uv resolves its Git source from tool.uv.sources. Published Curator
metadata contains only fastmss==0.1.0, without a direct URL.

Provide prepared, word-aligned single-speaker Lhotse cuts from at least two
speakers in `$FASTMSS_MANIFEST_DIR/source_train.jsonl.gz`. Source audio, configs
and output paths must be visible on every worker. `meetings.yaml` inherits the
defaults shipped in the installed FastMSS package.

```python
from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.simulation import FastMSS

pipeline = Pipeline(name="meeting_simulation", stages=[FastMSS(
    config_paths="tutorials/audio/fastmss/meetings.yaml",
    output_dir="/shared/generated-meetings",
)])
with RayClient():
    meetings = pipeline.run(RayActorPoolExecutor(show_progress=False))
```

Each YAML is an independent shard. The example generates two approximately
10-second meetings. FastMSS handles simulation; Curator reuses
`FilePartitioningStage` and `ManifestReaderStage` to emit one `AudioTask` per
meeting. This does not perform ASR or estimate diarization: its RTTM and text are
simulation ground truth, which can feed existing audio pipelines.

`audio_filepath`, `duration` and `rttm_filepath` describe the full mixture.
Curator joins supervision text by start time, breaking ties by supervision ID
and retaining overlaps. Missing transcripts fail the shard. Other fields pass
through unchanged. Failed runs retain logs; retries use fresh directories.

The pinned engine requires `min_max_spk` with `2 <= minimum <= maximum`.
`save_anechoic=true` requires `reverberate=true` and `reverb_prob=1`.
Curator checks resolved settings before preprocessing. Pass Hydra settings with
`overrides=["n_meetings=20"]`; paths, workers and manifest generation are managed
by the stage. An optional `fastmss_root` selects a local checkout for development;
by default the recipe is resolved inside the worker's installed distribution.
