# NeMo Curator Configuration Files

This directory contains examples for how to use a YAML file with Hydra for running common pipelines in NeMo Curator.

To run the Python script with a given YAML configuration file:

```bash
python run.py --config-path ./{target_dir} --config-name {target_file}.yaml param_1=... param_2=...
```

Where `{target_dir}` is the subdirectory containing the YAML file, `{target_file}` is the YAML file name, and `param_1=... param_2=...` are any parameters in the YAML file which are formatted with:

When the configuration file is named `pipeline.yaml`, `--config-name` may be omitted. Use `--config-name` for differently named configuration files.

```bash
param_1: ???
param_2: ???
```

The user may directly edit the YAML file to set these values, or include them in the command line without editing the YAML file as demonstrated above.

For example, to run the pipeline specified by `text/heuristic_filter_english_pipeline.yaml`:

```bash
python run.py --config-path ./text --config-name heuristic_filter_english_pipeline.yaml input_path=./input_dir output_path=./output_dir
```

## Executor and stage settings

Top-level keys choose the executor; omit all of them to use the pipeline's default executor:

```yaml
backend: xenna              # xenna (default) or ray_data
execution_mode: streaming   # Xenna only: streaming (default) or batch
executor_kwargs:            # passed to the executor constructor
  config: {logging_interval: 30}
  ignore_head_node: true
```

Each entry under `stages` is a Hydra `_target_` plus constructor arguments. `resources`, `num_workers`, and `num_workers_per_node` are applied with `stage.with_(...)` instead:

```yaml
stages:
  - _target_: nemo_curator.stages.audio.metrics.wer.GetPairwiseWerStage
    resources: {cpus: 2.0}
    num_workers: 4
```

Instead of `stages`, a config can define one `workflow` entry: a Hydra target whose `run(executor=...)` method is called, as in `tutorials/audio/audio_pretrain/pipeline.yaml`. After a pipeline run, `run.py` logs the number of output tasks and per-stage totals of processing time, items processed, and custom stage metrics.
