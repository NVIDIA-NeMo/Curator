# NeMo Curator Configuration Files

This directory contains examples for how to use a YAML file with Hydra for running common pipelines in NeMo Curator.

To run the Python script with a given YAML configuration file:

```bash
python run.py --config-path ./{target_dir} --config-name {target_file}.yaml param_1=... param_2=...
```

Where `{target_dir}` is the subdirectory containing the YAML file, `{target_file}` is the YAML file name, and `param_1=... param_2=...` are any parameters in the YAML file which are formatted with:

```bash
param_1: ???
param_2: ???
```

The user may directly edit the YAML file to set these values, or include them in the command line without editing the YAML file as demonstrated above.

For example, to run the pipeline specified by `text/heuristic_filter_english_pipeline.yaml`:

```bash
python run.py --config-path ./text --config-name heuristic_filter_english_pipeline.yaml input_path=./input_dir output_path=./output_dir
```
