# Curating the FLEURS Dataset with NeMo Curator

The [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset is a speech dataset in 102 languages, built on top of the machine translation FLoRes benchmark. These scripts demonstrate how to download and read the audio data, run audio speech recognition (ASR) inference, calculate the word error rate (WER), calculate the duration, filter the audio data, and save the results.

It can be run in 2 ways:

```bash
python pipeline.py --raw_data_dir "/path/to/store/processed/data"
```

Alternatively, the user may configure the pipeline using a YAML file. First, the user should edit the `pipeline.yaml` file to specify the `raw_data_dir`. Then, the `run.py` script can be run with:

```
SCRIPT_DIR=/path/to/Curator/tutorials/audio/fleurs
python ${SCRIPT_DIR}/run.py --config-path ${SCRIPT_DIR} --config-name pipeline.yaml
```
