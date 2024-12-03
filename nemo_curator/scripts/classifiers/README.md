## Text Classification

The Python scripts in this directory demonstrate how to run classification on your text data with each of these 4 classifiers:

- Domain Classifier
- Quality Classifier
- AEGIS Safety Models
- FineWeb Educational Content Classifier

For more information about these classifiers, please see NeMo Curator's [Distributed Data Classification documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/distributeddataclassification.html).

### Usage

#### Domain classifier inference

```bash
# same as `python domain_classifier_inference.py`
domain_classifier_inference \
    --input-data-dir /path/to/data/directory \
    --output-data-dir /path/to/output/directory \
    --input-file-type "jsonl" \
    --input-file-extension "jsonl" \
    --output-file-type "jsonl" \
    --input-text-field "text" \
    --batch-size 64 \
    --autocast \
    --max-chars 2000 \
    --device "gpu"
```

Additional arguments may be added for customizing a Dask cluster and client. Run `domain_classifier_inference --help` for more information.

#### Quality classifier inference

```bash
# same as `python quality_classifier_inference.py`
quality_classifier_inference \
    --input-data-dir /path/to/data/directory \
    --output-data-dir /path/to/output/directory \
    --input-file-type "jsonl" \
    --input-file-extension "jsonl" \
    --output-file-type "jsonl" \
    --input-text-field "text" \
    --batch-size 64 \
    --autocast \
    --max-chars 2000 \
    --device "gpu"
```

Additional arguments may be added for customizing a Dask cluster and client. Run `quality_classifier_inference --help` for more information.

#### AEGIS classifier inference

```bash
# same as `python aegis_classifier_inference.py`
aegis_classifier_inference \
    --input-data-dir /path/to/data/directory \
    --output-data-dir /path/to/output/directory \
    --input-file-type "jsonl" \
    --input-file-extension "jsonl" \
    --output-file-type "jsonl" \
    --input-text-field "text" \
    --batch-size 64 \
    --max-chars 6000 \
    --device "gpu" \
    --aegis-variant "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0" \
    --token "hf_1234"
```

- `--aegis-variant` can be `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0`, `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0`, or a path to your own PEFT of LlamaGuard 2.
- `--token` is your HuggingFace token, which is used when downloading the base Llama Guard model.

Additional arguments may be added for customizing a Dask cluster and client. Run `aegis_classifier_inference --help` for more information.

#### FineWeb-Edu classifier inference

```bash
# same as `python fineweb_edu_classifier_inference.py`
fineweb_edu_classifier_inference \
    --input-data-dir /path/to/data/directory \
    --output-data-dir /path/to/output/directory \
    --input-file-type "jsonl" \
    --input-file-extension "jsonl" \
    --output-file-type "jsonl" \
    --input-text-field "text" \
    --batch-size 64 \
    --autocast \
    --max-chars 2000 \
    --device "gpu"
```

Additional arguments may be added for customizing a Dask cluster and client. Run `fineweb_edu_classifier_inference --help` for more information.
