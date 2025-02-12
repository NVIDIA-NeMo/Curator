## Text Classification

The Python scripts in this directory demonstrate how to run classification on your text data with each of these classifiers:

- Domain Classifier
- Multilingual Domain Classifier
- Quality Classifier
- AEGIS Safety Models
- Instruction Data Guard Model
- FineWeb Educational Content Classifier
- FineWeb Mixtral Educational Classifier
- FineWeb Nemotron-4 Educational Classifier
- Content Type Classifier
- Prompt Task and Complexity Classifier

For more information about these classifiers, please see NeMo Curator's [Distributed Data Classification documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/distributeddataclassification.html).

### Usage

#### Domain Classifier Inference

This classifier is recommended for English-only text data.

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

#### Multilingual Domain Classifier Inference

This classifier supports domain classification in 52 languages. Please see [nvidia/multilingual-domain-classifier on Hugging Face](https://huggingface.co/nvidia/multilingual-domain-classifier) for more information.

```bash
# same as `python multilingual_domain_classifier_inference.py`
multilingual_domain_classifier_inference \
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

Additional arguments may be added for customizing a Dask cluster and client. Run `multilingual_domain_classifier_inference --help` for more information.

#### Quality Classifier DeBERTa Inference

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

#### AEGIS Classifier Inference

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

#### Instruction Data Guard Classifier Inference

```bash
# same as `python instruction_data_guard_classifier_inference.py`
instruction_data_guard_classifier_inference \
    --input-data-dir /path/to/data/directory \
    --output-data-dir /path/to/output/directory \
    --input-file-type "jsonl" \
    --input-file-extension "jsonl" \
    --output-file-type "jsonl" \
    --input-text-field "text" \
    --batch-size 64 \
    --max-chars 6000 \
    --device "gpu" \
    --token "hf_1234"
```

In the above example, `--token` is your HuggingFace token, which is used when downloading the base Llama Guard model.

Additional arguments may be added for customizing a Dask cluster and client. Run `instruction_data_guard_classifier_inference --help` for more information.

#### FineWeb-Edu Classifier Inference

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

#### FineWeb Mixtral Edu Classifier Inference

```bash
# same as `python fineweb_mixtral_edu_classifier_inference.py`
fineweb_mixtral_edu_classifier_inference \
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

Additional arguments may be added for customizing a Dask cluster and client. Run `fineweb_mixtral_edu_classifier_inference --help` for more information.

#### FineWeb Nemotron-4 Edu Classifier Inference

```bash
# same as `python fineweb_nemotron_edu_classifier_inference.py`
fineweb_nemotron_edu_classifier_inference \
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

Additional arguments may be added for customizing a Dask cluster and client. Run `fineweb_nemotron_edu_classifier_inference --help` for more information.

#### Content Type Classifier DeBERTa Inference

```bash
# same as `python content_type_classifier_inference.py`
content_type_classifier_inference \
    --input-data-dir /path/to/data/directory \
    --output-data-dir /path/to/output/directory \
    --input-file-type "jsonl" \
    --input-file-extension "jsonl" \
    --output-file-type "jsonl" \
    --input-text-field "text" \
    --batch-size 64 \
    --autocast \
    --max-chars 5000 \
    --device "gpu"
```

Additional arguments may be added for customizing a Dask cluster and client. Run `content_type_classifier_inference --help` for more information.

#### Prompt Task and Complexity Classifier Inference

```bash
# same as `python prompt_task_complexity_classifier_inference.py`
prompt_task_complexity_classifier_inference \
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

Additional arguments may be added for customizing a Dask cluster and client. Run `prompt_task_complexity_classifier_inference --help` for more information.
