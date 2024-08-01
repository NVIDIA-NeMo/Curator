# Curating Datasets for Parameter Efficient Fine-tuning

This tutorial demonstrates the usage of NeMo Curator's Python API to curate a dataset for
parameter-efficient fine-tuning (PEFT).

In this tutorial, we use the [Enron Emails dataset](https://huggingface.co/datasets/neelblabla/enron_labeled_emails_with_subjects-llama2-7b_finetuning),
which is a dataset of emails with corresponding classification labels for each email. Each email has
a subject, a body and a category (class label). We demonstrate various filtering and processing
operations that can be applied to each record.

## Walkthrough
For a detailed walkthrough of this tutorial, please see the following blog post:
* [Curating Custom Datasets for LLM Parameter-Efficient Fine-Tuning with NVIDIA NeMo Curator](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-parameter-efficient-fine-tuning-with-nvidia-nemo-curator/).

## Usage
After installing the NeMo Curator package, you can simply run the following command:
```
python tutorials/peft-curation/main.py
```

By default, this tutorial will use at most 8 workers to run the curation pipeline. If you face any
out of memory issues, you can reduce the number of workers by supplying the `--n-workers=N` argument,
where `N` is the number of workers to spawn.
