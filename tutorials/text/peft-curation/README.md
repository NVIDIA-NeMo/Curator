# Curating Datasets for Parameter Efficient Fine-tuning

This tutorial demonstrates the usage of NeMo Curator's Python API to curate a dataset for
parameter-efficient fine-tuning (PEFT).

In this tutorial, we use the [Enron Emails dataset](https://huggingface.co/datasets/neelblabla/enron_labeled_emails_with_subjects-llama2-7b_finetuning),
which is a dataset of emails with corresponding classification labels for each email. Each email has
a subject, a body and a category (class label). We demonstrate various filtering and processing
operations that can be applied to each record.

We show how to format a record using Hugging Face's tokenizer's `apply_chat_template()` function and
also how to count the number of tokens for each record.

## Usage
After installing the NeMo Curator package, you can simply run the following command:
```
LOGURU_LEVEL="ERROR" python tutorials/text/peft-curation/main.py
```

We use `LOGURU_LEVEL="ERROR"` to help minimize console output and produce cleaner logs for the user.
