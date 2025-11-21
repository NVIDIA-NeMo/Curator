# Megatron Tokenization Pipeline
This tutorial demonstrates how to tokenize the TinyStories dataset from Parquet files using `MegatronTokenizerWriter` for training with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## Usage
After installing the NeMo Curator package, you can simply run the following command:
```
LOGURU_LEVEL="ERROR" python tutorials/text/megatron-tokenizer/main.py
```

We use LOGURU_LEVEL="ERROR" to help minimize console output and produce cleaner logs for the user.

The script first checks whether the Tinystories dataset is already prepared; if not, it downloads it and saves it into ten parquet files. Using the `--input-path` and `--output-path` flags, you can configure where the tokenized files are read from and written to, while the `--tokenizer-model` flag specifies which tokenizer will be used to process the data. The `--append-eod` option allows you to add an end-of-document token to each processed document.

The pipeline generates pairs of files—one with the `.bin` extension and another with `.idx`. Megatron refers to these paired outputs as file prefixes: the `.bin` files contain the tokenized documents, and the `.idx` files store metadata about the corresponding `.bin` files.
