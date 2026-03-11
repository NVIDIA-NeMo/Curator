# Megatron Tokenization Pipeline
This tutorial demonstrates how to tokenize the TinyStories dataset from Parquet files using `MegatronTokenizerWriter` for training with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## Usage
After installing the NeMo Curator package, you can simply run the following command:
```
LOGURU_LEVEL="ERROR" python tutorials/text/megatron-tokenizer/main.py
```

We use `LOGURU_LEVEL="ERROR"` to help minimize console output and produce cleaner logs for the user.

The script first checks whether the TinyStories dataset is already prepared; if not, it downloads it and saves it into 10 Parquet files. Using the `--input-path` and `--output-path` flags, you can configure where the tokenized files are read from and written to, while the `--tokenizer-model` flag specifies which tokenizer will be used to process the data. The `--append-eod` option allows you to add an end-of-document token to each processed document.

## MegatronTokenizerWriter

The `MegatronTokenizerWriter` is responsible for creating tokenized files compatible with Megatron-LM. This writer is intended to serve as a drop-in replacement for Megatron-LM’s original [`preprocess_data.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py) script, offering equivalent output while allowing integration into the NeMo Curator pipeline. It accepts `DocumentBatch` objects containing processed data, enabling seamless composition with other pipeline stages in NeMo Curator. The configuration options for this writer are:

- **model_identifier**: The tokenizer to load from the Hugging Face Hub using the `AutoTokenizer.from_pretrained` method
- **tokenization_batch_size**: The number of documents to feed into `tokenizer.batch_encode_plus` and write to disk at a time
- **append_eod**: Whether to append an end-of-document token to each sequence. Defaults to `False`
- **text_field**: The field in the `DocumentBatch` that contains the text to tokenize
- **cache_dir**: The directory where the cached tokenizer model will be stored. If not specified, it will use the [default transformers cache](https://huggingface.co/docs/transformers/en/installation#cache-directory) directory
- **hf_token**: The Hugging Face token for authentication (required if not logged in and attempting to access gated models)

## File Prefixes

The pipeline generates pairs of files with identical names—one with a `.bin` extension and another with a `.idx` extension. Megatron-LM refers to the filename without the file extension as the "file prefix."

### Bin File

The `.bin` files store the raw tokens of the processed documents. Token storage format depends on the tokenizer's vocabulary size: if the vocabulary size exceeds 2^16 (65,536), tokenized documents are stored as `numpy.int32` values, consuming 4 bytes per token; otherwise, they use `numpy.uint16`, consuming 2 bytes per token.

By default, following the approach in Megatron-LM's `preprocess_data.py` script, data is tokenized without special tokens (`add_special_tokens=False`). However, you can optionally append an end-of-document token to each sequence, which is retrieved from the `tokenizer.eos_token_id` attribute.

### Idx File

The `.idx` files store metadata about the tokenized documents in their corresponding `.bin` files. Specifically, the file structure contains:

- **9 bytes**: The `_INDEX_HEADER` constant. See the [Megatron-LM implementation](https://github.com/NVIDIA/Megatron-LM/blob/64cbae55ac85cd73fbadbc3c0d715c8123c5e13b/megatron/core/datasets/indexed_dataset.py#L38)
- **8 bytes**: The `.idx` file version (always set to `1`)
- **1 byte**: The `token_dtype_code`. See the [dtype code definitions](https://github.com/NVIDIA/Megatron-LM/blob/64cbae55ac85cd73fbadbc3c0d715c8123c5e13b/megatron/core/datasets/indexed_dataset.py#L41)
- **8 bytes**: The total number of sequences
- **8 bytes**: The total number of documents
- **8 bytes**: The initial document index

> [!NOTE]
> Megatron-LM initializes the document indices list with `[0]`, while the sequence list starts empty. This is why the document count is always one unit greater than the sequence count.

Following the header, the file contains **20 bytes per document** structured as follows:
  - **4 bytes**: The sequence length of the document
  - **8 bytes**: The sequence offset (byte position in the `.bin` file)
  - **8 bytes**: The document index

For more details about Megatron's DataLoading solution and tokenization pipeline refer to [`megatron.core.datasets`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/datasets) README.
