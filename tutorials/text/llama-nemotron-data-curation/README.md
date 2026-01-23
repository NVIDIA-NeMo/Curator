# Curating the Llama Nemotron Reasoning Dataset with NVIDIA NeMo Curator

The [Llama Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) is a curated collection of approximately 30 million high-quality synthetic samples, designed to enhance the reasoning capabilities of large language models.
Organized into distinct subsets for supervised fine-tuning (SFT) or reinforcement learning (RL), it encompasses samples from various problem domains.
All samples are in JSON lines (JSONL) format and contain metadata such as license type, source model, as well as the [Llama Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/llama-nemotron/) model(s) trained with that sample.

Each sample consists of a prompt, along with an expected response with detailed chain-of-thought (CoT) reasoning traces followed by responses (i.e., "reasoning on"), as well as samples with direct responses (i.e., "reasoning off").
Here is an example of what a sample from the dataset may look like:

```bash
{
  "input": [
    {"role": "user", "content": "Can you explain the Pythagorean theorem?"}
  ],
  "output": "<think>The user is asking for an explanation of the Pythagorean theorem. This is a fundamental principle in geometry related to right-angled triangles. I should mention the formula and what each variable represents.</think>The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c².",
  "reasoning": "on",
  "system_prompt": "detailed thinking on",
  "category": "math",
  "license": "apache_v2",
  "generator": "llama-3.3-70b",
  "used_in_training": ["Ultra"],
  "version": "v1"
}
```

The relevant attributes for this tutorial are as follows:

- `input`: the prompt(s) to the model in the multi-turn chat completions message format. It always contains a message with the role `user`, followed by zero or more turns
- `output`: the expected response from the model (ground truth)
- `reasoning`: whether the sample is for reasoning "on" mode or not
    - If the value is "on", then the output contains a detailed CoT trace encoded inside think HTML tags followed by the output
    - If the value is "off", then the output doesn't contain any reasoning traces and contains a direct response
- `system_prompt`: the (suggested) system prompt to control the reasoning mode of the system. For Llama Nemotron training, the system prompt is always either "detailed thinking on" or "detailed thinking off". Needless to say, this field is tied to the value in the field `reasoning` (and vice versa)
- `used_in_training`: the list of Llama Nemotron models that used this sample for training. For instance, a value of `["Ultra", "Nano"]` indicates that this sample was used for training Llama Nemotron and Ultra, but not Super

This tutorial demonstrates how a user can process a subset of the Llama Nemotron dataset using NeMo Curator. The output files are created in the `input/output` JSONL format, suitable for use with various training frameworks, including [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo). You can easily modify this pipeline as you see fit and adapt it to your domain- or business-specific needs, and the resulting dataset can be used to train a reasoning model with a modest computing budget.

## Environment Setup

Setup requirements:

- Hardware: This tutorial can be run entirely on CPU workers
- Recommended environment: This tutorial was developed and tested with a Conda environment

Please refer to NeMo Curator's [documentation](https://docs.nvidia.com/nemo/curator/latest/) for instructions on how to download NeMo Curator via PyPI, source, or Docker.

## Prerequisites

### Download input dataset

The input dataset can be downloaded from Hugging Face here: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset

The following commands can be used to download the dataset:

```bash
# If needed: apt-get update && apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
```

Alternatively, the dataset can be downloaded via Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/Llama-Nemotron-Post-Training-Dataset", 
    repo_type="dataset", 
    local_dir="/path/to/save/data",
    # allow_patterns=["SFT/chat/chat.jsonl", "SFT/math/math_v1.1.jsonl"],  # Select specific files or directories (if desired)
)
```

Please ensure that the above dataset was downloaded correctly. You can check:

```bash
$ ls /path/to/Llama-Nemotron-Post-Training-Dataset/SFT
chat  code  math  safety  science
$ du -sh /path/to/Llama-Nemotron-Post-Training-Dataset/SFT
122G    /path/to/Llama-Nemotron-Post-Training-Dataset/SFT
```

The above example ensures that the full SFT dataset was downloaded and is ready to use for the tutorial. If you only selected a subset of the data to download, then you should check that it matches the files on the [Hugging Face page](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset).

### Tokenizer access instructions

The tokenizer used by this tutorial is called [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Using it requires requesting access:

1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Access request"
3. Fill out the form and wait for approval
4. Once approved, log in to your Hugging Face account via the Hugging Face CLI. In the terminal, this can be done via `huggingface-cli login`

### Download FastText language identification model

The FastText language identification model is used to identify and filter out non-English text from the dataset. It can be downloaded from here: https://fasttext.cc/docs/en/language-identification.html

The following command can be used to download the FastText language identification model to your current working directory:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -P ./
```

## Usage

This tutorial can be run with:

```bash
LOGURU_LEVEL="ERROR" python main.py \
    --input-dir "/path/to/Llama-Nemotron-Post-Training-Dataset/SFT" \
    --filename-filter "chat" "math_v1.1" \
    --json-blocksize "100mb" \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --lang-id-model-path "/path/to/lid.176.ftz" \
    --max-token-count 16384 \
    --max-completion-token-count 8192 \
    --keep-columns "input" "output" \
    --output-dir "/path/to/curated-data" \
    --num-cpus 16
```

Setting `LOGURU_LEVEL="ERROR"` helps minimize long logs. Removing it can be helpful for debugging.

The user may set `--hf-token` as needed for the tokenizer.

Since the entire input dataset is very large, we recommend curating a focused subset of the data that aligns closely with your domain-specific tasks. To help with this, we provide a way to filter files before reading. There are many ways to subset the Llama Nemotron dataset, but we recommend starting with the math and chat subsets because they contain strong examples of domain-agnostic reasoning. To filter files by name, the user may pass `--filename-filter` followed by any number of strings, such as "chat" and "math_v1.1". When reading the input data directory, the list of files will be filtered to only include files with names containing at least 1 of the strings provided by `--filename-filter`. If `--filename-filter` is not specified, then all files within the directory (over 30 million rows) will be used.

The above script applies basic filtering to the input dataset:

- Only take samples used for Nemotron Nano training
- Remove empty and malformed samples
- Remove non-English samples
- Remove samples with total length (system prompt, input, and output responses) longer than 16k tokens (with chat template applied via the tokenizer)
- Remove samples with output responses longer than 8k tokens (with chat template applied via the tokenizer)
- Only keep columns specified by the `--keep-columns` parameter. We recommend keeping the "input", "output", and "completion_token_count" columns (the "completion_token_count" column always needs to be kept, so that we can sort the samples)

After filtering, it sorts all samples by completion (output response) length, then "interleaves" thinking ON/thinking OFF for curriculum learning. The idea here is to sort the samples in increasing order of difficulty, using the completion token count as a measure of sample difficulty. By default, we interleave one record at a time (i.e., 1 thinking ON sample, then 1 thinking OFF sample, then 1 thinking ON sample, etc.). The user may pass `--chunk-size` followed by an integer to interleave 10 records at a time (10 thinking ON samples, then 10 thinking OFF samples, then 10 thinking ON samples, and so on), 100 records at a time, etc. as desired. We interleave samples from the "reasoning on" and "reasoning off" buckets to gradually introduce complexity.

## Debugging Out of Memory Errors

If you are running into out of memory (OOM) errors, there are a couple of approaches you can try. One is to avoid very large partitions of data. By default, the JSONL data is read with a blocksize of 100 MB per partition. To customize the file reading logic, the user may specify `--json-blocksize "100mb"` with any string representation for the partition size in MB (e.g., "100mb", "256mb").

It can be useful to play around with the `--num-cpus` parameter as well. The goal is to maximize it for improved performance without oversubscribing your available hardware.

## Next Steps

To see how to train a reasoning model with the resulting dataset, please refer to this NeMo tutorial: [Train Your Own Reasoning Model in 48 Hours on a Single GPU](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/reasoning).

The NeMo tutorial expects the `/path/to/curated-data/training.jsonl` file generated by this tutorial as input.
