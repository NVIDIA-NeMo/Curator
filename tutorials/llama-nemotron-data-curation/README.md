# Curating the Llama Nemotron Reasoning Dataset with NVIDIA NeMo Curator

The [Llama Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) is a curated collection of approximately 30 million high-quality synthetic samples, designed to enhance the reasoning capabilities of large language models.
Organized into distinct subsets for supervised fine-tuning (SFT) or reinforcement learning (RL), it encompasses samples from various problem domains.
All samples are in JSON lines (JSONL) format and contain metadata such as license type, source model, as well as the [Llama Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/llama-nemotron/) model(s) trained with that sample.

Each sample consists of a prompt, along with an expected reponse with detailed chain-of-thought (CoT) reasoning traces followed by responses (i.e., "reasoning on"), as well as samples with direct responses (i.e., "reasoning off").
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

This tutorial demonstrates how a user can process a subset the Llama Nemotron dataset using NeMo Curator. The output files are created in the `input/output` JSONL format, suitable for use with various training frameworks, including [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo). You can easily modify this pipeline as you see fit and adapt it to your domain- or business-specific needs, and the resulting dataset can be used to train a reasoning model with a modest computing budget.

## Environment Setup

Setup requirements:

- Hardware:
  - This tutorial can be run entire on a CPU with 4 workers 64GB RAM.
  - This tutorial can also be run on a single H100 GPU.
- Recommended environment: This tutorial was developed and tested with a Conda environment

Please refer to NeMo Curator's [README](https://github.com/NVIDIA/NeMo-Curator?tab=readme-ov-file#get-started) for instructions on how to download NeMo Curator via PyPI, source, or Docker.

## Prerequisites

### Download input dataset

The input dataset can be downloaded from Hugging Face here: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset

The following commands can be used to download the dataset:

```bash
git lfs install
git clone https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
```

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
python main.py \
    --input-dir "/path/to/Llama-Nemotron-Post-Training-Dataset/SFT" \
    --filename-filter "chat" "math_v1.1" \
    --remove-columns "category" "generator" "license" "reasoning" "system_prompt" "used_in_training" "version" \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --lang-id-model-path "/path/to/lid.176.ftz" \
    --max-token-count 16384 \
    --max-completion-token-count 8192 \
    --output-dir "/path/to/curated-data" \
    --device "gpu" \
    --n-workers 4
```

Since the entire input dataset is very large, we recommend curating a focused subset of the data that aligns closely with your domain-specific tasks. To help with this, we provide a way to filter files before reading. There are many ways to subset the Llama Nemotron dataset, but we recommend starting with the math and chat subsets because they contain strong examples of domain-agnostic reasoning. To filter files by name, the user may pass `--filename-filter` followed by any number of strings, such as "chat" and "math_v1.1". When reading the input data directory, the list of files will be filtered to only include files with names containing at least 1 of the strings provided by `--filename-filter`. If `--filename-filter` is not specified, then all files within the directory (over 30 million rows) will be used.

The above script applies basic filtering to the input dataset:

- Only take samples used for Nemotron Nano training
- Remove empty and malformed samples
- Remove non-English samples
- Remove samples with total length (system prompt, input, and output responses) longer than 16k tokens (with chat template applied via the tokenizer)
- Remove samples with output responses longer than 8k tokens (with chat template applied via the tokenizer)
- Remove any columns specified by the `--remove-columns` parameter. We recommend removing the columns specified above

After filtering, it sorts all samples by completion (output response) length, then "interleaves" thinking ON/thinking OFF for curriculum learning. The idea here is to sort the samples in increasing order of difficulty, using the completion token count as a measure of sample difficulty. We interleave samples from the "reasoning on" and "reasoning off" buckets to gradually introduce complexity. For large datasets, we recommend setting `--device "gpu"` and using an approximate interleaving approach which interleaves the data by Dask partition rather than row by row. If you prefer to interleave globally (per row), then you can specify the boolean flag `--global-interleave` from the script command.

If you are interested in counting and displaying the number of rows after each step in the pipeline, then you can specify the boolean flag `--generate-statistics` from the script command. Please note that enabling this is computationally expensive and will slow down the pipeline, so it is not recommended for large datasets. The default value is False.

## Debugging Out of Memory Errors

If you are running into out of memory (OOM) errors, there are a couple of approaches you can try. One is to avoid very large partitions of data. By default, the JSONL data is read with a blocksize of 100 MB per Dask partition. To customize the file reading logic, the user may specify `--json-blocksize "1gb"` with any string representation for the partition size (e.g., "1gb", "256mb"). Alternatively, the user may specify `--json-files-per-partition 2` with any integer to represent the number of JSONL files per Dask partition. Please note that either the blocksize or files per partition can be specified, but not both. For GPU workflows, a good general rule of thumb is to set the blocksize to 1/32 of the total GPU memory. In general, a blocksize between 100 MB and 1 GB is considered ideal.

You may also encounter errors about Dask workers unexpectedly shutting down. To help mitigate this, consider lowering the `--n-workers` parameter. By default, we set the number of Dask workers equal to the number of CPU cores. It may be helpful to set `--n-workers` to half or a fourth of the number of CPU cores and possibly reduce the number from there. For example, if `lscpu` shows `CPU(s): 96`, then setting `--n-workers 48` or `--n-workers 24` may help optimize performance while avoiding memory issues. In the example bash script, we set `--n-workers 4` as a safe option to help avoid errors.

## Next Steps

To see how to train a reasoning model with the resulting dataset, please refer to this NeMo tutorial: [Train Your Own Reasoning Model in 48 Hours on a Single GPU](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/reasoning).

Before running the NeMo tutorial, you should combine all of the resulting JSONL files from this tutorial into a single file called `training.jsonl`. You can use the following command to combine all of the JSONL files:

```bash
find /path/to/curated-data -type f -name "*.jsonl" -size +0c -print0 | xargs -0 cat | awk 'NF' > training.jsonl
```

Please note that the above command contains some additional logic to help ignore any empty JSONL files, which may have resulted from the filtering done by this tutorial.
