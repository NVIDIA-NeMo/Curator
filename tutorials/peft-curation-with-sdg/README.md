# Curating Datasets for Parameter Efficient Fine-tuning with Synthetic Data Generation

This tutorial demonstrates the usage of NeMo Curator's Python API data curation as well as synthetic
data generation, and qualitative score assignment to prepare a dataset for parameter-efficient fine-tuning (PEFT) of LLMs.

We demonstrate the pipeline using the [Law StackExchange dataset](https://huggingface.co/datasets/ymoslem/Law-StackExchange),
which is a dataset of legal question/answers. Each record consists of a question, some context as
well as human-provided answers.

In this tutorial, we implement various filtering and processing operations on the records. We then
demonstrate the usage of external LLM services for synthetic data generation and reward models to
assign qualitative metrics to each synthetic record. We further NeMo Curator's facilities
to iteratively augment and refine the data until the dataset has reached the desired size.

> **Note:** The use of external LLM services for synthetic data generation is entirely optional.
> Similarly, this tutorial can be executed on a local machine without the need for a GPU. To fully
> experience all the capabilities of this code, see the "Optional Prerequisites" section below.

## Optional Prerequisites

The following is a list of optional dependencies to allow experimentation with all the features
showcased in this code:

* In order to run the data curation pipeline with semantic deduplication enabled, you would need an
NVIDIA GPU.
* To generate synthetic data, you would need a synthetic data generation model compatible with the OpenAI API. Out of the box, this tutorial supports the following model through the [build.nvidia.com](https://build.nvidia.com) API gateway:
  * [Nemotron-4 340B Instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct)
  * [LLaMa 3.1 405B Instruct](https://build.nvidia.com/meta/llama-3_1-405b-instruct)
* For assigning qualitative metrics to the generated records, you would need a reward model compatible with the OpenAI API (such as the [Nemotron-4 340B Reward](https://build.nvidia.com/nvidia/nemotron-4-340b-reward) model).

> **Note:** A valid [build.nvidia.com](https://build.nvidia.com) API key is required to use any of the above models.

## Usage
After installing the NeMo Curator package, you can simply run the following commands:
```bash
# Running the basic pipeline (no GPUs or external LLMs needed)
python tutorials/peft-curation-with-sdg/main.py

# Run with synthetic data generation and semantic dedeuplication
python tutorials/peft-curation-with-sdg/main.py \
    --api-key YOUR_BUILD.NVIDIA.COM_API_KEY \
    --device gpu

# To control the amount of synthetic data to generate using LLaMa 3.1 405B
python tutorials/peft-curation-with-sdg/main.py \
    --api-key YOUR_BUILD.NVIDIA.COM_API_KEY \
    --device gpu \  # Use the GPU and enable semantic deduplication
    --synth-gen-rounds 1 \ # Do 1 round of synthetic data generation
    --synth-gen-ratio 0.001 \  # Generate synthetic data using 0.1% of the real data
    --synth-gen-model "meta/llama-3.1-405b-instruct" # Use LLaMa 3.1 405B

# To control the amount of synthetic data to generate using Nemotron-4 340B
python tutorials/peft-curation-with-sdg/main.py \
    --api-key YOUR_BUILD.NVIDIA.COM_API_KEY \
    --device gpu \  # Use the GPU and enable semantic deduplication
    --synth-gen-rounds 1 \ # Do 1 round of synthetic data generation
    --synth-gen-ratio 0.001 \  # Generate synthetic data using 0.1% of the real data
    --synth-gen-model "nvidia/nemotron-4-340b-instruct" # Use Nemotron-4 340B
```

By default, this tutorial will use at most 8 workers to run the curation pipeline. If you face any
out of memory issues, you can reduce the number of workers by supplying the `--n-workers=N` argument,
where `N` is the number of workers to spawn.

Once the code finishes executing, the curated dataset will be available under `data/curated/final`.
By default, the script outputs splits for training (80%), validation (10%) and testing (10%).
