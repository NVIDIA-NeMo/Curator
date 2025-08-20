# Tutorials
The following is a set of tutorials that demonstrate various functionalities and features of NeMo Curator. These tutorials are meant to provide the coding foundation for building applications that consume the data that NeMo Curator curates.

## Get Started
To get started, we recommend starting with the following tutorials to become familiar with various functionalities of NeMo Curator and get an idea of what a data curation pipeline might look like:
1. **[tinystories](./tinystories)**, which overviews core functionalities such as downloading, filtering, PII removal and exact deduplication.
2. **[peft-curation](./peft-curation)**, which overviews operations suitable for curating small-scale datasets which are used for task-specific fine-tuning.
3. **[synthetic-data-hello-world](./synthetic-data-hello-world)**, which overviews basic synthetic data generation facilities for interfacing with external models such as [Nemotron-4 340B Instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct).
4. **[peft-curation-with-sdg](./peft-curation-with-sdg)**, which combines data processing opeartions and synthetic data generation using [Nemotron-4 340B Instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct) or [LLaMa 3.1 405B Instruct](https://build.nvidia.com/meta/llama-3_1-405b-instruct) into a single pipeline. Additionally, this tutorial also demonstrates advanced functions such as reward score assignment via [Nemotron-4 340B Reward](https://build.nvidia.com/nvidia/nemotron-4-340b-reward), as well as semantic deduplication to remove semantically similar real or synthetic records.
5. **[pretraining-data-curation](./pretraining-data-curation/)**, which overviews data curation pipeline for creating LLM pretraining dataset at scale.


## List of Tutorials

<div align="center">

| Tutorial | Description | Additional Resources |
| --- | --- | --- |
| [bitext_cleaning](./bitext_cleaning/) | Highlights several bitext-specific functionalities within NeMo Curator's API | |
| [curator-llm-pii](./curator-llm-pii/) | Demonstrates how to use NVIDIA's NeMo Curator library to modify text data containing Personally Identifiable Information (PII) using large language models (LLMs) | |
| [dapt-curation](./dapt-curation) | Data curation sample for domain-adaptive pre-training (DAPT), focusing on [ChipNeMo](https://blogs.nvidia.com/blog/llm-semiconductors-chip-nemo/) data curation as an example | [Blog post](https://developer.nvidia.com/blog/streamlining-data-processing-for-domain-adaptive-pretraining-with-nvidia-nemo-curator/) |
| [distributed_data_classification](./distributed_data_classification) | Demonstrates machine learning classification with NVIDIA's Hugging Face models at scale in a distributed environment | |
| [image-curation](./image-curation/) | Explores all of the functionality that NeMo Curator has for image dataset curation | |
| [llama-nemotron-data-curation](./llama-nemotron-data-curation/) | Demonstrates how a user can process a subset the Llama Nemotron dataset using NeMo Curator | |
| [multimodal_dapt_curation](./multimodal_dapt_curation/) | Covers multimodal extraction and data curation for domain-adaptive pre-training (DAPT) | |
| [nemo-retriever-synthetic-data-generation](./nemo_retriever_synthetic_data_generation) | Demonstrates the use of NeMo Curator synthetic data generation modules to leverage [NIM models](https://ai.nvidia.com) for generating synthetic data and perform data quality assesement on generated data using LLM-as-judge and embedding-model-as-judge. The generated data would be used to evaluate retrieval/RAG pipelines |
| [nemotron_340B_synthetic_datagen](./nemotron_340B_synthetic_datagen) | Demonstrates the use of NeMo Curator synthetic data generation modules to leverage [Nemotron-4 340B Instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct) for generating synthetic preference data | |
| [nemotron-cc](./nemotron-cc/) | How to use NeMo Curator to build the data curation pipeline used to create the Nemotron-CC dataset | [Blog post](https://developer.nvidia.com/blog/building-nemotron-cc-a-high-quality-trillion-token-dataset-for-llm-pretraining-from-common-crawl-using-nvidia-nemo-curator/) |
| [peft-curation](./peft-curation/) | Data curation sample for parameter efficient fine-tuning (PEFT) use-cases | [Blog post](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-parameter-efficient-fine-tuning-with-nvidia-nemo-curator/) |
| [peft-curation-with-sdg](./peft-curation/) | Demonstrates a pipeline to leverage external models such as [Nemotron-4 340B Instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct) for synthetic data generation, data quality annotation via [Nemotron-4 340B Reward](https://build.nvidia.com/nvidia/nemotron-4-340b-reward), as well as other data processing steps (semantic deduplication, HTML tag removal, etc.) for parameter efficient fine-tuning (PEFT) use-cases  | [Use this data to fine-tune your own model](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama-3/sdg-law-title-generation/llama3-sdg-lora-nemofw.ipynb) |
| [pretraining-data-curation](./pretraining-data-curation/) | Demonstrates accelerated pipeline for curating large-scale data for LLM pretraining in a distributed environment | |
| [pretraining-vietnamese-data-curation](./pretraining-vietnamese-data-curation/) | Demonstrates how to use NeMo Curator to process large-scale and high-quality Vietnamese data in a distributed environment | |
| [single_node_tutorial](./single_node_tutorial) | A comprehensive example to demonstrate running various NeMo Curator functionalities locally | |
| [synthetic-data-hello-world](./synthetic-data-hello-world) | An introductory example of synthetic data generation using NeMo Curator | |
| [synthetic-preference-data](./synthetic-preference-data) | Demonstrates the use of NeMo Curator synthetic data generation modules to leverage [LLaMa 3.1 405B Instruct](https://build.nvidia.com/meta/llama-3_1-405b-instruct) for generating synthetic preference data |
| [synthetic-retrieval-evaluation](./synthetic-retrieval-evaluation) | Demonstrates the use of NeMo Curator synthetic data generation modules to leverage [LLaMa 3.1 405B Instruct](https://build.nvidia.com/meta/llama-3_1-405b-instruct) for generating synthetic data to evaluate retrieval pipelines |
| [tinystories](./tinystories) | A comprehensive example of curating a small dataset to use for model pre-training. | [Blog post](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/)
| [zyda2-tutorial](./zyda2-tutorial) | A comprehensive tutorial on how to reproduce [Zyda2 dataset](https://huggingface.co/datasets/Zyphra/Zyda2) with NeMo Curator. | [Nvidia blog post](https://developer.nvidia.com/blog/train-highly-accurate-llms-with-the-zyda-2-open-5t-token-dataset-processed-with-nvidia-nemo-curator/) [Zyphra blog post](https://www.zyphra.com/post/building-zyda-2)
</div>
