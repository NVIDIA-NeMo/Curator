
.. _data-curator-syntheticdata:

======================================
Synthetic Data Generation
======================================
--------------------------------------
Background
--------------------------------------
Synthetic data generation has become increasing useful in large language model training.
It is used in pretraining, fine-tuning, and evalutation.
Synthetically generated data can be useful for adapting an LLM to low resource languages/domains, or performing knowledge distillation from other models among other purposes.
There are a variety of ways to construct synthetic data generation pipelines, with numerous LLM and classical filters.

NeMo Curator has a simple, easy-to-use set of tools that allow you to use prebuilt synthetic generation pipelines or build your own.
Any model inference service that uses the OpenAI API is compatible with the synthetic data generation module, allowing you to generate your data from any model.
NeMo Curator has prebuilt synthetic data generation pipelines for supervised fine-tuning (SFT) and preference data that were used to generate data for the training of `Nemotron-4 340B <https://research.nvidia.com/publication/2024-06_nemotron-4-340b>`_.
And, you can easily interweave filtering and deduplication steps in your synthetic data pipeline with the other modules in NeMo Curator.