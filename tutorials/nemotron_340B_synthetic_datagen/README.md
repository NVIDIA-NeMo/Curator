# Synthetic Preference Data Generation Using Nemotron-4 340B

The provided notebook will demonstrate how to leverage [Nemotron-4 340B Instruct](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct), and [Nemotron-4 340B Reward](https://build.nvidia.com/nvidia/nemotron-4-340b-reward) through [build.nvidia.com](https://build.nvidia.com/explore/discover).

The build will be a demonstration of the following pipeline, as discuss in the [release blog](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/), and [technical blog](https://developer.nvidia.com/blog/leverage-our-latest-open-models-for-synthetic-data-generation-with-nvidia-nemotron-4-340b/). The pipeline is designed to create a preference dataset suitable for training a custom reward model using the [SteerLM method](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/steerlm.html), however consecutive responses (e.g. sample 1 with 2, 3 with 4, etc.) share the same prompt so the dataset can also be used for preference pairs for training an RLHF Reward Model or for DPO - using the helpfulness score.

![image](https://developer-blogs.nvidia.com/wp-content/uploads/2024/06/SDG-Pipeline-1-625x352.png)

> NOTE: There are no specific dependencies outside of those outlined in the notebook for this tutorial!
