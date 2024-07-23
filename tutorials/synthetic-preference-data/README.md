# Synthetic Preference Data Generation Using Nemotron-4 340B

The provided notebook will demonstrate how to leverage [Llama 3.1 405B Instruct](https://build.nvidia.com/meta/llama3.1-405b-instruct), and [Nemotron-4 340B Reward](https://build.nvidia.com/nvidia/nemotron-4-340b-reward) through [build.nvidia.com](https://build.nvidia.com/explore/discover).

The build will be a demonstration of the following pipeline!

![image](./SDG%20Pipeline.png)

The pipeline is designed to create a preference dataset suitable for training a custom reward model using the [SteerLM method](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/steerlm.html), however consecutive responses (e.g. sample 1 with 2, 3 with 4, etc.) share the same prompt so the dataset can also be used for preference pairs for training an RLHF Reward Model or for DPO - using the helpfulness score.
