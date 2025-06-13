# PII Modification with NeMo Curator and Large Language Models
This tutorial demonstrates how to use NVIDIA's NeMo Curator library to modify text data containing Personally Identifiable Information (PII) using large language models (LLMs). We'll explore both asynchronous and synchronous approaches using `AsyncLLMPiiModifier` and `LLMPiiModifier`.

PII modification with NeMo Curator provides a sophisticated approach to privacy protection while maintaining data utility. The LLM-based modifiers offer intelligent, context-aware transformations that preserve the natural flow and usefulness of the dataset.

## Usage

The tutorial follows the steps below:

# Install Nemo Curator
Please follow the instructions in NeMo Curator's [README](https://github.com/NVIDIA/NeMo-Curator/tree/main?tab=readme-ov-file#install-nemo-curator) to install the NeMo Curator package and import libraries

Step 1: Pull docker container from NGC
```
docker pull nvcr.io/nvidia/nemo:25.04.rc2

```
Step 2: Once container image is pulled, start and exec into the container

```
docker run -it --gpus=all \
-p 8888:8888 \
-v /var/run/docker.sock:/var/run/docker.sock \
-v ${PWD}/workspace/nemo-test:/workspace/nemo-test \
nvcr.io/nvidia/nemo:25.04.rc2 bash

```

Step 3: Start up Jupyter lab once in the container with `Jupyter lab`

Step 4: Run through notebooks 1 and 2 in the workspace
