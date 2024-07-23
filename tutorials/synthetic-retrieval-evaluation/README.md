# Synthetic Data for Evaluating Retrieval Pipelines

This example walksthrough how to generate Synthetic Data using the LLaMa 3.1 405B model.

## How to Run the example

```
docker run -it --net=host --gpus all -v path/to/dir:/mount nvcr.io/nvidia/pytorch:24.02-py3
cd /mount
pip install openai
export BUILD_NVIDIA_API_KEY=nvapi-...
jupyter notebook
```
