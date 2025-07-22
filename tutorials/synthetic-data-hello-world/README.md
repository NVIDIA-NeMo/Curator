# Synthetic Data Generation: Hello World Example

The provided notebook will walk you through the currently available Synthetic Generation tools and pipelines available out-of-the-box through NeMo Curator!

We'll walk through an example of each pipeline, as well as how you could make modifications to the provided pipelines.

> NOTE: Currently, the `convert_response_to_yaml_list()` method is extremely strict - manual parsing of the intermediate results is recommended in all cases. In the notebook we have wrapped these in `try/except` blocks to ensure you can move through the notebook without being impeded by the error.

### Covered Tools:

Through the following tools, NeMo Curator offers the following tools, which are compatible with both OpenAI API compatible models hosted on `build.nvidia.com`, as well as any LLM NIM that is locally running.

- NeMo Curator OpenAI Client (Sync and Async)
- Model Usage
- Synthetic Data Generation

### Covered Pipelines:

Through the use of the `NemotronGenerator`, NeMo Curator offers the following pipelines:

- Math Question Generation Pipeline
- Writing Task Generation Pipeline
- Open Question Generation Pipeline
- Closed Question Generation Pipeline
- Python Question Generation Pipeline
- Dialogue Generation Pipeline
- Two-Turn Prompt Generation Pipeline
- Entity Classification
    - Classify Math Entity
    - Classify Python Entity

> NOTE: If you are using the `build.nvidia.com` endpoint for Llama-3.3-Nemotron-Super-49B-v1 as your model for the above pipelines, during times of high load, it's possible that pipelines might time-out. In this case, we would recommend running the pipeline in a piecewise fashion and saving the intermediate outputs.
