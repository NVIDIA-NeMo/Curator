(text-gen-data-pipelines-writing-task)=
# Writing Task Generation Pipeline

This pipeline generates writing prompts (for example, essays, poems) for dialogue data, as used in Nemotron-4 340B.

## Steps
1. Generate tasks to write an email, essay, etc. about a topic
2. Revise the tasks to be more detailed

## Example Usage

```python
from nemo_curator.synthetic import NemotronGenerator
from nemo_curator import OpenAIClient  # or appropriate client
from openai import OpenAI

# Initialize the client and generator
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="your-api-key")
llm_client = OpenAIClient(client)
generator = NemotronGenerator(llm_client)

model = "mistralai/mixtral-8x7b-instruct-v0.1"

writing_tasks_responses = generator.generate_writing_tasks(
    topic="Climate Change and Sustainable Living",
    text_material_type="Poems",
    n_openlines=5,
    model=model,
)
writing_tasks_list = generator.convert_response_to_yaml_list(
    writing_tasks_responses[0], model=model
)

revised_writing_tasks_responses = generator.revise_writing_tasks(
    openline=writing_tasks_list[0], n_revisions=5, model=model
)
revised_writing_tasks = generator.convert_response_to_yaml_list(
    revised_writing_tasks_responses[0], model=model
)
```

### End-to-End Pipeline

```python
writing_tasks = generator.run_writing_pipeline(
    topics=[
        "Climate Change and Sustainable Living",
        "Space Exploration and the Universe",
        ...,
    ],
    text_material_types=["Poems", "Essays", ...],
    n_openlines=5,
    n_revisions=2,
    model=model,
)

print(writing_tasks[0])
# Output:
# Write a poem about the most effective sources of renewable energy.
``` 