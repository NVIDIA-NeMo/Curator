---
description: "Generate multi-turn dialogues and two-turn prompts for preference data using LLMs playing both user and assistant roles"
categories: ["how-to-guides"]
tags: ["dialogue", "multi-turn", "conversation", "preference-data", "two-turn", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-dialogue)=
# Dialogue Generation Pipeline

This pipeline generates multi-turn dialogues and two-turn prompts for preference data, as used in Nemotron-4 340B. The pipeline creates conversations where an LLM alternates between playing the user and assistant roles, generating realistic dialogue exchanges.

## Before You Start

- **LLM Client Setup**: The `NemotronGenerator` requires an `LLMClient` instance to interface with language models. Refer to the {ref}`LLM services documentation <text-generate-data-connect-service>` for details on configuring your client with specific model providers.

---

## Setup Steps

### Set up the LLM Client

Configure your LLM client (example with OpenAI):

```python
from openai import OpenAI

openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="<insert NVIDIA API key>"
)
```

### Create the NeMo Curator Client Wrapper

Wrap the client with NeMo Curator's client wrapper:

```python
from nemo_curator.services import OpenAIClient

llm_client = OpenAIClient(openai_client)
```

### Initialize the Generator

Create the NemotronGenerator instance:

```python
from nemo_curator.synthetic import NemotronGenerator

generator = NemotronGenerator(llm_client)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "mistralai/mixtral-8x7b-instruct-v0.1"
openline = "Write a poem about the moon."
n_user_turns = 3
```

### Generate Dialogue

Choose between multi-turn dialogues for full conversations or two-turn prompts for preference data:

:::: {tab-set}

::: {tab-item} Multi-Turn Dialogue
:sync: sync-multi

Use the generator to create a multi-turn conversation:

```python
dialogue = generator.generate_dialogue(
    openline=openline,
    user_model=model,
    assistant_model=model,
    n_user_turns=n_user_turns,
)

print(dialogue)
# Output:
# [{"role": "user", "content": "Write a poem about the moon."},
#  {"role": "assistant", "content": "The silver moon hangs in the night sky..."},
#  {"role": "user", "content": "Can you make it more romantic?"},
#  {"role": "assistant", "content": "Beneath the moon's enchanting glow..."},
#  {"role": "user", "content": "Add some imagery about stars too."},
#  {"role": "assistant", "content": "The moon and stars dance together..."}]
```

:::

::: {tab-item} Two-Turn Prompts
:sync: sync-two

For preference data creation, generate shorter two-turn conversations:

```python
two_turn_dialogue = generator.generate_two_turn_prompt(
    openline=openline,
    user_model=model,
    assistant_model=model,
)

print(two_turn_dialogue)
# Output:
# [{"role": "user", "content": "Write a poem about the moon."},
#  {"role": "assistant", "content": "The moon is bright. It shines at night."},
#  {"role": "user", "content": "Can you make the poem longer?"}]
```

:::

::::

## Advanced Configuration

Customize the dialogue generation with additional parameters:

```python
# Use different models for user and assistant
user_model = "mistralai/mixtral-8x7b-instruct-v0.1"
assistant_model = "meta/llama-3.1-8b-instruct"

# Configure model-specific parameters
user_model_kwargs = {"temperature": 0.8, "max_tokens": 100}
assistant_model_kwargs = {"temperature": 0.5, "max_tokens": 200}

dialogue = generator.generate_dialogue(
    openline="Explain quantum physics in simple terms.",
    user_model=user_model,
    assistant_model=assistant_model,
    n_user_turns=4,
    user_model_kwargs=user_model_kwargs,
    assistant_model_kwargs=assistant_model_kwargs,
)
```
