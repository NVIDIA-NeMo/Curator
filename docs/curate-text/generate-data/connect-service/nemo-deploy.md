---
description: "Deploy and connect to self-hosted model endpoints using NVIDIA NeMo's Export and Deploy module for unlimited queries"
categories: ["how-to-guides"]
tags: ["nemo-deploy", "self-hosted", "deployment", "conversation-formatting", "performance"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-generate-data-connect-service-nemo-deploy)=
# NeMo Deploy

Deploy and connect to your own self-hosted model endpoints using [NeMo's Export and Deploy](https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference) module.

## Before You Start

### Model Name Specification

When initializing `NemoQueryLLM`, specify the model's name. While NemoQueryLLM is built for querying a single model, NeMo Curator allows changing the queried model on your local server for each request.

### Conversation Formatting

Large language models take a tokenized string as input, not a list of conversation turns. Each model uses a specific format during alignment. For example, Mixtral-8x7B-Instruct-v0.1 uses:

```sh
<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
```

While OpenAI API services handle this formatting internally, with NeMo Deploy you must specify the format. NeMo Curator provides formatters for common models:

- `Mixtral8x7BFormatter` for Mixtral-8x7B-Instruct-v0.1
- `NemotronFormatter` for Nemotron-4 340B 

---

## Usage

After deploying a model following the [NeMo Deploy Guide](https://docs.nvidia.com/nemo-framework/user-guide/24.09/deployment/llm/optimized/tensorrt_llm.html), you can query it like this:

```python
from nemo.deploy.nlp import NemoQueryLLM
from nemo_curator import NemoDeployClient
from nemo_curator.synthetic import Mixtral8x7BFormatter

model = "mistralai/mixtral-8x7b-instruct-v0.1"
nemo_client = NemoQueryLLM(url="localhost:8000", model_name=model)
client = NemoDeployClient(nemo_client)
responses = client.query_model(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Write a limerick about the wonders of GPU computing.",
        }
    ],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    conversation_formatter=Mixtral8x7BFormatter(),
)
print(responses[0])
```
