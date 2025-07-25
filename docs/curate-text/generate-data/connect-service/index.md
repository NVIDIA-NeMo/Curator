---
description: "Connect to large language model services using OpenAI API format or NeMo Deploy for synthetic data generation"
categories: ["workflows"]
tags: ["llm-services", "openai", "nemo-deploy", "api-integration", "reward-models"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-generate-data-connect-service)=
# Connect to an LLM Service

Connect to large language model services using the [OpenAI API](https://github.com/openai/openai-python?tab=readme-ov-file#openai-python-api-library) format or [NeMo Deploy](https://docs.nvidia.com/nemo-framework/user-guide/latest/deployment/index.html). The OpenAI API format enables querying models across many platforms beyond OpenAI's own services.

## Choosing a Service

Consider the following comparison of features to help select the service type that best matches your deployment requirements and infrastructure preferences:

:::{list-table} Service Comparison
:header-rows: 1
:stub-columns: 1
:widths: 20 40 40

* - Feature
  - OpenAI API Compatible Services
  - NeMo Deploy
* - **Hosting**
  - Externally hosted with rate limits
  - Self-hosted with unlimited queries
* - **Setup**
  - Minimal setup required
  - More control and better performance
* - **Models**
  - Works with any compatible service
  - Optimized for NVIDIA models
:::

---

## Implementation Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`link-external;1.5em;sd-mr-1` OpenAI Compatible Services
:link: text-generate-data-connect-service-openai
:link-type: ref
Connect to hosted model endpoints using the OpenAI API format
+++
{bdg-secondary}`hosted-models`
{bdg-secondary}`openai`
{bdg-secondary}`api`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` NeMo Deploy
:link: text-generate-data-connect-service-nemo-deploy
:link-type: ref
Deploy and connect to your own self-hosted model endpoints
+++
{bdg-secondary}`self-hosted`
{bdg-secondary}`deployment`
{bdg-secondary}`performance`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Reward Models
:link: text-generate-data-connect-service-reward-models
:link-type: ref
Query reward models to score conversations and filter datasets
+++
{bdg-secondary}`scoring`
{bdg-secondary}`filtering`
{bdg-secondary}`quality`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

OpenAI <openai>
NeMo Deploy <nemo-deploy>
Reward Models <reward-model>
```
