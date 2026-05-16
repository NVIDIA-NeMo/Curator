---
description: "Advanced tutorials for customizing NeMo Curator video pipelines including adding custom code, models, stages, and environments"
categories: ["video-curation"]
tags: ["customization", "pipeline", "advanced", "models", "docker", "environments", "custom-code"]
personas: ["mle-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-pipeline-cust-series)=
# Video Pipeline Customization Tutorials

Use the tutorials in this section to learn how to customize video pipelines using NeMo Curator.

You can customize the **environments**, **code**, **models**, and **stages** used in NeMo Curator pipelines.

## Before You Start

Before you begin customizing NeMo Curator pipelines, make sure that you have:

- Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
- A working NeMo Curator development environment.  
- Reviewed the [container environments](reference-infrastructure-container-environments) reference.

---

```{tip}
Tutorials are organized by complexity and typically build on one another.
```

---

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Add Custom Environment
:link: video-tutorials-pipeline-cust-env
:link-type: ref
Learn how to create and configure custom environments for your video pipelines, including setting up dependencies and runtime configurations.
+++
{bdg-secondary}`container`
{bdg-secondary}`environments`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Add Custom Code
:link: video-tutorials-pipeline-cust-add-code
:link-type: ref
Learn how to extend pipeline functionality by adding custom code modules, including data processing scripts and utility functions.
+++
{bdg-secondary}`container` {bdg-secondary}`code`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Add Custom Model
:link: video-tutorials-pipeline-cust-add-model
:link-type: ref
Learn how to integrate custom machine learning models into your video pipelines, including model loading, inference, and optimization.
+++
{bdg-secondary}`container` {bdg-secondary}`models`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Add Custom Stage
:link: video-tutorials-pipeline-cust-add-stage
:link-type: ref
Learn how to create and integrate custom pipeline stages to add new processing capabilities to your video curation workflow.
+++
{bdg-secondary}`container` {bdg-secondary}`environments` {bdg-secondary}`code` {bdg-secondary}`models` {bdg-secondary}`stages`
:::
::::

```{toctree}
:hidden:
:maxdepth: 4

add-cust-env.md
add-cust-code.md
add-cust-model.md
add-cust-stage.md
```
