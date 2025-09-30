---
description: "Deploy NeMo Curator in production environments"
categories: ["workflows"]
tags: ["deployment", "production", "infrastructure"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "universal"
orphan: true
---

(admin-deployment)=

# Deploy NeMo Curator

Use the following Admin guides to set up NeMo Curator in a production environment.

## Prerequisites

Before deploying NeMo Curator in a production environment, review the comprehensive requirements:

- **System**: Ubuntu 22.04/20.04, Python 3.10, 3.11, or 3.12
- **Hardware**: Multi-core CPU, 16GB+ RAM (optional: NVIDIA GPU with 16GB+ VRAM)
- **Software**: Ray (distributed computing framework), container runtime
- **Infrastructure**: High-bandwidth networking, storage for datasets

For detailed system, hardware, and software requirements, see [Production Deployment Requirements](admin-deployment-requirements).

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Requirements <requirements>

```