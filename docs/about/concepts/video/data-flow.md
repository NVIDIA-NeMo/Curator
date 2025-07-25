---
description: "Understanding data flow in video curation pipelines including Ray object store and streaming optimization"
categories: ["concepts-architecture"]
tags: ["data-flow", "distributed", "ray", "streaming", "performance", "video-curation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "video-only"
only: not ga
---

(about-concepts-video-data-flow)=
# Data Flow

Understanding how data moves through NeMo Curator's video curation pipelines is key to optimizing performance and resource usage.

- Data moves between stages via Ray's distributed object store, allowing for efficient, in-memory data transfer between distributed actors.
- In streaming mode, only the final results are written to disk, reducing I/O overhead and improving throughput.
- The autoscaler continuously balances resources to maximize pipeline throughput, dynamically allocating workers to stages as needed.

This architecture ensures that large-scale video datasets can be processed efficiently, with minimal data movement and optimal use of available hardware. 