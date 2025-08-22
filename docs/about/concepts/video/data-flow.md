---
description: "Understanding data flow in video curation pipelines including Ray object store and streaming optimization"
categories: ["concepts-architecture"]
tags: ["data-flow", "distributed", "ray", "streaming", "performance", "video-curation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "video-only"
---

(about-concepts-video-data-flow)=
# Data Flow

Understanding how data moves through NeMo Curator's video curation pipelines is key to optimizing performance and resource usage.

- Data moves between stages via Ray's distributed object store, enabling efficient, in-memory transfer between distributed actors.
- In streaming mode, the executor returns only final stage outputs while intermediate state stays in memory, reducing I/O overhead and improving throughput.
- The autoscaler continuously balances resources to maximize pipeline throughput, dynamically allocating workers to stages as needed.
- Writer stages persist outputs at the end of the pipeline, including clip media, embeddings (pickle and parquet variants), and metadata JSON files.

This architecture ensures that large-scale video datasets can be processed efficiently, with minimal data movement and optimal use of available hardware. 