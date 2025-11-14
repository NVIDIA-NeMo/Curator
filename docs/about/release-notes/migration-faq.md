---
description: "Frequently asked questions about migrating NeMo Curator workflows from Dask to Ray"
categories: ["migration", "release-notes"]
tags: ["migration", "dask", "ray", "faq", "troubleshooting"]
personas: ["data-scientist-focused", "mle-focused", "devops-focused"]
difficulty: "beginner"
content_type: "faq"
modality: "universal"
---

(migration-faq)=

# NeMo Curator Migration FAQ

Frequently asked questions about migrating from Dask to Ray-based NeMo Curator.

```{seealso}
For step-by-step migration instructions, refer to the {ref}`Migration Guide <migration-guide>`.
```

---

## General Architecture and Migration

```{dropdown} Why is NeMo Curator migrating from Dask to Ray? What are the main benefits?

The migration aims to deliver significantly better scalability and performance, especially for large, heterogeneous, and GPU-intensive workflows. Ray simplifies cluster resource management and enables more dynamic scheduling and throughput optimization compared to Dask. The underlying orchestration and task management are completely rewritten for efficiency.
```

```{dropdown} Is the new Ray-based pipeline faster than the Dask or Spark versions? Will you be publishing benchmarks?

Internal benchmarks show Ray matches or outperforms previous implementations, especially on multi-GPU clusters and for streaming, model-heavy workloads. We are finalizing transparent benchmark numbers and reproducible environments so users can run their own comparisons and verify performance in their clusters.
```

```{dropdown} Will we need to completely refactor our pipelines to switch to Ray?

Some migration effort is expected. High-level abstractions—like tasks and stages—still map well, but the new system follows a strictly linear, stage-wise architecture (meaning each stage feeds directly into the next, without branches or fan-outs). If your pipeline structure or branching logic differs, adaptation will be needed. Additionally, Ray cluster deployment may require some effort if you have pre-existing automation or scripts for Dask or Spark.

:::{note}
Ray pipelines in NeMo Curator are strictly linear. If you need to apply multiple independent filters or models in parallel, combine them within a single stage (you can parallelize inside your code using subprocesses if needed). Direct fan-out or branching between stages is not supported to maximize throughput optimization and scheduling simplicity.
:::
```

---

## Pipeline Design and Operation

```{dropdown} How is the Ray pipeline structured? Is it streaming, batch, or hybrid?

Ray pipelines use a {ref}`streaming architecture <about-concepts-video-architecture>` where data tasks flow from stage to stage in-memory wherever possible (across Ray's object store), minimizing costly file system I/O. Production runs process data continuously for maximum throughput. For more details, refer to {ref}`Key Abstractions <about-concepts-video-abstractions>`.
```

```{dropdown} How do I save or checkpoint intermediate results between stages?

Each stage can write outputs or checkpoints to disk as needed (for recovery or later analysis). Pipeline stages do not wait for these writes to complete; instead, writes should be implemented in a non-blocking way inside your stage logic. For most use cases, this allows the main dataflow and processing to remain high-throughput and non-blocking. If you need to guarantee write success and handle failures, you can use subprocess-based logging or retry logic within your process functions.
```

```{dropdown} Does each stage in the pipeline have to be run sequentially, or can we do things in parallel or have branching logic?

Pipelines must be linear (no branching or fan-out between stages). If you need to apply multiple independent filters or models, group them together into a single stage and manage that in your own code (possibly with subprocesses).
```

```{dropdown} Can I process multiple data sources or snapshots at once?

Yes. The task creation stage lists all work units (for example, files from all snapshots) upfront. Ray then orchestrates parallel processing, maximizing resource utilization across the whole dataset—streaming batches through as tasks complete.
```

```{dropdown} We often want to keep "dropped" data (for example, failed filters) for further analysis. Can we do that?

Yes. With the new approach, you can configure logic in your process functions to write out filtered or dropped examples as a "splitting" operation rather than a hard filter—enabling later analysis, retraining, or recycling of low-quality data.
```

---

## Resource Management and Performance Optimization

```{dropdown} How does the system allocate resources across stages with different speeds or requirements?

An internal adaptive scheduler monitors throughput for each stage every few minutes (configurable) and dynamically adjusts worker counts and allocations to maximize total pipeline throughput and avoid bottlenecks.
```

```{dropdown} How do I specify resource requirements (GPUs, CPUs, RAM) for a pipeline stage?

Each stage specifies its own resource needs in code (see {ref}`Pipeline Execution Backends <reference-execution-backends>` for configuration details). Set required CPU or GPU count, GPU VRAM, and other specs directly. Ray packs tasks optimally (for example, several light jobs on one GPU).
```

```{dropdown} What happens if a model's actual memory usage exceeds what I specified (OOM errors)?

Ray will restart any failed actor (worker), but handling logic for OOM or other errors is up to your code. Use try/except blocks to decide whether to retry, skip, log, or stop.
```

```{dropdown} Can NeMo Curator scale to multi-node, multi-GPU clusters?

Yes. Ray natively orchestrates jobs across nodes and GPUs. For multi-node operations (for example, distributed model training or global deduplication), you'll set up communication (such as NCCL for PyTorch) within the relevant stage's setup logic.
```

```{dropdown} How does the system manage GPU locality and optimal task placement?

Internally, an allocator ensures data locality for heavy data transfer stages (for example, tasks passing large data between GPU-intensive stages tend to run on the same GPU or node to minimize transfer times). These are handled by [Cosmos Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) or Ray's underlying allocators and optimization routines—not directly inside NeMo Curator itself. These frameworks attempt to colocate data and computation to minimize data transfer and maximize throughput, according to your resource constraints and specifications.
```

---

## Data Processing, Models, and Quality

```{dropdown} How do you handle quality filtering for low-resource languages where no good models exist?

NeMo Curator suggests several strategies (see {ref}`Heuristic Filtering <text-process-data-filter-heuristic>` and {ref}`Quality Assessment <about-concepts-text-data-processing>` for details):

- Use available multilingual models (for example, Qwen, Mistral, or other models with many language capabilities)
- Annotate high-quality English data with a classifier, translate these data to the target language, and then train a smaller in-language model
- Upsample high-quality data or downsample and rewrite lower-quality data instead of discarding it (using an LLM rewrite for quality improvement)
```

```{dropdown} What is the "ensemble model bucketing" workflow?

A set of models assigns a quality score (for example, 0–20), bucketed into high, medium, or low groups. High-quality data might be upsampled or have synthetic variants generated; low-quality data is rewritten rather than simply dropped—especially valuable for low-resource contexts.
```

```{dropdown} Can the system track the number of documents or tokens processed, filtered, or passed at each stage?

Each task is a data class. You can add whatever statistics you need (input or output counts, tokens dropped, and so on) within stage logic for detailed reporting or logging.
```

---

## Fault Tolerance, Checkpointing, and Observability

```{dropdown} What if a stage fails (for example, OOM or disk error)? Will the pipeline crash?

Ray will relaunch failed workers or actors, but robust error handling and resumption (for example, from last completed task) should be implemented in your stage logic as needed.
```

```{dropdown} How is observability handled? Can I track pipeline performance, actor counts, and task durations?

All Ray pipelines expose resource and processing metrics via a built-in Grafana dashboard (with process time per task or actor, resource utilization, and so on). You can also summarize stats from task data at pipeline completion for custom reporting. For configuration options, refer to {ref}`Pipeline Execution Backends <reference-execution-backends>` and the [Ray Dashboard documentation](https://docs.ray.io/en/latest/ray-observability/getting-started.html).
```

```{dropdown} Can I resume processing from mid-pipeline if interrupted?

For streaming pipelines, each completed task can be tracked (temporarily with final outputs), so resumption logic is "start from task N+1". If you rely on explicit intermediate checkpoints, you can extend the process logic to save and reload state as needed.
```

---

## Extensibility, Customization, and Collaboration

```{dropdown} How customizable is the pipeline? Can I easily add my own stages, models, or data annotations?

Yes, by design. Add new stages or modify process functions to integrate custom logic, models, or data preprocessing and postprocessing (see {ref}`Key Abstractions <about-concepts-video-abstractions>` for examples). Extend or fork example pipelines to suit new use cases.
```

```{dropdown} Can I contribute region or language-specific filters or tools back to NeMo Curator?

Contributions are strongly encouraged. Submit pull requests or join community discussions to help expand NeMo Curator's capabilities for diverse regions and languages.
```

---

## Deployment, Infrastructure, and Practicalities

```{dropdown} How do I deploy NeMo Curator or Ray clusters?

Ray clusters can be deployed on any major cloud platform (AWS, GCP, Azure) using standard Ray tools (see [Ray documentation](https://docs.ray.io/en/latest/cluster/getting-started.html)). No custom infrastructure is needed. NVIDIA provides ready-to-use {ref}`Docker images <reference-infrastructure-container-environments>` and up-to-date quickstart guides. For complete deployment details, refer to {ref}`Production Deployment Requirements <admin-deployment-requirements>`.
```

```{dropdown} Do I need to build custom Docker images?

Not for most standard uses. Use {ref}`provided images <reference-infrastructure-container-environments>` or extend as needed (for example, to add proprietary or additional filters). Check out the official Docker container releases on the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator).
```

```{dropdown} Is there integration with existing cluster or orchestration tools like Airflow or Slurm?

Ray manages its own orchestration natively. For more traditional batch job orchestration, Airflow or Slurm can be used to launch clusters or submit jobs, but within a Ray-run pipeline orchestration is handled within Ray.
```

---

## Advanced Topics

```{dropdown} What if I need a fan-in or reduce operation (not just map or scatter)?

The current model supports scatter and map, but not reduce or fan-in across tasks (for global aggregations). For most data curation workflows, this is not a limitation; custom logic can be used where needed.
```

```{dropdown} How do I manage large, multi-node training jobs with fine-grained control (for example, PyTorch-NCCL setup)?

For multi-node operations, you're responsible for managing communication setup (for example, establishing NCCL channels for distributed training within the relevant stage's setup logic).
```

```{dropdown} Are there examples of multi-type data (text, image, audio, video) pipelines?

Yes. NeMo Curator supports multiple data modalities including {ref}`text <gs-text>`, {ref}`image <gs-image>`, {ref}`audio <gs-audio>`, and {ref}`video <gs-video>`.
```

---

## Support, Communication, and Community

```{dropdown} Where can I get support, ask questions, or contribute?

All support and community engagement happens on [GitHub](https://github.com/NVIDIA-NeMo/Curator). We encourage you to:

- **Open an issue** for bugs, feature requests, or questions
- **Start a discussion** to share ideas or ask for guidance
- **Submit a pull request** if you build innovative filters or features

Regular community calls and check-ins are also offered to connect with the team and other users.
```

---

## Additional Resources

```{seealso}
- [Ray Documentation and Getting Started](https://docs.ray.io/)
- {ref}`Migration Guide <migration-guide>` for step-by-step migration instructions
- [NGC Container Catalog](https://catalog.ngc.nvidia.com/) for Docker images
```

```{tip}
If you find something missing or want to share a best practice or feature, please [open an issue](https://github.com/NVIDIA-NeMo/Curator/issues) or [submit a pull request](https://github.com/NVIDIA-NeMo/Curator/pulls) on GitHub.
```
