---
description: "Release notes and version history for NeMo Curator platform updates and new features"
categories: ["reference"]
tags: ["release-notes", "changelog", "updates"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(about-release-notes)=

# NeMo Curator Release Notes: {{ current_release }}

## What's New in 26.02

### Benchmarking Infrastructure

New comprehensive benchmarking framework for performance monitoring and optimization:

- **End-to-End Pipeline Benchmarking**: Automated benchmarks for all curation modalities (text, image, video, audio)
- **Performance Tracking**: Integration with MLflow for metrics tracking and Slack for notifications
- **Nightly Benchmarks**: Continuous performance monitoring across:
  - Text pipelines: exact deduplication, fuzzy deduplication, semantic deduplication, score filters, modifiers
  - Image curation workflows with DALI-based processing
  - Video processing pipelines with scene detection and semantic deduplication
  - Audio ASR inference and quality assessment
- **Grafana Dashboards**: Real-time monitoring of pipeline performance and resource utilization

### Ray Actor Pool Executor Improvements

Enhanced features for the experimental Ray Actor Pool execution backend:

- **Progress Bars**: New visual feedback for long-running actor pool operations, making it easier to monitor pipeline execution
- **Improved Load Balancing**: Better worker distribution and task scheduling
- **Enhanced Stability**: Continued refinements to the experimental executor

Learn more in the [Execution Backends documentation](../../reference/infrastructure/execution-backends.md).

### Enhanced Embedding Generation

Expanded embedding support with new model integrations:

- **vLLM Integration**: High-performance LLM-based embedding generation with automatic batching
- **Sentence Transformers**: Support for popular sentence embedding models
- **Unified API**: Consistent embedding interface across text, image, and video modalities

### YAML Configuration Support

Declarative pipeline configuration for text curation workflows:

- **YAML-Based Pipelines**: Define entire curation pipelines in YAML configuration files
- **Pre-Built Configurations**: Ready-to-use configs for common workflows:
  - Code filtering, exact/fuzzy/semantic deduplication
  - Heuristic filtering (English and non-English)
  - FastText language identification
- **Reproducible Workflows**: Version-controlled pipeline definitions for consistent results

Example:
```bash
python -m nemo_curator.config.run --config_file heuristic_filter_english_pipeline.yaml
```

### Workflow Results API

New API for tracking and analyzing pipeline execution:

- **WorkflowRunResult**: Structured results object capturing execution metrics
- **Performance Metrics**: Automatic tracking of processing time, throughput, and resource usage
- **Better Debugging**: Detailed logs and error reporting for failed stages

## Improvements from 25.09

### Video Curation

- **Model Updates**: Removed InternVideo2 dependency; updated to more performant alternatives
- **vLLM 0.14.1**: Upgraded for better video captioning compatibility and performance
- **FFmpeg 8.0.1**: Latest FFmpeg with improved codec support and performance
- **Enhanced Tutorials**: Improved video processing examples with real-world scenarios

### Audio Curation

- **Enhanced Documentation**: Comprehensive ASR inference and quality assessment guides
- **Improved WER Filtering**: Better guidance for Word Error Rate filtering thresholds
- **Manifest Handling**: More robust JSONL manifest processing for large audio datasets

### Image Curation

- **Optimized Batch Sizes**: Reduced default batch sizes for better CPU memory usage (batch_size=50, num_threads=4)
- **Memory Guidance**: Added troubleshooting documentation for out-of-memory errors
- **Tutorial Improvements**: Updated examples optimized for typical GPU configurations

### Text Curation

- **ID Field Standardization**: Unified ID naming conventions across all deduplication workflows
- **Performance Optimizations**: Fused document iterate and extract stages for reduced overhead
- **Better Memory Management**: Improved handling of large-scale semantic deduplication
- **Small Cluster Warnings**: Automatic warnings when n_clusters is too small for effective deduplication
- **FilePartitioning Improvements**: One worker per partition for better parallelization

### Deduplication Enhancements

- **Cloud Storage Support**: Fixed ParquetReader/Writer and pairwise I/O for S3, GCS, and Azure Blob
- **Non-Blocking ID Generation**: Improved ID generator performance for large datasets
- **Empty Batch Handling**: Better error handling for filters processing empty data batches

## Dependency Updates

- **Transformers**: Pinned to 4.55.2 for stability and compatibility
- **vLLM**: Updated to 0.14.1 with video pipeline compatibility fixes
- **FFmpeg**: Upgraded to 8.0.1 for enhanced multimedia processing
- **Security Patches**:
  - Addressed CVEs in aiohttp, urllib3, python-multipart, setuptools
  - Removed vulnerable thirdparty aiohttp file from Ray
  - Updated to secure dependency versions

## Bug Fixes

- Fixed fasttext predict call compatibility with numpy>2
- Fixed broken NeMo Framework documentation links
- Fixed MegatronTokenizerWriter to download only necessary tokenizer files
- Fixed ID generator blocking issues for large-scale processing
- Fixed vLLM API compatibility with video captioning pipeline
- Fixed Gliner tutorial examples and SDG workflow bugs
- Improved semantic deduplication unit test reliability

## Infrastructure & Developer Experience

- **Secrets Detection**: Automated secret scanning in CI/CD workflows
- **Dependabot Integration**: Automatic dependency update pull requests
- **Enhanced Install Tests**: Comprehensive installation validation across environments
- **AWS Runner Support**: CI/CD execution on AWS infrastructure
- **Docker Optimization**: Improved layer caching and build times with uv
- **Code Linting**: Standardized code quality checks with markdownlint and pre-commit hooks
- **Cursor Rules**: Development guidelines and patterns for IDE assistance

## Breaking Changes

- **InternVideo2 Removed**: Video pipelines must use alternative embedding models (Cosmos-Embed1)
- **ID Field Standardization**: Custom deduplication workflows may need updates to use standardized ID field names

## Documentation Improvements

- **Heuristic Filter Guide**: Comprehensive documentation for language-specific filtering strategies
- **Distributed Classifier**: Enhanced GPU memory optimization guidance with length-based sequence sorting
- **Installation Guide**: Clearer instructions with troubleshooting for common issues
- **Memory Management**: New guidance for handling CPU/GPU memory constraints
- **AWS Integration**: Updated tutorials with correct AWS credentials setup

---

## What's Next

Future releases will focus on:

- **Code Curation**: Specialized pipelines for curating code datasets
- **Math Curation**: Mathematical reasoning and problem-solving data curation
- **Generation Features**: Completing the Ray refactor for synthetic data generation
- **PII Processing**: Enhanced privacy-preserving data curation with Ray backend
- **Blending & Shuffling**: Large-scale multi-source dataset blending and shuffling operations

```{toctree}
:hidden:
:maxdepth: 4

Migration Guide <migration-guide>
Migration FAQ <migration-faq>

```
