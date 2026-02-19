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

### Stage and Pipeline Benchmarking

Benchmarking framework for performance monitoring:

- **Stage and Pipeline Benchmarking**: Automated benchmarks for curation modalities (text, image, video, audio)
- **Performance Tracking**: Metrics tracking across:
  - Text pipelines: exact deduplication, fuzzy deduplication, semantic deduplication, score filters, modifiers
  - Image curation workflows with DALI-based processing
  - Video processing pipelines with splitting, scene detection, captioning, and semantic deduplication
  - Audio ASR inference and quality assessment

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
python run.py --config-path ./text --config-name heuristic_filter_english_pipeline.yaml input_path=./input_dir output_path=./output_dir
```

### Pipeline Performance and Metric Logging

Enhanced tracking of pipeline execution:

- **Performance Metrics**: Automatic tracking of processing time, throughput, and resource usage
- **Better Debugging**: Detailed logs and error reporting for failed stages

## Improvements from 25.09

### Video Curation

- **Model Updates**: Removed InternVideo2 dependency; updated to more performant alternatives
- **vLLM 0.15.1**: Upgraded for better video captioning compatibility and performance
- **FFmpeg 8.0.1**: Latest FFmpeg with improved codec support and performance
- **Enhanced Tutorials**: Improved video processing examples with real-world scenarios

### Audio Curation

- **Enhanced Documentation**: Comprehensive ASR inference and quality assessment guides
- **Improved WER Filtering**: Better guidance for Word Error Rate filtering thresholds
- **Manifest Handling**: More robust JSONL manifest processing for large audio datasets

### Image Curation

- **Optimized Batch Sizes**: Configurable batch sizes for better CPU/GPU memory usage (batch_size=100, num_threads=16)
- **Memory Guidance**: Added troubleshooting documentation for out-of-memory errors
- **Tutorial Improvements**: Updated examples optimized for typical GPU configurations

### Text Curation

- **Better Memory Management**: Improved handling of large-scale semantic deduplication

### Deduplication Enhancements

- **Cloud Storage Support**: Fixed ParquetReader/Writer and pairwise I/O for S3, GCS, and Azure Blob
- **Non-Blocking ID Generation**: Improved ID generator performance for large datasets
- **Empty Batch Handling**: Better error handling for filters processing empty data batches

## Dependency Updates

- **Transformers**: Pinned to 4.55.2 for stability and compatibility
- **vLLM**: Updated to 0.15.1 with video pipeline compatibility fixes
- **FFmpeg**: Upgraded to 8.0.1 for enhanced multimedia processing
- **Security Patches**:
  - Addressed CVEs in aiohttp, urllib3, python-multipart, setuptools
  - Removed vulnerable thirdparty aiohttp file from Ray
  - Updated to secure dependency versions

## Bug Fixes

- Fixed fasttext predict call compatibility with numpy>2
- Fixed broken NeMo Framework documentation links
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
- **Cursor Rules**: Development guidelines and patterns for IDE assistance

## Breaking Changes

- **InternVideo2 Removed**: Video pipelines must use alternative embedding models (Cosmos-Embed1)

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
