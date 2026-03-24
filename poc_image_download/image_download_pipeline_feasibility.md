# Feasibility Analysis: Moving cc-img-dl to NeMo Curator

## Executive Summary

The `cc-img-dl` image download pipeline is a strong candidate for integration into NeMo Curator. The pipeline's architecture — manifest-driven URL enumeration, HTTP-based asset download, and local/cloud storage — maps cleanly onto Curator's existing `DocumentDownloadExtractStage` pattern used by Common Crawl, Wikipedia, and ArXiv. This document analyzes the feasibility, benefits, and tradeoffs of that move from two angles: **benefit to Curator** (long-term maintainability) and **benefit to the cc-img-dl team** (what they gain if their I/O-bound workload runs on Curator's distributed infrastructure).

---

## 1. Architecture Comparison

### cc-img-dl Pipeline Flow

> **TL;DR**
> ```
> Parquet Manifest ──read──▶ Validate URLs ──shard──▶ N JSONL shard files
>                                                         │
>                          ┌────────────────────────────────┘
>                          ▼  (per shard, in batches of 1000)
>                   Compliance Check ──filter──▶ Async Download ──write──▶ Local/S3
>                  (robots.txt + TDMRep)        (httpx + semaphore)
> ```

The pipeline has two phases: a **prep phase** that runs once, and a **per-shard worker loop** that runs N times (once per shard, potentially in parallel via AWS Batch).

**Prep phase (runs once):**

1. Read a Parquet manifest (`url_manifest.parquet`) containing columns: `url_id`, `asset_url`, `asset_mime`, `target_path`, `source_ref`
2. Validate each URL (reject non-HTTP, private IPs, loopback — SSRF protection)
3. Deterministically assign each URL to one of N shards via `sha256(url_id) % N`
4. Write N JSONL shard files to disk (one file per shard)

**Per-shard worker loop (runs once per shard):**

5. Load the shard's JSONL file and resume from the last checkpoint (if any)
6. For each batch of 1000 URLs:
   - a. **Compliance check**: Fetch and cache `robots.txt` and `.well-known/tdmrep.json` per origin. Evaluate each URL against both. Remove blocked URLs.
   - b. **Download**: Concurrently fetch allowed image URLs via async HTTP (bounded by `asyncio.Semaphore(concurrency)`). Retry 5xx with exponential backoff; respect 429 Retry-After; skip 4xx.
   - c. **Store**: Write downloaded bytes to local filesystem (or S3 in production).
   - d. **Checkpoint**: Save `last_url_id` to a JSON file every 100 downloads and at end of each batch.

### Curator's Existing Download Pattern (Text)

```
_EmptyTask → URLGenerationStage → DocumentDownloadStage → DocumentIterateExtractStage
                (FileGroupTask)      (FileGroupTask)          (DocumentBatch)
```

### Mapping

| cc-img-dl Concept | Curator Equivalent | Notes |
|---|---|---|
| `iter_records()` from Parquet | `URLGenerator.generate_urls()` | Reads manifest, yields URL batches |
| `write_shards()` + `shard_id_for()` | Pipeline task partitioning | Curator/Xenna handles this automatically via task-level parallelism |
| `ComplianceChecker` (robots + TDM) | New `ProcessingStage` (filter) | No Curator equivalent today; would be a new stage |
| `download_image()` | `DocumentDownloader._download_to_path()` | Very close analog; cc-img-dl uses async httpx, Curator uses subprocess/wget |
| `LocalStorage.put()` | `ImageWriterStage` or new writer | Curator already has image I/O stages |
| `Checkpoint` | Xenna fault tolerance | Xenna handles preemption/retry natively; checkpoint logic becomes unnecessary |
| `Config` dataclass | Stage constructor params + YAML config | Curator uses stage-level configuration |
| `run_shard()` orchestration | `Pipeline.run()` + `XennaExecutor` | Curator handles all orchestration |

---

## 2. Is It Feasible?

**Yes.** The mapping above shows near-1:1 structural correspondence. Specifically:

1. **The pipeline is downloading, not crawling.** It consumes a pre-built manifest of known URLs — it does not discover new URLs, follow links, or parse HTML for further navigation. This is the same pattern as Curator's Common Crawl and ArXiv downloaders, which consume pre-built URL lists.

2. **No spider/crawler logic.** The compliance layer (robots.txt + TDMRep) is a *filter* applied before download, not a crawl-control mechanism. It can be modeled as a `ProcessingStage` that filters a `FileGroupTask` or `DocumentBatch` based on compliance checks — exactly how Curator handles filtering in other modalities.

3. **The task decomposition is natural:**
   - Stage 1: Read manifest → produce `FileGroupTask`s with URL batches
   - Stage 2: Compliance filter → remove non-compliant URLs
   - Stage 3: Download images → write bytes to storage
   - Stage 4 (optional): Post-download verification/metadata writing

4. **Existing infrastructure covers ~70% of the need.** Curator already has `URLGenerator`, `DocumentDownloader`, `DocumentDownloadStage`, `FileGroupTask`, `ImageBatch`, `ImageWriterStage`, and the full pipeline/executor machinery. The new pieces are the compliance filter stage and an image-specific downloader.

---

## 3. Benefit to Curator (Maintainability & Long-Term Utility)

### 3.1 Fills a Gap in the Image Modality

Curator's image pipeline today starts *after* images exist on disk: `ImageReaderStage → Embedder → Filters → Writer`. There is no image acquisition stage. The text modality has full download support (Common Crawl, Wikipedia, ArXiv), but images have none. Adding image download closes this gap and makes Curator a complete image curation solution.

### 3.2 Reusable Compliance Infrastructure

The compliance layer (robots.txt + TDMRep checking) is not image-specific. Once built as Curator stages, it can be reused by:
- **PDF download** pipelines (the Nemotron Parse pipeline reads PDFs from URLs)
- **Video download** pipelines (future)
- **Any URL-based data acquisition** in any modality

This is a high-value shared component that benefits the entire Curator ecosystem.

### 3.3 Establishes the "Asset Download" Pattern

Today, Curator's download pattern is text-specific (`DocumentDownloader` → `DocumentIterator` → `DocumentExtractor`). The image pipeline generalizes this to a simpler "download binary asset from URL" pattern that:
- Does not require iteration (one URL = one file, not one archive = many records)
- Does not require extraction (download bytes directly, not HTML → text)
- Applies to images, PDFs, audio files, video files

This creates a reusable `AssetDownloader` abstraction that can serve multiple modalities.

### 3.4 More Users & Contributors

Integrating the cc-img-dl team's pipeline brings their testing, bug reports, and operational experience into Curator. This increases the project's bus factor and ensures the download infrastructure is exercised at scale.

### 3.5 Consistent Tooling and Standards

Moving the code into Curator means it automatically gets:
- Ruff linting/formatting (already close — cc-img-dl uses Ruff too)
- Pytest coverage requirements (80% minimum)
- Type annotations
- NVIDIA copyright headers
- CI/CD pipeline
- Documentation generation

---

## 4. Benefit to the cc-img-dl Team (Network-Bound Pipeline)

### 4.1 The Pipeline Is I/O-Bound — Curator Gives Free Parallelism

The cc-img-dl pipeline is **overwhelmingly network-bound**. The CPU does almost nothing except:
- Parse a Parquet manifest (trivial, one-time)
- Make HTTP requests (I/O wait)
- Write bytes to disk (I/O wait)
- Check compliance caches (memory lookup)

Current parallelism is `asyncio.Semaphore(20)` within a single Python process. To scale, the team's plan involves manual sharding + AWS Batch array jobs — essentially reinventing distributed task scheduling.

**Curator + Xenna provides this for free:**
- Automatic multi-node distribution via Ray
- Per-stage resource allocation (0.5 CPU per download worker → pack many onto one node)
- Autoscaling based on workload
- Streaming execution mode (process tasks as they arrive, don't wait for all shards)
- Built-in metrics and monitoring

The team would go from "write sharding logic + AWS Batch job arrays + Step Functions orchestration" to `pipeline.run()`.

### 4.2 Fault Tolerance Without Custom Checkpointing

cc-img-dl implements its own checkpoint system (`Checkpoint` class) that tracks `last_url_id` in a JSON file. This is fragile:
- Linear scan resume (must re-iterate to the checkpoint URL)
- No handling of partially-downloaded batches
- Single-shard granularity

Xenna provides task-level fault tolerance natively:
- Tasks that fail or get preempted are automatically retried
- No checkpoint files needed
- Each task is independently retryable (no ordering dependency)

The team can delete their entire `checkpoint.py` module.

### 4.3 No Manual Sharding

The `write_shards()` + `shard_id_for()` logic is entirely replaced by Curator's task distribution. The `URLGenerationStage` produces one `FileGroupTask` per URL batch, and the executor distributes them across workers automatically. Deterministic sharding, shard file I/O, and shard-level processing loops all go away.

### 4.4 Storage Abstraction Already Exists

Curator's writer stages already support local and remote storage (via fsspec). The team's `StorageBackend` protocol and `LocalStorage` class are replaced by Curator's existing infrastructure.

### 4.5 Observability

Curator provides per-stage timing metrics, task throughput counters, and integration with Ray's dashboard. The team currently uses `print()` statements for progress tracking. They would get structured logging (loguru), per-stage perf stats, and real-time monitoring.

---

## 5. What Curator Should NOT Take On

### 5.1 Crawling / Spider Logic

The cc-img-dl pipeline is explicitly **not** a web crawler. It does not:
- Discover new URLs by following links
- Parse HTML for embedded resources
- Build or maintain a URL frontier
- Implement politeness delays per-domain (beyond compliance)

Curator should remain a data **curation** framework, not a web scraping framework. The image download pipeline fits because it operates on a pre-built manifest — the same pattern as downloading Common Crawl WARCs from a URL list.

### 5.2 Compliance as a Core Curator Feature

The compliance layer (robots.txt + TDMRep) should be modeled as **optional, user-provided filter stages**, not baked into Curator's core download infrastructure. Reasons:
- Different teams have different compliance requirements
- Compliance policies vary by jurisdiction and use case
- Curator's existing downloaders (CC, Wikipedia, ArXiv) don't include compliance checking

The compliance stages should live under `stages/image/download/compliance/` as opt-in components that users can add to their pipeline.

### 5.3 AWS-Specific Infrastructure

The cc-img-dl team's AWS scale plan (Batch, ElastiCache, DynamoDB, NAT Gateway) is deployment-specific. Curator provides the pipeline definition and execution engine; deployment infrastructure lives outside the framework.

---

## 6. Cons and Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Async incompatibility.** cc-img-dl uses `asyncio` + `httpx` for concurrent downloads. Curator stages are synchronous (`process()` returns, no `await`). | Medium | Use `concurrent.futures.ThreadPoolExecutor` inside `process()` (same pattern as `CommonCrawlWARCReader`). HTTP downloads are I/O-bound, so threads work well. Alternatively, use `asyncio.run()` inside `process()`. |
| **Compliance cache sharing.** In cc-img-dl, the compliance cache is shared across all URLs in a shard (single `InMemoryCache` instance). In Curator, each worker is independent. | Low | Use `setup()` to create per-worker caches. Since compliance is per-origin and cached for 24h, most cache hits will still occur within a single worker processing URLs from the same origins. For cross-worker sharing, a Redis/DynamoDB cache can be injected. |
| **Dependency additions.** `httpx` is not currently a Curator dependency. | Low | Can use `requests` (already a Curator dependency via Common Crawl) or add `httpx` as an optional dependency under an `image_download` extra. |
| **Testing external HTTP services.** Compliance and download stages make real HTTP requests. | Low | cc-img-dl already has good test coverage with `pytest-httpx` mocks. Same pattern works in Curator. |
| **Maintenance burden.** New stages need ongoing maintenance. | Low | The stages are simple (filter + HTTP download). The compliance logic is self-contained and rarely changes. The download logic mirrors existing Common Crawl patterns. |
| **Scope creep.** Future requests to add crawling, link following, or URL discovery. | Medium | Clear documentation that Curator supports *manifest-driven download*, not crawling. The compliance stages are opt-in filters, not crawl controllers. |

---

## 7. Proposed Curator Directory Structure

```
nemo_curator/stages/image/
├── download/
│   ├── __init__.py
│   ├── manifest.py            # ManifestURLGenerator (reads Parquet, yields URL batches)
│   ├── downloader.py          # ImageDownloader (HTTP download with retries)
│   ├── stage.py               # ImageDownloadStage (CompositeStage combining the above)
│   └── compliance/
│       ├── __init__.py
│       ├── robots.py          # RobotsComplianceFilter (ProcessingStage)
│       ├── tdm.py             # TDMComplianceFilter (ProcessingStage)
│       └── checker.py         # ComplianceFilterStage (combines robots + TDM)
```

### Pipeline Usage (User-Facing API)

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.download import ImageDownloadStage

pipeline = Pipeline(
    name="image_download",
    stages=[
        ImageDownloadStage(
            manifest_path="url_manifest.parquet",
            output_dir="/data/images",
            enable_compliance=True,
            concurrency=50,
        ),
    ],
)
pipeline.run()
```

Or, for fine-grained control:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.download.manifest import ManifestURLGenerator, ManifestReaderStage
from nemo_curator.stages.image.download.compliance import ComplianceFilterStage
from nemo_curator.stages.image.download.downloader import ImageDownloaderStage

pipeline = Pipeline(
    name="image_download",
    stages=[
        ManifestReaderStage(manifest_path="url_manifest.parquet", urls_per_task=1000),
        ComplianceFilterStage(user_agent="my-bot/1.0", policy="conservative"),
        ImageDownloaderStage(output_dir="/data/images", max_retries=3),
    ],
)
pipeline.run()
```

---

## 8. Effort Estimate

| Component | Effort | Notes |
|---|---|---|
| `ManifestReaderStage` | Small | Thin wrapper around existing Parquet reading + `URLGenerator` pattern |
| `ImageDownloaderStage` | Small | Analogous to `DocumentDownloadStage`; rewrite `download_image()` as sync with thread pool |
| `ComplianceFilterStage` | Medium | New stage type; port compliance logic from cc-img-dl with per-worker caching |
| Tests | Medium | Port existing cc-img-dl tests + add Curator-specific integration tests |
| Tutorial | Small | Example script showing end-to-end usage |
| Documentation | Small | Add to image modality docs |
| **Total** | **~1–2 weeks** | Assuming one developer familiar with both codebases |

---

## 9. Recommendation

**Move it.** The alignment is strong, the effort is modest, and both sides benefit:

- **Curator** gets image acquisition, reusable compliance infrastructure, and a generalized asset download pattern that serves multiple modalities.
- **The cc-img-dl team** gets distributed execution, fault tolerance, autoscaling, observability, and elimination of ~40% of their code (sharding, checkpointing, orchestration, storage abstraction) that Curator already handles.

The key constraint — that Curator should not become a web crawler — is naturally satisfied because the pipeline operates on pre-built manifests, not live web discovery.
