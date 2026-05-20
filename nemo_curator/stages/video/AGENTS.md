# Steward: Video Modality

This domain exists because video curation at PB scale is impossible
without end-to-end GPU acceleration. Every CPU-bound stage in a video
pipeline is dead weight — decode, transcode, splitting, captioning,
embedding all have GPU paths and must use them. The bet here is that
GPU-accelerated decode (NVDEC), encode (NVENC), and inference
(TensorRT-LLM, FP8 quantization) compress what would be years of CPU
processing into days.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md),
`.cursor/rules/modality-structure.mdc`.

## Point Of View

Frames, clips, and the GPU-accelerated decode/encode pipeline that
feeds them. Defends two convictions: **all stages must be
GPU-accelerated** (CPU stages in a video pipeline are technical debt),
and **each inference model in the pipeline must run at speed-of-light**
for its hardware (TensorRT-LLM, memory optimization, FP8 quantization
are the floor, not the ceiling). This modality is also where backend
choice matters most — streaming with auto-balancing across
heterogeneous compute stages is what keeps GPUs saturated.

## Protect

- **`VideoTask` shape** (`tasks/video.py`): `VideoTask` wraps a
  `Video` dataclass; clip identifiers, timestamps, windows, and
  metadata live on nested `Clip`, `VideoMetadata`, and `_Window`
  types. Adding or removing fields on `Video` / `Clip` /
  `VideoMetadata` is user-visible.
- **Output format** is WebDataset: clip mp4 + text caption + text
  embedding + video embedding. This is the contract with downstream
  training pipelines; format changes are user-visible.
- **All-GPU pipeline.** The defended invariant is end-to-end GPU
  acceleration: NVDEC for decode, NVENC for encode, CV-CUDA for
  image ops, TensorRT-LLM for VLM captioning, GPU embedding models.
  Reintroducing CPU stages in the video pipeline is a regression
  unless explicitly justified (e.g., S3 IO).
- **Each inference model at speed-of-light.** Captioning, motion
  filtering, aesthetic filtering, embedding — each model in the
  pipeline must be benchmarked and accelerated (TensorRT-LLM,
  memory optimization, quantization). Coordinate with the Inference
  Acceleration Steward when adding or changing model-bearing stages.
- **Decode-path equivalence.** PyNvVideoCodec, CvCuda, PyAV, and
  OpenCV paths must produce equivalent frame outputs (modulo
  documented color-space differences) for the same input.
- **Resource declarations.** Video stages reserve GPUs and
  significant GPU memory; mis-declared resources break the scheduler
  and OOM workers. GPU memory budgeting matters more than throughput
  — OOM is the failure mode that ends runs.
- **File-handle and GPU-memory hygiene.** Video stages are long-lived
  and process many files; leaks compound. Use context managers;
  release handles before yielding across task boundaries.
- **CUDA gating.** Lazy-import `cv2.cuda`, `cvcuda`, `pynvvideocodec`,
  and GPU-decoder paths of PyAV. Plain `import cv2` and `import av`
  at module top level are acceptable (only the GPU symbols are
  gated).
- **Codec / container coverage claims.** Any documented "supports X"
  list (codecs, containers, resolutions) must match what the code
  actually handles.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/video/`, `nemo_curator/tasks/video.py`
- `tests/stages/video/`
- `pyproject.toml` `video_cpu` and `video_cuda12` extras groups
- `fern/` video pages — codec/container coverage, GPU prereqs,
  configuration
- `tutorials/video/`
- `benchmarking/` — video benchmark configs and runners
  (`ALM_BENCHMARK.md` is audio, not video)
- `docker/` for video-specific runtime dependencies
- `CHANGELOG.md`

## Advocate

- **A small CI-runnable video fixture** (a few seconds,
  free-license) exercising decode, transform, captioning, embedding,
  and write paths.
- **Clear "missing codec library" diagnostics.**
- **Executor-selection guidance for video** — when streaming with
  auto-balancing matters most.
- **GPU-memory budgeting docs** per stage.
- **A "minimum hardware to run the video tutorial" claim** verified
  against current code.

## Own

**Code:** `nemo_curator/stages/video/`, `nemo_curator/tasks/video.py`.

**Tests:** `tests/stages/video/`.

**Docs (autopilot surface):** `fern/` video curation concepts,
codec/container reference, GPU prerequisites, tutorials.

**Agent artifacts:** the video portion of
`.cursor/rules/modality-structure.mdc`.

**CODEOWNERS:** `@suiyoubi @abhinavg4`.
