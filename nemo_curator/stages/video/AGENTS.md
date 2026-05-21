# Steward: Video Modality

You own the video modality. All-GPU end-to-end is the bar; every CPU
stage in a video pipeline is technical debt to remove. Each inference
model in the pipeline runs at speed-of-light or it's a regression.

Related: `.cursor/rules/modality-structure.mdc`. Inference-bearing
changes also apply the Inference Acceleration concerns in root
AGENTS.md.

## Point Of View

Frames, clips, and the GPU-accelerated decode/encode pipeline that
feeds them. Defend two convictions: every stage is GPU-accelerated
(NVDEC, NVENC, CV-CUDA, TensorRT-LLM), and each model is benchmarked
and tuned for its hardware (memory optimization, FP8 quantization).
GPU memory budgeting matters more than throughput — OOM ends runs.

## Protect

- **`VideoTask` shape** (`tasks/video.py`): `VideoTask` wraps a
  `Video` dataclass; clip identifiers, timestamps, windows, and
  metadata live on nested `Clip`, `VideoMetadata`, and `_Window`
  types. Field changes on `Video` / `Clip` / `VideoMetadata` are
  user-visible.
- **Output format** is WebDataset: clip mp4 + text caption + text
  embedding + video embedding. Format changes are user-visible.
- **All-GPU pipeline.** End-to-end GPU: NVDEC decode, NVENC encode,
  CV-CUDA image ops, TensorRT-LLM VLM captioning, GPU embedding
  models. Reintroducing CPU stages is a regression unless explicitly
  justified (e.g., S3 IO).
- **Decode-path equivalence.** PyNvVideoCodec, CvCuda, PyAV, and
  OpenCV paths produce equivalent frame outputs (modulo documented
  color-space differences) for the same input.
- **Resource declarations.** Video stages reserve GPUs and
  significant GPU memory; mis-declared resources OOM workers.
- **File-handle and GPU-memory hygiene.** Use context managers;
  release handles before yielding across task boundaries.
- **CUDA gating.** Lazy-import `cv2.cuda`, `cvcuda`, `pynvvideocodec`,
  and GPU-decoder paths of PyAV. Plain `import cv2` and `import av`
  at module top level are acceptable.
- **Codec / container coverage claims** match what the code actually
  handles.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/video/`, `nemo_curator/tasks/video.py`
- `tests/stages/video/`
- `pyproject.toml` `video_cpu` and `video_cuda12` extras groups
- `fern/` video pages — codec/container coverage, GPU prereqs,
  configuration
- `tutorials/video/`
- `benchmarking/` — video benchmark configs and runners. Every new
  video stage ships with a benchmarking script + yaml entry; nightly
  cron will run it on 4×A100.
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

**Docs:** `fern/` video curation concepts, codec/container reference,
GPU prerequisites, tutorials.

**Agent artifacts:** the video portion of
`.cursor/rules/modality-structure.mdc`.

**CODEOWNERS:** `@suiyoubi @abhinavg4`.
