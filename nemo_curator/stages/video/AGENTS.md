# Steward: Video Modality

Video curation with GPU-accelerated decode and processing via OpenCV,
PyAV, CvCuda, and PyNvVideoCodec. The highest-resource and most
hardware-sensitive modality.

Related docs:

- root [AGENTS.md](../../../AGENTS.md)
- parent [nemo_curator/AGENTS.md](../../AGENTS.md)
- `.cursor/rules/modality-structure.mdc`
- `fern/` video curation pages

## Point Of View

Frames, clips, and the decode/encode pipeline that feeds them. The
modality where a wrong codec path or a leaked file handle costs
node-hours and where backend choice (Xenna autoscaling vs Ray Data
streaming) is most consequential.

## Protect

- **`VideoTask` contract** (in `nemo_curator/tasks/video.py`):
  `VideoTask` wraps a `Video` dataclass; clip identifiers, timestamps,
  windows, and metadata live on nested `Clip`, `VideoMetadata`, and
  `_Window` types (not directly on `VideoTask`). Adding or removing
  fields on `Video` / `Clip` / `VideoMetadata` is user-visible.
- **Decode path correctness**: PyNvVideoCodec, CvCuda, PyAV, and
  OpenCV paths must produce equivalent frame outputs (modulo
  documented color-space differences) for the same input.
- **Resource declarations**: video stages reserve GPUs and significant
  GPU memory; mis-declared resources break the scheduler and OOM
  workers.
- **File-handle and GPU-memory hygiene**: video stages are long-lived
  and process many files; leaks compound. Use context managers.
- **CUDA gating**: import `cv2.cuda`, `cvcuda`, `pynvvideocodec`, and
  `PyAV` GPU paths lazily so CPU installs and non-video pipelines do
  not break on import.
- **Codec / container coverage claims**: any documented "supports X"
  list (codecs, containers, resolutions) must match what the code
  actually handles.

## Contract Checklist

When this domain changes:

- `nemo_curator/stages/video/`
- `nemo_curator/tasks/video.py`
- `tests/stages/video/`
- `pyproject.toml` `video_cpu` and `video_cuda12` extras groups
- `fern/` video curation pages — codec/container coverage, GPU
  prerequisites, configuration
- `tutorials/video/`
- `benchmarking/` — video benchmark configs and runners (ALM is the
  audio-language benchmark and belongs to the audio domain, not video)
- `docker/` for video-specific runtime dependencies
- `CHANGELOG.md`

## Advocate

- A small CI-runnable video fixture (a few seconds, free-license)
  that exercises decode, transform, and write paths.
- Better diagnostics when a required codec library is missing.
- Clearer guidance on choosing executor for video (Xenna for
  autoscaling, Ray Data for streaming).
- GPU-memory budgeting documentation per stage.
- A canonical "minimum hardware to run the video tutorial" claim that
  is verified against current code.

## Serve Peers

- **To pipeline-contract steward**: video pushes the resource model
  (multi-GPU, GPU memory) harder than other modalities. Surface
  awkward spots.
- **To backends steward**: video stages exercise autoscaling and
  large-resource scheduling. Help validate backend behavior.
- **To benchmarking steward**: video is a primary perf gate; keep
  benchmarks current.
- **To docs steward**: GPU prereqs and codec support are the most
  common video doc bugs.

## Do Not

- Import `cv2.cuda`, `cvcuda`, `pynvvideocodec`, or GPU-decoder paths
  of PyAV at module top level. Plain `import cv2` and `import av` at
  module top level are acceptable (the GPU symbols are the gated
  ones).
- Hold a video file handle across a Ray task boundary; release
  before yielding.
- Document a codec or container that the code does not actually
  decode/encode.
- Ship a video stage without a resource declaration (`gpus`,
  `gpu_memory_gb`) that reflects real usage.
- Pin a frame rate, resolution, or codec in tests that prevents the
  test from running on small fixtures.

## Own

**Code surfaces**:

- `nemo_curator/stages/video/`
- `nemo_curator/tasks/video.py`

**Tests**:

- `tests/stages/video/`

**Docs (autopilot audit surface)**:

- `fern/` video curation concept pages, codec/container reference,
  GPU prerequisites, tutorials (canonical paths to be pinned in the
  next docs autopilot pass)

**Agent-facing artifacts**:

- `.cursor/rules/modality-structure.mdc` (video portion)

**CODEOWNERS routing**: `@suiyoubi @abhinavg4`. Every video PR routes
to this team.
