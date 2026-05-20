# Steward: Video Modality

Video curation with GPU-accelerated decode via OpenCV, PyAV, CvCuda,
and PyNvVideoCodec. The highest-resource and most hardware-sensitive
modality.

Related: root [AGENTS.md](../../../AGENTS.md), parent
[nemo_curator/AGENTS.md](../../AGENTS.md),
`.cursor/rules/modality-structure.mdc`.

## Point Of View

Frames, clips, and the decode/encode pipeline that feeds them. The
modality where a wrong codec path or a leaked file handle costs
node-hours and where backend choice (Xenna autoscaling vs Ray Data
streaming) is most consequential.

## Protect

- **`VideoTask` shape** (`tasks/video.py`): `VideoTask` wraps a
  `Video` dataclass; clip identifiers, timestamps, windows, and
  metadata live on nested `Clip`, `VideoMetadata`, and `_Window`
  types — not directly on `VideoTask`. Adding/removing fields on
  `Video` / `Clip` / `VideoMetadata` is user-visible.
- **Decode-path equivalence.** PyNvVideoCodec, CvCuda, PyAV, and
  OpenCV paths must produce equivalent frame outputs (modulo
  documented color-space differences) for the same input.
- **Resource declarations.** Video stages reserve GPUs and
  significant GPU memory; mis-declared resources break the scheduler
  and OOM workers.
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

- A small CI-runnable video fixture (a few seconds, free-license)
  exercising decode, transform, and write paths.
- Clear "missing codec library" diagnostics.
- Executor-selection guidance for video (Xenna for autoscaling, Ray
  Data for streaming).
- GPU-memory budgeting docs per stage.
- A "minimum hardware to run the video tutorial" claim verified
  against current code.

## Own

**Code:** `nemo_curator/stages/video/`, `nemo_curator/tasks/video.py`.

**Tests:** `tests/stages/video/`.

**Docs (autopilot surface):** `fern/` video curation concepts,
codec/container reference, GPU prerequisites, tutorials.

**Agent artifacts:** the video portion of
`.cursor/rules/modality-structure.mdc`.

**CODEOWNERS:** `@suiyoubi @abhinavg4`.
