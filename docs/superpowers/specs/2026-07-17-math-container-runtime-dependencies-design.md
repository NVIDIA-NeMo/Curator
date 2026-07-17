# Math Container Runtime Dependencies Design

## Problem

Removing the system FFmpeg installation also removed `libtool`, whose Ubuntu
dependency chain had incidentally installed `libmagic`. The Python package
`python-magic` remained installed transitively through `comment-parser`, so
CPU tests passed on a GitHub-hosted Ubuntu runner that already provided the
native library. The published container did not provide `libmagic`, causing
math extraction imports to fail at runtime.

## Design

- Install Ubuntu 24.04's `libmagic1t64` package explicitly in
  `docker/Dockerfile`. It provides `libmagic1` and depends on the matching
  `libmagic-mgc` database.
- Declare `python-magic==0.4.24` directly in the `math_cpu` extra instead of
  relying on `comment-parser` to install it transitively.
- Add a no-mock extraction test to the existing source-mirrored
  `test_math_content_extractor.py` and run that exact test against the image
  produced by the PR container-build job.
- Add a `math` GPU test group using `math_cuda12`. Add live GPU integration
  coverage to the existing source-mirrored FineMath classifier and LLM cleanup
  test files, following the repository's established direct stage-usage
  pattern rather than constructing a pipeline inside the unit tests.

## CI Behavior

The container build will fail immediately if either the Python wrapper, native
shared library, or compiled magic database is absent. The math GPU job will
validate that `math_cuda12` resolves in the built image and that the FineMath
and vLLM cleanup stages load real models and execute on CUDA without mocks.

## Compatibility

The changes target `main` and use files shared with `r1.3.0` so the resulting
commits can be cherry-picked. No public Python APIs change, and no `AGENTS.md`
files are added or modified.
