# Vendored Tutorial Files

This directory is a build-time destination for tutorial files used by selected
benchmark scripts.

It is intentionally mostly empty in the source tree. During package builds,
`benchmarking/setup.py` copies a small allowlist of required files from the peer
`tutorials/` directory into this package. That lets non-editable
`nemo-curator-benchmarking` installs run tutorial-backed benchmarks without
requiring a full Curator source checkout at runtime.

Do not manually copy tutorial files here. Add required tutorial-backed benchmark
files to `VENDORED_TUTORIAL_FILES` in `benchmarking/setup.py` instead.
