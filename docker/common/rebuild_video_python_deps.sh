#!/bin/bash
set -xeuo pipefail

export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
if [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
    CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
    if [ "${CMAKE_BUILD_PARALLEL_LEVEL}" -gt 8 ]; then
        CMAKE_BUILD_PARALLEL_LEVEL=8
    fi
fi
export CMAKE_BUILD_PARALLEL_LEVEL

ffmpeg -version | head -n 1 | grep "ffmpeg version 8.1.2"

AV_VERSION="$(python3 -c 'import importlib.metadata as m; print(m.version("av"))')"
OPENCV_VERSION="$(python3 -c 'import importlib.metadata as m; print(m.version("opencv-python-headless"))')"
OPENCV_PYTHON_TAG="${OPENCV_PYTHON_TAG:-92}"

# Diagnostic build: keep PyAV's PyPI wheel so it uses its bundled FFmpeg.
uv pip install --reinstall --only-binary av "av==${AV_VERSION}"

# Avoid cv2 namespace conflicts and rebuild headless OpenCV without FFmpeg support.
uv pip uninstall opencv-python || true
ENABLE_HEADLESS=1 \
    CMAKE_ARGS="-DWITH_FFMPEG=OFF ${CMAKE_ARGS:-}" \
    uv pip install --reinstall \
        "opencv-python-headless @ git+https://github.com/opencv/opencv-python.git@${OPENCV_PYTHON_TAG}"

python3 - <<'PY'
import importlib.metadata as metadata
from pathlib import Path

import av
import cv2

site_packages = Path("/opt/venv/lib/python3.13/site-packages")

print(f"av {metadata.version('av')} linked libraries: {av.library_versions}")
print(f"opencv-python-headless {metadata.version('opencv-python-headless')}")

if not (site_packages / "av.libs").exists():
    raise SystemExit("av.libs is missing; PyAV is not using the vendored FFmpeg wheel")

opencv_libs = site_packages / "opencv_python_headless.libs"
vendored_ffmpeg = []
if opencv_libs.exists():
    vendored_ffmpeg = [
        path
        for path in opencv_libs.iterdir()
        if path.name.startswith(("libav", "libswresample", "libswscale"))
    ]
if vendored_ffmpeg:
    raise SystemExit(f"OpenCV still vendors FFmpeg libraries: {vendored_ffmpeg}")

video_io_block = cv2.getBuildInformation().split("Video I/O:", 1)[1].split("\n\n", 1)[0]
print(video_io_block)
ffmpeg_lines = [line for line in video_io_block.splitlines() if line.strip().startswith("FFMPEG:")]
if ffmpeg_lines and not any(line.split(":", 1)[1].strip() == "NO" for line in ffmpeg_lines):
    raise SystemExit("OpenCV was not built with FFMPEG disabled")
if not ffmpeg_lines:
    print("FFMPEG: not listed in OpenCV Video I/O block; treating as disabled")
PY
