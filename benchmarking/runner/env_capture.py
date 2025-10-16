import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger


def capture_environment_artifacts(output_path: Path) -> None:
    Path(output_path).mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.check_output(["pip", "freeze"], text=True, timeout=120)  # noqa: S603, S607
        (output_path / "pip-freeze.txt").write_text(freeze)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to capture pip freeze: {e}")
    try:
        # Try micromamba first, then conda as fallback
        cmd = None
        if shutil.which("micromamba"):
            cmd = ["micromamba", "list", "--explicit"]
        elif shutil.which("conda"):
            cmd = ["conda", "list", "--explicit"]

        if cmd:
            exp = subprocess.check_output(cmd, text=True, timeout=120)  # noqa: S603
            (output_path / "conda-explicit.txt").write_text(exp)
        else:
            logger.warning("Neither micromamba nor conda found in PATH, skipping conda-explicit.txt")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to capture conda list: {e}")

    sysenv = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": os.getenv("_"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }
    (output_path / "sys-env.json").write_text(json.dumps(sysenv))


def collect_basic_env() -> dict[str, Any]:
    try:
        ray_ver = subprocess.check_output(["python", "-c", "import ray,sys;print(ray.__version__)"], text=True).strip()  # noqa: S603, S607
    except Exception:  # noqa: BLE001
        ray_ver = "unknown"
    return {
        "hostname": platform.node(),
        "ray_version": ray_ver,
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "image_digest": os.getenv("IMAGE_DIGEST", "unknown"),
    }
