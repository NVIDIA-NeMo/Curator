"""
InternVideo2 Multi-Modality integration module.

This module provides access to InternVideo2 multi-modality models
through the git submodule located at externals/InternVideo.
"""

import subprocess
import sys
from pathlib import Path


def _ensure_internvideo_installed() -> bool | None:
    """Ensure the InternVideo package is installed."""
    # Path relative to the ray_curator package
    # The path should be: ray-curator/externals/InternVideo/InternVideo2/multi_modality
    # From ray_curator/modules/internvideo/__init__.py, we need to go up 3 levels
    submodule_path = Path(__file__).parent.parent.parent.parent / "externals" / "InternVideo" / "InternVideo2" / "multi_modality"

    if not submodule_path.exists():
        msg = (
            f"InternVideo submodule not found at {submodule_path}. "
            "Please run: git submodule update --init --recursive"
        )
        raise ImportError(
            msg
        )

    # Check if the package is already installed and importable
    try:
        import internvideo2_multi_modality  # noqa: F401
        return True
    except ImportError:
        print(f"InternVideo2 package not found, installing from {submodule_path}")

    # Install the package in regular mode to avoid import context issues
    try:
        print(f"Installing InternVideo2 multi-modality from {submodule_path}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", str(submodule_path)
        ], capture_output=True, text=True, check=True)

        print("InternVideo2 multi-modality Installation completed successfully")

    except subprocess.CalledProcessError as e:
        print(f"InternVideo2 multi-modality Installation failed: {e.stderr}")
        msg = f"Installation failed: {e.stderr}"
        raise ImportError(msg)
    except Exception as e:  # noqa: BLE001
        msg = (
            f"Failed to install InternVideo2 multi-modality from {submodule_path}. "
            f"Error: {e}"
        )
        raise ImportError(
            msg
        )

# Ensure the package is installed
_ensure_internvideo_installed()

# Now import the models from the installed package
from internvideo2_multi_modality import (
    InternVideo2_CLIP,
    InternVideo2_CLIP_small,
    InternVideo2_Stage2_audiovisual,
    InternVideo2_Stage2_visual,
    build_bert,
    interpolate_pos_embed_internvideo2_new,
    pretrain_internvideo2_1b_patch14_224,
    pretrain_internvideo2_6b_patch14_224,
)

__all__ = [
    "InternVideo2_CLIP",
    "InternVideo2_CLIP_small",
    "InternVideo2_Stage2_audiovisual",
    "InternVideo2_Stage2_visual",
    "build_bert",
    "interpolate_pos_embed_internvideo2_new",
    "pretrain_internvideo2_1b_patch14_224",
    "pretrain_internvideo2_6b_patch14_224",
]
