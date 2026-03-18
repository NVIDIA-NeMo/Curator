# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixtures for video caption integration tests."""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# vLLM 0.14+ spawns its EngineCore via fork() by default.  If any prior code
# (e.g. HF AutoProcessor) has created threads, the forked child inherits
# their lock state and deadlocks.  Forcing spawn avoids this entirely.
# TOKENIZERS_PARALLELISM=false prevents HF fast-tokenizer threads from being
# created in the first place (belt-and-suspenders).
# Both must be set before any import that might start threads.
# ---------------------------------------------------------------------------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Override the session-level autouse Ray cluster fixture from the root conftest.
# Integration tests call stages directly — no Ray pipeline needed.
# The local fixture takes precedence over the parent conftest for all tests
# in this directory.
# ---------------------------------------------------------------------------

# Repo root: tests/stages/video/caption/conftest.py is 4 levels deep.
_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:  # type: ignore[override]
    """No-op override: caption integration tests run stages directly, not via Ray."""
    return


# ---------------------------------------------------------------------------
# CLI option: --model-dir
# Priority: CLI arg > env var QWEN_MODEL_DIR > skip
# ---------------------------------------------------------------------------

_ENV_VAR = "QWEN_MODEL_DIR"

# Small video committed to the repo so tests are self-contained and portable.
# 3s, 240x136, 2fps, mpeg4 — 3.8 KB
_DEFAULT_VIDEO_FIXTURE = Path(__file__).parent / "fixtures" / "test_video.mp4"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model-dir",
        action="store",
        default=None,
        help=(
            "Root directory that contains model weights "
            "(e.g. /path/to/models containing Qwen/Qwen2.5-VL-7B-Instruct/). "
            f"Falls back to ${_ENV_VAR}. Required for integration tests."
        ),
    )


@pytest.fixture(scope="session")
def qwen_model_dir(request: pytest.FixtureRequest) -> str:
    """Resolve the model-weights root directory using precedence:

    1. --model-dir CLI argument
    2. QWEN_MODEL_DIR environment variable
    3. Skip — no personal path fallback so the test is portable
    """
    cli_value: str | None = request.config.getoption("--model-dir", default=None)
    if cli_value:
        return cli_value

    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        return env_value

    pytest.skip(f"Model directory not specified. Pass --model-dir /path/to/models or set ${_ENV_VAR}.")


@pytest.fixture(scope="session", autouse=True)
def pipeline_tmpdir() -> Generator[Path, None, None]:
    """Point make_pipeline_named_temporary_file at a short writable tmpdir.

    Uses .tmp/ inside the repo root, short enough to keep vLLM's ZMQ IPC
    socket path under the 107-character.
    Unix limit (vLLM appends a UUID to TMPDIR for its socket file).
    """
    tmp = _REPO_ROOT / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    old = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(tmp)
    # tempfile caches gettempdir(); clear the cache so the new value is picked up
    tempfile.tempdir = str(tmp)
    yield tmp
    tempfile.tempdir = None
    if old is None:
        del os.environ["TMPDIR"]
    else:
        os.environ["TMPDIR"] = old
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def video_fixture_path() -> Path:
    """Return the path to the small video fixture used for integration tests."""
    path = _DEFAULT_VIDEO_FIXTURE
    if not path.exists():
        pytest.fail(f"Test video fixture missing from repo: {path}")
    return path


@pytest.fixture(scope="module")
def enhancement_stage(qwen_model_dir: str):
    """Instantiate and set up CaptionEnhancementStage once per module.

    setup() loads vLLM's LLM() with Qwen2.5-14B-Instruct — the text-only
    caption enhancement model.
    """
    from nemo_curator.models.qwen_lm import _QWEN_LM_MODEL_ID
    from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage

    weight_path = Path(qwen_model_dir) / _QWEN_LM_MODEL_ID
    if not weight_path.exists():
        pytest.skip(
            f"Qwen14B weights not found at {weight_path}. "
            f"Pass --model-dir or set ${_ENV_VAR} to a directory "
            f"containing Qwen/Qwen2.5-14B-Instruct/."
        )

    stage = CaptionEnhancementStage(
        model_dir=qwen_model_dir,
        model_variant="qwen",
        model_batch_size=1,
        fp8=False,
        max_output_tokens=128,  # short output keeps the test fast
        enforce_eager=True,  # skip CUDA graph capture
        verbose=False,
    )
    stage.setup()
    return stage
