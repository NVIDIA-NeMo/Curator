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
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Override the session-level autouse Ray cluster fixture from the root conftest.
# Integration tests call stages directly — no Ray pipeline needed.
# The local fixture takes precedence over the parent conftest for all tests
# in this directory.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:  # type: ignore[override]
    """No-op override: caption integration tests run stages directly, not via Ray."""
    return


# Small video committed to the repo so tests are self-contained and portable.
# 3s, 240x136, 2fps, mpeg4 — 3.8 KB
_DEFAULT_VIDEO_FIXTURE = Path(__file__).parent / "fixtures" / "test_video.mp4"


@pytest.fixture(scope="session", autouse=True)
def pipeline_tmpdir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Point TMPDIR at a short writable path so vLLM's ZMQ IPC socket stays
    under the 107-character Unix limit.
    """
    tmp = tmp_path_factory.mktemp("pipeline")
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


@pytest.fixture(scope="session")
def video_fixture_path() -> Path:
    """Return the path to the small video fixture used for integration tests."""
    path = _DEFAULT_VIDEO_FIXTURE
    if not path.exists():
        pytest.fail(f"Test video fixture missing from repo: {path}")
    return path


@pytest.fixture(scope="module")
def enhancement_stage():
    """Instantiate and set up CaptionEnhancementStage once per module.

    setup() loads vLLM's LLM() with Qwen2.5-14B-Instruct via HF auto-download.
    """
    # Set env vars here (not module-level) so they don't affect other GPU tests
    # collected in the same pytest session (e.g. tests/core/test_serve.py).
    # vLLM 0.14+ spawns its EngineCore via fork() by default; forcing spawn
    # avoids deadlocks when HF AutoProcessor has already created threads.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _custom_hf = os.environ.get("CUSTOM_HF_DATASET", "")
    if _custom_hf:
        os.environ.setdefault("HF_HOME", _custom_hf)

    from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage

    model_dir = os.environ.get("CURATOR_TEST_MODEL_DIR", "")
    stage = CaptionEnhancementStage(
        model_dir=model_dir,
        model_variant="qwen",
        model_batch_size=1,
        fp8=False,
        max_output_tokens=128,  # short output keeps the test fast
        vllm_kwargs={"enforce_eager": True},  # skip CUDA graph capture
        verbose=False,
    )
    stage.setup()
    yield stage
    # Release GPU memory after module tests so subsequent tests can load models
    import gc

    import torch

    del stage
    gc.collect()
    torch.cuda.empty_cache()
