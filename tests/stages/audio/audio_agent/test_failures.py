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

"""Specific failure evidence must survive executor and worker wrappers."""

from __future__ import annotations

import pytest

from nemo_curator.stages.audio.audio_agent.failures import classify


@pytest.mark.parametrize(
    ("error", "code"),
    [
        ("FileNotFoundError: [Errno 2] No such file: input.jsonl", "bad_path"),
        ("HTTPError: 401 Unauthorized for gated model", "model_auth_or_download"),
        ("RuntimeError: CUDA out of memory in cuda_graph capture", "out_of_memory"),
        ("OSError: [Errno 28] No space left on device", "disk_full"),
        ("PermissionError: [Errno 13] Permission denied", "permission_denied"),
        ("SSLError: certificate verify failed", "tls_certificate"),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    ["{}", "ray.exceptions.RayTaskError: {}", "Node setup failed for stage ASR:\n{}", "Error opening audio: {}"],
)
def test_specific_cause_takes_priority_over_wrapper(error: str, code: str, wrapper: str) -> None:
    assert classify(wrapper.format(error))["code"] == code


@pytest.mark.parametrize(
    "error",
    [
        "Node setup failed for stage VAD:\nModuleNotFoundError: No module named 'silero_vad'",
        "ModuleNotFoundError: No module named 'soundfile'\nTraceback in working_dir_files/_ray_pkg_123",
    ],
)
def test_worker_import_evidence_is_recognized_in_either_order(error: str) -> None:
    assert classify(error)["code"] == "worker_env_mismatch"


def test_worker_context_alone_does_not_diagnose_an_environment_mismatch() -> None:
    assert classify("Node setup failed for stage ASR: unexplained failure")["code"] == "unknown_failure"


def test_a_bare_import_error_does_not_establish_a_worker_mismatch() -> None:
    assert classify("ModuleNotFoundError: No module named 'soundfile'")["code"] == "missing_dependency"


@pytest.mark.parametrize("error", ["CUDA_ERROR_UNSUPPORTED_PTX_VERSION", "cuda.graph capture_end failed"])
def test_kernel_or_capture_failure_is_recognized_without_claiming_oom(error: str) -> None:
    assert classify(error)["code"] == "asr_decoder_cuda_graph"


@pytest.mark.parametrize("error", ["an array has an invalid shape", "100 rows retained", "a tray was moved"])
def test_generic_signatures_do_not_match_substrings(error: str) -> None:
    assert classify(error)["code"] == "unknown_failure"
