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

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.inference.indic_conformer_tensorrt import load_engine_metadata


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": "indic_conformer_hybrid",
        "precision": "fp16",
        "engine_file": "encoder.plan",
        "model_file": "model.nemo",
        "sample_rate": 16_000,
        "feature_count": 80,
        "subsampling_factor": 4,
        "encoder_dim": 512,
        "input_names": ["audio_signal", "length"],
        "output_names": ["outputs", "encoded_lengths"],
        "profile": {
            "min": {"batch": 1, "feature_frames": 8},
            "opt": {"batch": 8, "feature_frames": 800},
            "max": {"batch": 16, "feature_frames": 4001},
        },
    }


def _write_metadata(tmp_path: Path, metadata: object) -> Path:
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_load_engine_metadata_accepts_indic_conformer_bundle(tmp_path: Path) -> None:
    metadata = load_engine_metadata(_write_metadata(tmp_path, _metadata()))

    assert metadata["encoder_dim"] == 512
    assert metadata["profile"]["max"]["feature_frames"] == 4001


def test_load_engine_metadata_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"metadata\.json"):
        load_engine_metadata(tmp_path)


def test_load_engine_metadata_rejects_wrong_model_type(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["model_type"] = "indic_parakeet_rnnt"

    with pytest.raises(ValueError, match="model type"):
        load_engine_metadata(_write_metadata(tmp_path, metadata))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("precision", "fp32", "precision"),
        ("input_names", ["audio_signal"], "input names"),
        ("encoder_dim", 0, "encoder_dim"),
        ("profile", {"min": None, "opt": {}, "max": {}}, "profile points"),
        (
            "profile",
            {
                "min": {"batch": 8, "feature_frames": 8},
                "opt": {"batch": 4, "feature_frames": 800},
                "max": {"batch": 16, "feature_frames": 4001},
            },
            "profile batch",
        ),
    ],
)
def test_load_engine_metadata_rejects_invalid_contract_fields(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    metadata = _metadata()
    metadata[key] = value

    with pytest.raises((TypeError, ValueError), match=message):
        load_engine_metadata(_write_metadata(tmp_path, metadata))


def test_load_engine_metadata_rejects_non_object_manifest(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a JSON object"):
        load_engine_metadata(_write_metadata(tmp_path, []))
