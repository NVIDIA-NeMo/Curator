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

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarking.data_prep import prepare_audio_sortformer_data as prep


def _valid_rows() -> list[dict]:
    return [
        {
            "audio_filepath": f"audio/{filename}",
            "audio_item_id": filename.removesuffix(".wav"),
            "duration": 2000.0,
        }
        for filename in prep.EXPECTED_AUDIO_FILENAMES
    ]


def test_public_workload_contains_34_unique_meetings() -> None:
    assert prep.AMI_HF_REPO_ID == "diarizers-community/ami"
    assert prep.AMI_SPLIT_NUM_ROWS == {"validation": 18, "test": 16}
    assert len(prep.EXPECTED_AUDIO_FILENAMES) == 34
    assert len(set(prep.EXPECTED_AUDIO_FILENAMES)) == 34


def test_manifest_contract_accepts_complete_unique_workload() -> None:
    prep._validate_manifest_contract(_valid_rows(), "test manifest")


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("audio_filepath", "audio/unexpected.wav", "unexpected audio file"),
        ("audio_item_id", "ami_sdm_validation_001", "duplicate audio_item_id"),
        ("duration", 0, "finite positive duration"),
    ],
)
def test_manifest_contract_rejects_invalid_rows(field: str, replacement: object, match: str) -> None:
    rows = deepcopy(_valid_rows())
    rows[0][field] = replacement
    with pytest.raises(RuntimeError, match=match):
        prep._validate_manifest_contract(rows, "test manifest")


def test_manifest_contract_rejects_truncated_workload() -> None:
    with pytest.raises(RuntimeError, match="exactly 34 rows"):
        prep._validate_manifest_contract(_valid_rows()[:-1], "test manifest")


def test_verify_model_rejects_empty_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / prep.MODEL_FILENAME
    model_path.touch()

    assert prep.verify_model(model_path) is False
