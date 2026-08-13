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

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
from pytest import MonkeyPatch

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("soundfile")

from benchmarking.scripts import audio_sortformer_contract as contract


def _valid_rows() -> list[dict]:
    return [
        {
            "audio_filepath": f"/datasets/audio_sortformer_ami_sdm/audio/{filename}",
            "audio_item_id": filename.removesuffix(".wav"),
            "session_name": filename.removesuffix(".wav"),
            "duration": contract.DATASET_PUBLISHED_MEAN_DURATION_S,
            "timestamps_start": [0.0, 1.0, 2.0],
            "timestamps_end": [1.0, 2.0, 3.0],
            "speakers": ["speaker_a", "speaker_b", "speaker_c"],
        }
        for filename in contract.EXPECTED_AUDIO_FILENAMES
    ]


def test_pinned_sources_are_public_and_sized() -> None:
    assert contract.DATASET_HF_REPO_ID == "diarizers-community/ami"
    assert contract.DATASET_LICENSE == "cc-by-4.0"
    assert contract.DATASET_SPLITS == ("validation", "test")
    assert contract.DATASET_SPLIT_NUM_ROWS == {"validation": 18, "test": 16}
    assert contract.DATASET_NUM_ROWS == 34
    assert contract.DATASET_SOURCE_DOWNLOAD_BYTES == 2_034_736_248
    assert contract.DATASET_DECODED_AUDIO_BYTES == 2_157_687_735
    assert len(set(contract.EXPECTED_AUDIO_FILENAMES)) == 34
    assert len(contract.AUDIO_CORPUS_SHA256) == 64
    assert contract.MODEL_SIZE_BYTES == 471_367_680
    assert contract.MODEL_LICENSE == "nvidia-open-model-license"


def test_validate_manifest_contract_accepts_complete_public_split(monkeypatch: MonkeyPatch) -> None:
    rows = _valid_rows()
    monkeypatch.setattr(contract, "REFERENCE_ANNOTATIONS_SHA256", contract.reference_annotations_sha256(rows))

    contract.validate_manifest(rows, "test manifest")


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("audio_filepath", "/data/unexpected.wav", "unexpected audio file"),
        ("audio_item_id", "ami_sdm_validation_001", "duplicate audio_item_id"),
        ("session_name", "ami_sdm_validation_001", "duplicate session_name"),
        ("duration", 0, "finite positive duration"),
        ("timestamps_end", [1.0], "same nonzero length"),
        ("speakers", ["", "speaker_b", "speaker_c"], "empty reference speaker"),
    ],
)
def test_validate_manifest_contract_rejects_invalid_rows(
    field: str, replacement: object, match: str, monkeypatch: MonkeyPatch
) -> None:
    rows = deepcopy(_valid_rows())
    monkeypatch.setattr(contract, "REFERENCE_ANNOTATIONS_SHA256", contract.reference_annotations_sha256(rows))
    rows[0][field] = replacement

    with pytest.raises(RuntimeError, match=match):
        contract.validate_manifest(rows, "test manifest")


def test_validate_manifest_contract_rejects_truncated_split() -> None:
    with pytest.raises(RuntimeError, match="exactly 34 rows"):
        contract.validate_manifest(_valid_rows()[:-1], "test manifest")


def test_validate_model_rejects_same_size_tamper(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    model_path = tmp_path / "model.nemo"
    good_content = b"good"
    model_path.write_bytes(b"evil")
    monkeypatch.setattr(contract, "MODEL_SIZE_BYTES", len(good_content))
    monkeypatch.setattr(contract, "MODEL_SHA256", contract.hashlib.sha256(good_content).hexdigest())

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        contract.validate_model(model_path)
