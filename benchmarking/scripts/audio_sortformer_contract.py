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

"""Pinned public input contract for the Sortformer nightly benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_HF_REPO_ID = "diarizers-community/ami"
DATASET_REVISION = "8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0"
DATASET_CONFIG = "sdm"
DATASET_SPLITS = ("validation", "test")
DATASET_SPLIT_NUM_ROWS = {"validation": 18, "test": 16}
DATASET_SPLIT_SOURCE_DOWNLOAD_BYTES = {"validation": 1_062_066_366, "test": 972_669_882}
DATASET_SPLIT_DECODED_AUDIO_BYTES = {"validation": 1_113_699_511, "test": 1_043_988_224}
DATASET_LICENSE = "cc-by-4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_NUM_ROWS = sum(DATASET_SPLIT_NUM_ROWS.values())
DATASET_SOURCE_DOWNLOAD_BYTES = sum(DATASET_SPLIT_SOURCE_DOWNLOAD_BYTES.values())
DATASET_DECODED_AUDIO_BYTES = sum(DATASET_SPLIT_DECODED_AUDIO_BYTES.values())
DATASET_PUBLISHED_TOTAL_DURATION_S = 67_427.088
DATASET_PUBLISHED_MEAN_DURATION_S = DATASET_PUBLISHED_TOTAL_DURATION_S / DATASET_NUM_ROWS
MIN_DATASET_DURATION_S = 67_426.0
MAX_DATASET_DURATION_S = 67_428.0
REFERENCE_ANNOTATIONS_SHA256 = "ad548f866d578402a03dc6e10fb92c613092f862e7ca0ec0592a8e74c114ad99"

MODEL_HF_REPO_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"
MODEL_LICENSE = "nvidia-open-model-license"
MODEL_LICENSE_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
MODEL_SIZE_BYTES = 471_367_680
MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"

MANIFEST_FILENAME = "manifest.jsonl"
SOURCE_METADATA_FILENAME = "source_metadata.json"
EXPECTED_AUDIO_FILENAMES = tuple(
    f"ami_sdm_{split}_{index:03d}.wav" for split in DATASET_SPLITS for index in range(DATASET_SPLIT_NUM_ROWS[split])
)
DATASET_AUDIO_SHA256 = {
    "ami_sdm_validation_000.wav": "234da95a325d29a3a1cd1e138470504ec99a0f39a094bbd033895fadf324f196",
    "ami_sdm_validation_001.wav": "ac7fa631929af2232d5d3798e4da6afbd9ea30dbc36654e95c7b792124fef368",
    "ami_sdm_validation_002.wav": "832c95247c9a4bc04c3f7adb128158297e3513a0c63b017055e3cc9d938eb6c2",
    "ami_sdm_validation_003.wav": "a76f820b0af6adb274f94454521cd82d446a69dace3075a7e623c720d3e19c01",
    "ami_sdm_validation_004.wav": "6bc15f79d36d630c681c640c6dedf51a19fd18ffda894f89804a380222f86e4c",
    "ami_sdm_validation_005.wav": "b826877e3b2bbf8dc172c578017a06e6fbace086834f3ee9dfd73ccf6f212a8e",
    "ami_sdm_validation_006.wav": "762d9b3aed62494828cca0ebaa0f0ed304fd39c9c2bd73d327098234b8fd1234",
    "ami_sdm_validation_007.wav": "eef59d96abe292a2b421b39d8ea8fea250210a9028bfe92758d58e51d18f9f7a",
    "ami_sdm_validation_008.wav": "c23493df44c3be46665d6e78484f695c95164e48561d96d00c6c3c1ff156825f",
    "ami_sdm_validation_009.wav": "5162f6d349a575c2461218b1be8c8aea436410da2c7bb58c654197136d213e4f",
    "ami_sdm_validation_010.wav": "a46e8a53874f6ace4b19eb00ad1cd1131e37c569548d9d3f3786f3c5478f9e1a",
    "ami_sdm_validation_011.wav": "c9904b7c544eb2acec78ccfbe2567840172162686f26bba34b0a35d021be02d0",
    "ami_sdm_validation_012.wav": "037e723a01725fe5119c006bd153c06c3025186710979180c068a81a484e3789",
    "ami_sdm_validation_013.wav": "4855c323c6590973c6063b38abc3f39b15b7f46a91fcfd272e48d7514e7e3a4d",
    "ami_sdm_validation_014.wav": "9b194efd606e44b14f74845a90d9aa9f592d084d30f9ae312faf857866040ec6",
    "ami_sdm_validation_015.wav": "1557798b61a3673c636bd7018949daada1d4f3b53fcbc16cc804be1e40a7031d",
    "ami_sdm_validation_016.wav": "dc8157fd7379e1c7b6a9b69fd5fad6a6bebbc359f914e906a82119d0613e2593",
    "ami_sdm_validation_017.wav": "850a857d883609a29484424c005f794f97ff4a7de319dbbd199da0951a0c2393",
    "ami_sdm_test_000.wav": "e3650cc84250caa0992ac0398f4a3127a8c14ed2c93edc10535245335cc41077",
    "ami_sdm_test_001.wav": "46f81fb403e40a98c1b694c842404c0db24f3a6c67397b247faf0d3f42cf9163",
    "ami_sdm_test_002.wav": "dc11abf7f87d1ca14a4d08f31d2356ec06541a19bae63700d6e64951d69c0cb7",
    "ami_sdm_test_003.wav": "8a58a53413f7a47a29f8bd4d1b49e42cd4874eecc0c3f023c28a4702ecb50d88",
    "ami_sdm_test_004.wav": "dea4276863ba77b49d304e28638ae354bcd815e4ebeecf98902ce30782f8ba5a",
    "ami_sdm_test_005.wav": "23fd54fd6dbe23870f10317b256f7c648fe0fd587e56d2f940840309f1578b60",
    "ami_sdm_test_006.wav": "7f568d1eafa0802a0eb6d8ad975eb182568dfda2b815ae338fd8b9967aa2b5d1",
    "ami_sdm_test_007.wav": "d4f4556adb90eac335d3a50e5cfcd36b3186e8de0ba1b3f0b07a3d21088e7ba3",
    "ami_sdm_test_008.wav": "7a131dcaaf871a32a0af9ac0c9a21acbf78ed4f2e343d29969a7f4551c966723",
    "ami_sdm_test_009.wav": "b2957066c100051d7835ac50231ec55ad5f35fafb2e2a9340670a0acaefb4173",
    "ami_sdm_test_010.wav": "f55f83cf1f563c460030bf2790849428a102c41f296c401cedfa1be96ab1f2f7",
    "ami_sdm_test_011.wav": "16f5463568010ac6c8c71d2d5f9d703b71b200c7748c8b6a5f608118bf8b16d0",
    "ami_sdm_test_012.wav": "6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150",
    "ami_sdm_test_013.wav": "5e8b41c7ab6f7dca07c52c354ece85e31a71491a47ef0493407cb89b12a41f60",
    "ami_sdm_test_014.wav": "53211f01571e702be9c7871dd042951c3a787c85d5f501fa950967acd5374b9f",
    "ami_sdm_test_015.wav": "6c0644431679e59adc70129f2ba6c2fb737fa731cf1ffe8a033cd7f0c441170e",
}
AUDIO_SAMPLE_RATE = 16_000
MONO_CHANNELS = 1
TIMESTAMP_TOLERANCE_S = 1e-3


def source_metadata() -> dict[str, Any]:
    """Return the exact public source metadata written beside staged inputs."""
    return {
        "dataset": {
            "hf_repo_id": DATASET_HF_REPO_ID,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "splits": list(DATASET_SPLITS),
            "split_num_rows": dict(DATASET_SPLIT_NUM_ROWS),
            "split_source_download_bytes": dict(DATASET_SPLIT_SOURCE_DOWNLOAD_BYTES),
            "split_decoded_audio_bytes": dict(DATASET_SPLIT_DECODED_AUDIO_BYTES),
            "license": DATASET_LICENSE,
            "license_url": DATASET_LICENSE_URL,
            "public": True,
            "gated": False,
            "num_rows": DATASET_NUM_ROWS,
            "source_download_bytes": DATASET_SOURCE_DOWNLOAD_BYTES,
            "decoded_audio_bytes": DATASET_DECODED_AUDIO_BYTES,
            "published_total_duration_s": DATASET_PUBLISHED_TOTAL_DURATION_S,
            "audio_sha256": dict(DATASET_AUDIO_SHA256),
            "reference_annotations_sha256": REFERENCE_ANNOTATIONS_SHA256,
        },
        "model": {
            "hf_repo_id": MODEL_HF_REPO_ID,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "license": MODEL_LICENSE,
            "license_url": MODEL_LICENSE_URL,
            "public": True,
            "gated": False,
            "size_bytes": MODEL_SIZE_BYTES,
            "sha256": MODEL_SHA256,
        },
    }


def sha256(path: Path) -> str:
    """Hash a staged file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_annotations_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the semantic manifest fields independently of staged path prefixes."""
    canonical_rows = [
        {
            "audio_filename": Path(row["audio_filepath"]).name,
            "audio_item_id": str(row["audio_item_id"]),
            "session_name": str(row["session_name"]),
            "duration": float(row["duration"]),
            "timestamps_start": [float(value) for value in row["timestamps_start"]],
            "timestamps_end": [float(value) for value in row["timestamps_end"]],
            "speakers": [str(value) for value in row["speakers"]],
        }
        for row in rows
    ]
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_model(model_path: Path) -> None:
    """Require the exact pinned public Sortformer checkpoint."""
    if not model_path.is_file():
        msg = f"Sortformer model not found: {model_path}"
        raise FileNotFoundError(msg)
    actual_size = model_path.stat().st_size
    if actual_size != MODEL_SIZE_BYTES:
        msg = f"Sortformer model size mismatch: expected {MODEL_SIZE_BYTES}, found {actual_size}"
        raise RuntimeError(msg)
    actual_sha256 = sha256(model_path)
    if actual_sha256 != MODEL_SHA256:
        msg = f"Sortformer model SHA-256 mismatch: expected {MODEL_SHA256}, found {actual_sha256}"
        raise RuntimeError(msg)


def validate_reference_annotations(row: dict[str, Any], duration_s: float, label: str) -> None:
    """Validate the AMI reference turns retained for semantic scoring."""
    starts = row.get("timestamps_start")
    ends = row.get("timestamps_end")
    speakers = row.get("speakers")
    if not isinstance(starts, list) or not isinstance(ends, list) or not isinstance(speakers, list):
        msg = f"{label} must contain timestamps_start, timestamps_end, and speakers lists"
        raise TypeError(msg)
    if not starts or len(starts) != len(ends) or len(starts) != len(speakers):
        msg = f"{label} reference annotation lists must have the same nonzero length"
        raise RuntimeError(msg)

    for segment_index, (start, end, speaker) in enumerate(zip(starts, ends, speakers, strict=True)):
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not 0 <= float(start) < float(end) <= duration_s + TIMESTAMP_TOLERANCE_S
        ):
            msg = f"{label} has invalid reference segment {segment_index}: start={start!r}, end={end!r}"
            raise RuntimeError(msg)
        if not isinstance(speaker, str) or not speaker:
            msg = f"{label} has an empty reference speaker at segment {segment_index}"
            raise RuntimeError(msg)

    num_speakers = len(set(speakers))
    if not 3 <= num_speakers <= 4:  # noqa: PLR2004
        msg = f"{label} must contain 3-4 AMI speakers, found {num_speakers}"
        raise RuntimeError(msg)
