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

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.band.features import AudioFeatureExtractor
from nemo_curator.models.audio.band.predict import BandAdapter, BandPredictor
from nemo_curator.stages.audio.filtering.band import BandFilterStage


def test_legacy_cache_constructor_is_accepted_without_caching() -> None:
    with patch.object(BandPredictor, "_load_model"):
        predictor = BandPredictor("model.joblib", feature_cache_size=1)
    waveform = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    with (
        patch.object(
            AudioFeatureExtractor,
            "extract_band_features_from_waveform",
            return_value={"feature": 1.0},
        ) as extract,
        patch.object(
            AudioFeatureExtractor,
            "features_dict_to_vector",
            return_value=(np.array([1.0], dtype=np.float32), ["feature"]),
        ),
    ):
        predictor.extract_features_from_audio(waveform, 16000)
        predictor.extract_features_from_audio(waveform, 16000)

    assert predictor.feature_cache_size == 1
    assert predictor.feature_cache == {}
    assert extract.call_count == 2


_DOWNLOAD = "nemo_curator.models.audio.band.predict.hf_hub_download"


def test_explicit_valid_model_path_returns_without_download(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")

    with patch(_DOWNLOAD) as download:
        assert BandAdapter(model_path=str(model_path))._resolve_model_path() == str(model_path)

    download.assert_not_called()


@pytest.mark.parametrize(
    ("name", "content", "match"), [("missing.joblib", None, "model_path"), ("empty.joblib", b"", "empty")]
)
def test_explicit_bad_model_path_does_not_download(
    tmp_path: Path, name: str, content: bytes | None, match: str
) -> None:
    model_path = tmp_path / name
    if content is not None:
        model_path.write_bytes(content)

    with patch(_DOWNLOAD) as download, pytest.raises(FileNotFoundError, match=match):
        BandAdapter(model_path=str(model_path))._resolve_model_path()

    download.assert_not_called()


def test_default_model_path_uses_download() -> None:
    with patch(_DOWNLOAD, return_value="/cache/model.joblib") as download:
        assert BandAdapter(cache_dir="/cache")._resolve_model_path() == "/cache/model.joblib"

    download.assert_called_once()


def test_stage_setup_on_node_prefetches_only_default_model(tmp_path: Path) -> None:
    explicit = tmp_path / "model.joblib"
    explicit.write_bytes(b"model")

    with patch(_DOWNLOAD) as download:
        BandFilterStage(model_path=str(explicit)).setup_on_node()
        download.assert_not_called()

        BandFilterStage(cache_dir=str(tmp_path)).setup_on_node()
        download.assert_called_once()


def test_adapter_lifecycle_wraps_predictor(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    adapter = BandAdapter(model_path=str(model_path))

    with patch("nemo_curator.models.audio.band.predict.BandPredictor") as predictor:
        adapter.load_model(num_gpus=0)
        adapter.load_model(num_gpus=0)
        predictor.return_value.predict_audio.return_value = "full_band"

        assert adapter.predict_audio(torch.zeros(1, 8), 16000) == "full_band"

    predictor.assert_called_once_with(model_path=str(model_path))
    adapter.unload_model()
    with pytest.raises(RuntimeError, match="load_model"):
        adapter.predict_audio(torch.zeros(1, 8), 16000)
