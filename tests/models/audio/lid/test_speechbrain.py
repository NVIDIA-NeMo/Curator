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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.lid.speechbrain import SpeechBrainLIDAdapter, _normalize_language


class _Classifier:
    def __init__(self) -> None:
        self.batch: torch.Tensor | None = None
        self.lengths: torch.Tensor | None = None

    def classify_batch(
        self,
        batch: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, list[object]]:
        self.batch = batch
        self.lengths = lengths
        return None, torch.log(torch.tensor([0.8, 0.25])), None, ["TA: Tamil", ["en: English"]]


def test_label_normalization_accepts_speechbrain_singletons() -> None:
    assert _normalize_language(["TA: Tamil"]) == "ta"


def test_identify_batch_pads_valid_rows_and_preserves_empty_order() -> None:
    adapter = SpeechBrainLIDAdapter()
    classifier = _Classifier()
    adapter._classifier = classifier
    items = [
        {"waveform": np.ones(4, dtype=np.float32)},
        {"waveform": np.array([], dtype=np.float32)},
        {"waveform": np.ones(2, dtype=np.float32)},
    ]

    results = adapter.identify_batch(items)

    assert [(result.language, result.confidence) for result in results] == [
        ("ta", pytest.approx(0.8)),
        ("", 0.0),
        ("en", pytest.approx(0.25)),
    ]
    assert classifier.batch is not None
    assert classifier.batch.shape == (2, 4)
    torch.testing.assert_close(classifier.lengths, torch.tensor([1.0, 0.5]))


def test_download_uses_provider_source_not_stage_result_key() -> None:
    adapter = SpeechBrainLIDAdapter(source="organization/custom-lid", revision="abc123", cache_dir="/cache")

    with patch("nemo_curator.models.audio.lid.speechbrain._snapshot_download") as download:
        adapter.download_weights_on_node()

    download.assert_called_once_with(repo_id="organization/custom-lid", revision="abc123", cache_dir="/cache")


def test_local_source_needs_no_huggingface_download(tmp_path: Path) -> None:
    adapter = SpeechBrainLIDAdapter(source=str(tmp_path))

    with patch("nemo_curator.models.audio.lid.speechbrain._snapshot_download") as download:
        adapter.download_weights_on_node()

    download.assert_not_called()


def test_load_model_forwards_worker_device_and_actor_savedir(tmp_path: Path) -> None:
    classifier = MagicMock()
    classifier_type = MagicMock()
    classifier_type.from_hparams.return_value = classifier
    adapter = SpeechBrainLIDAdapter(source=str(tmp_path), savedir=str(tmp_path / "actors"))

    with (
        patch("nemo_curator.models.audio.lid.speechbrain._encoder_classifier_class", return_value=classifier_type),
        patch("torch.cuda.is_available", return_value=False),
    ):
        adapter.load_model(num_gpus=0)

    kwargs = classifier_type.from_hparams.call_args.kwargs
    assert kwargs["source"] == str(tmp_path)
    assert Path(kwargs["savedir"]).parent == tmp_path / "actors"
    assert kwargs["run_opts"] == {"device": "cpu"}
    assert adapter._classifier is classifier


def test_nonempty_batch_requires_loaded_classifier() -> None:
    with pytest.raises(RuntimeError, match="load_model"):
        SpeechBrainLIDAdapter().identify_batch([{"waveform": np.ones(4, dtype=np.float32)}])
