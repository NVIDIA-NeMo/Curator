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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.indic_conformer_hybrid import IndicConformerHybridASR
from nemo_curator.stages.audio.inference.asr.stage import ASRStage

_ADAPTER_TARGET = "nemo_curator.models.indic_conformer_hybrid.IndicConformerHybridASR"


def test_adapter_conforms_to_shared_protocol() -> None:
    assert isinstance(IndicConformerHybridASR("checkpoint.nemo"), ASRAdapter)


def test_local_nemo_path_is_used_without_hub_download(tmp_path: Path) -> None:
    checkpoint = tmp_path / "indic.nemo"
    checkpoint.touch()

    assert IndicConformerHybridASR._resolve_nemo_path(str(checkpoint)) == str(checkpoint)


def test_local_token_ids_are_mapped_through_aggregate_tokenizer() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")
    tokenizer = SimpleNamespace(
        token_id_offset={"hi": 100},
        ids_to_text=lambda ids: f"tokens={ids}",
    )
    adapter._model = SimpleNamespace(tokenizer=tokenizer)

    assert adapter._ids_to_text([1, 2], "hi") == "tokens=[101, 102]"


def test_empty_token_sequence_decodes_to_empty_text() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")

    assert adapter._ids_to_text([], "hi") == ""


def test_stage_prefetch_resolves_checkpoint_without_loading_model() -> None:
    stage = ASRStage(adapter_target=_ADAPTER_TARGET, model_id="ai4bharat/model")

    with patch.object(IndicConformerHybridASR, "_resolve_nemo_path") as resolve:
        stage.setup_on_node()

    resolve.assert_called_once_with("ai4bharat/model")


def test_transcribe_batch_routes_supported_languages_through_model() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")
    adapter._model = MagicMock()
    with patch.object(adapter, "generate", return_value=(["नमस्ते"], ["hi"])) as generate:
        results = adapter.transcribe_batch(
            [
                {
                    "waveform": np.zeros(160, dtype=np.float32),
                    "sample_rate": 16_000,
                    "language_code": "hi",
                },
                {
                    "waveform": np.zeros(160, dtype=np.float32),
                    "sample_rate": 16_000,
                    "language_code": "en",
                },
            ]
        )

    assert results[0].text == "नमस्ते"
    assert results[1].unsupported_language == "en"
    assert generate.call_count == 1


def test_generate_requires_upstream_resampling() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")
    adapter._model = MagicMock()

    with pytest.raises(ValueError, match="ASRStage must provide 16000 Hz"):
        adapter.generate([np.zeros(160, dtype=np.float32)], [8_000], ["hi"])
