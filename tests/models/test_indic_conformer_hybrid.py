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
import torch

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


def test_missing_local_nemo_path_fails_during_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing.nemo"

    with pytest.raises(FileNotFoundError, match=f"Local NeMo checkpoint not found: {checkpoint}"):
        IndicConformerHybridASR._resolve_nemo_path(str(checkpoint))


def test_existing_local_nemo_prefetch_does_not_use_huggingface(tmp_path: Path) -> None:
    checkpoint = tmp_path / "indic.nemo"
    checkpoint.touch()
    adapter = IndicConformerHybridASR(str(checkpoint))

    with (
        patch("huggingface_hub.HfApi") as api,
        patch("huggingface_hub.hf_hub_download") as download,
    ):
        adapter.download_weights_on_node()

    api.assert_not_called()
    download.assert_not_called()


def test_missing_local_nemo_path_fails_during_prefetch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing.nemo"
    adapter = IndicConformerHybridASR(str(checkpoint))

    with pytest.raises(FileNotFoundError, match=f"Local NeMo checkpoint not found: {checkpoint}"):
        adapter.download_weights_on_node()


def test_repo_id_resolves_from_local_snapshot_before_network(tmp_path: Path) -> None:
    checkpoint = tmp_path / "indic.nemo"
    checkpoint.touch()

    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snapshot:
        result = IndicConformerHybridASR._resolve_nemo_path("ai4bharat/model")

    assert result == str(checkpoint)
    snapshot.assert_called_once_with("ai4bharat/model", local_files_only=True)


def test_repo_id_cache_miss_directs_node_prefetch_without_downloading() -> None:
    with (
        patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError),
        patch("huggingface_hub.hf_hub_download") as download,
        pytest.raises(FileNotFoundError, match=r"run download_weights_on_node\(\) during node setup"),
    ):
        IndicConformerHybridASR._resolve_nemo_path("ai4bharat/model")

    download.assert_not_called()


def test_download_weights_on_node_prefetches_huggingface_checkpoint() -> None:
    adapter = IndicConformerHybridASR("ai4bharat/model")
    api = MagicMock()
    api.list_repo_files.return_value = ["README.md", "weights/model.nemo"]

    with (
        patch("huggingface_hub.HfApi", return_value=api),
        patch("huggingface_hub.hf_hub_download", return_value="/cache/model.nemo") as download,
    ):
        adapter.download_weights_on_node()

    api.list_repo_files.assert_called_once_with("ai4bharat/model")
    download.assert_called_once_with("ai4bharat/model", "weights/model.nemo")


def test_download_weights_on_node_resolves_existing_cache_when_offline() -> None:
    adapter = IndicConformerHybridASR("ai4bharat/model")

    with (
        patch.object(adapter, "_offline", return_value=True),
        patch.object(adapter, "_resolve_nemo_path", return_value="/cache/model.nemo") as resolve,
    ):
        adapter.download_weights_on_node()

    resolve.assert_called_once_with("ai4bharat/model")


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

    with patch.object(IndicConformerHybridASR, "download_weights_on_node") as prefetch:
        stage.setup_on_node()

    prefetch.assert_called_once_with()


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
    assert results[0].extras == {"language_code": "hi"}
    assert results[1].unsupported_language == "en"
    assert generate.call_count == 1


def test_generate_requires_upstream_resampling() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo")
    adapter._model = MagicMock()

    with pytest.raises(ValueError, match="ASRStage must provide 16000 Hz"):
        adapter.generate([np.zeros(160, dtype=np.float32)], [8_000], ["hi"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decode_mode": "beam"}, "decode mode"),
        ({"rnnt_precision": "int8"}, "RNNT precision"),
        ({"max_symbols_per_step": 0}, "max_symbols_per_step"),
    ],
)
def test_constructor_rejects_invalid_inference_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IndicConformerHybridASR("checkpoint.nemo", **kwargs)  # type: ignore[arg-type]


def test_generate_encodes_full_duration_ordered_batch_and_restores_input_order() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo", decode_mode="ctc")
    adapter._device = torch.device("cpu")
    model = MagicMock()

    def _encode(*, input_signal: torch.Tensor, input_signal_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del input_signal_length
        batch = input_signal.shape[0]
        return torch.zeros((batch, 2, 3)), torch.ones(batch, dtype=torch.long)

    model.side_effect = _encode
    adapter._model = model
    waveforms = [
        np.zeros(300, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        np.zeros(200, dtype=np.float32),
    ]
    with patch.object(adapter, "_decode_ctc_batch", side_effect=lambda _encoded, _length, langs: langs):
        texts, languages = adapter.generate(waveforms, [16_000] * 3, ["hi", "bn", "ta"])

    assert texts == ["hi", "bn", "ta"]
    assert languages == ["hi", "bn", "ta"]
    assert model.call_count == 1
    assert model.call_args.kwargs["input_signal"].shape[0] == 3


def test_generate_splits_long_audio_pads_tiny_tail_and_merges_in_time_order() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo", decode_mode="ctc")
    adapter._device = torch.device("cpu")
    model = MagicMock()

    def _encode(*, input_signal: torch.Tensor, input_signal_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((input_signal.shape[0], 2, 3)), input_signal_length

    model.side_effect = _encode
    adapter._model = model
    waveform = np.zeros(40 * 16_000 + 1, dtype=np.float32)
    with patch.object(adapter, "_decode_ctc_batch", return_value=["tail", "head"]):
        texts, languages = adapter.generate([waveform], [16_000], ["hi"])

    assert texts == ["head tail"]
    assert languages == ["hi"]
    lengths = model.call_args.kwargs["input_signal_length"].tolist()
    assert lengths == [1_600, 640_000]


def test_generate_uses_batched_rnnt_decoder() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo", decode_mode="rnnt")
    adapter._device = torch.device("cpu")
    model = MagicMock()
    model.side_effect = lambda *, input_signal, input_signal_length: (
        torch.zeros((input_signal.shape[0], 2, 3)),
        input_signal_length,
    )
    adapter._model = model

    with patch.object(adapter, "_decode_rnnt_batch", side_effect=lambda _encoded, _lengths, langs: langs) as decode:
        texts, languages = adapter.generate(
            [np.zeros(300, dtype=np.float32), np.zeros(100, dtype=np.float32), np.zeros(200, dtype=np.float32)],
            [16_000] * 3,
            ["hi", "bn", "ta"],
        )

    assert texts == ["hi", "bn", "ta"]
    assert languages == ["hi", "bn", "ta"]
    decode.assert_called_once()


def test_non_fp32_rnnt_precision_requires_cuda() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo", rnnt_precision="fp16")
    adapter._device = torch.device("cpu")
    adapter._model = MagicMock()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter._configure_rnnt_precision()


def test_empty_audio_can_remain_blank_without_setting_skip() -> None:
    adapter = IndicConformerHybridASR("checkpoint.nemo", empty_audio_marks_skip=False)
    adapter._model = MagicMock()

    result = adapter.transcribe_batch(
        [{"waveform": np.empty(0, dtype=np.float32), "sample_rate": 16_000, "language_code": "hi"}]
    )[0]

    assert result.text == ""
    assert result.skipped is False
    assert result.skip_reason is None
