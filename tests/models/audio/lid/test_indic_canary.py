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

from nemo_curator.models.audio.lid.base import AudioLIDResult
from nemo_curator.models.audio.lid.indic_canary import (
    _DEFAULT_RUNTIME_CLASS_PATH,
    IndicCanaryLIDAdapter,
    _language_code_from_special_token,
    _resolve_runtime_class,
    _validate_engine_dir,
)


def _engine_bundle(root: Path) -> Path:
    for relative_path in (
        "encoder/encoder.plan",
        "encoder/config.json",
        "decoder/rank0.engine",
        "decoder/config.json",
        "decoder/vocab.json",
        "preprocessor/config.json",
        "preprocessor/mel_basis.pt",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("<|hi|>", "hi"),
        ("<|eng|>", "eng"),
        ("<|en-US|>", "en-us"),
        ("<|pnc|>", None),
        ("<|itn|>", None),
        ("<|startoftranscript|>", None),
        ("hi", None),
        ("<|en-USA|>", None),
        ("<|0|>", None),
    ],
)
def test_language_special_token_parser(token: str, expected: str | None) -> None:
    assert _language_code_from_special_token(token) == expected


def test_engine_validation_lists_all_missing_required_files(tmp_path: Path) -> None:
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "encoder.plan").touch()

    with pytest.raises(FileNotFoundError, match=r"decoder/config.json") as error:
        _validate_engine_dir(tmp_path)

    assert "preprocessor/mel_basis.pt" in str(error.value)


def test_complete_engine_bundle_passes_node_prefetch(tmp_path: Path) -> None:
    engine_dir = _engine_bundle(tmp_path)

    IndicCanaryLIDAdapter(engine_dir=str(engine_dir)).download_weights_on_node()


def test_default_runtime_reuses_the_shared_indic_canary_runtime() -> None:
    runtime_class = _resolve_runtime_class(_DEFAULT_RUNTIME_CLASS_PATH)

    assert runtime_class.__name__ == "CanaryTRTLLM"
    assert runtime_class.__module__ == "nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime"


class _Runtime:
    captured: tuple[Path, dict[str, object]] | None = None

    def __init__(self, engine_dir: Path, **kwargs: object) -> None:
        type(self).captured = engine_dir, kwargs
        self.tokenizer = SimpleNamespace(
            prompt_format="canary1",
            id_to_token={4: "<|startoftranscript|>", 5: "<|pnc|>", 89: "<|hi|>", 185: "<|ta|>"},
            encode=lambda _prompt: [4],
            eos_id=3,
            pad_id=2,
        )
        self.decoder = SimpleNamespace(max_input_len=8)
        self.max_batch_size = 8
        self.device = "cpu"
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_injected_runtime_loads_lazily_and_filters_candidates(tmp_path: Path) -> None:
    engine_dir = _engine_bundle(tmp_path)
    adapter = IndicCanaryLIDAdapter(
        engine_dir=str(engine_dir),
        runtime_class=_Runtime,
        candidate_languages=[" HI "],
        kv_cache_free_gpu_memory_fraction=0.15,
        cross_kv_cache_fraction=0.25,
    )

    with patch("torch.cuda.is_available", return_value=True):
        adapter.load_model(num_gpus=1)

    assert adapter._language_by_token_id == {89: "hi"}
    assert _Runtime.captured == (
        engine_dir,
        {
            "device": "cuda:0",
            "kv_cache_free_gpu_memory_fraction": 0.15,
            "cross_kv_cache_fraction": 0.25,
        },
    )
    runtime = adapter._model
    with patch("torch.cuda.empty_cache"):
        adapter.unload_model()
    assert runtime.closed


def test_failed_candidate_filter_keeps_runtime_reachable_for_cleanup(tmp_path: Path) -> None:
    engine_dir = _engine_bundle(tmp_path)
    adapter = IndicCanaryLIDAdapter(
        engine_dir=str(engine_dir),
        runtime_class=_Runtime,
        candidate_languages=["not-a-canary-language"],
    )

    with (
        patch("torch.cuda.is_available", return_value=True),
        pytest.raises(RuntimeError, match="no language special tokens"),
    ):
        adapter.load_model(num_gpus=1)

    runtime = adapter._model
    assert runtime is not None
    with patch("torch.cuda.empty_cache"):
        adapter.unload_model()
    assert runtime.closed


def _loaded_adapter(*, candidate_languages: list[str] | None = None) -> IndicCanaryLIDAdapter:
    adapter = IndicCanaryLIDAdapter(engine_dir="unused", candidate_languages=candidate_languages)
    adapter._model = _Runtime(Path("unused"))
    adapter._language_by_token_id = adapter._collect_language_token_ids()
    return adapter


def test_prompts_and_language_parsing_match_reference_semantics() -> None:
    adapter = _loaded_adapter()
    assert adapter._lid_prompt_ids() == [4]
    assert adapter._parse_language([4, 89, 3], [4]) == AudioLIDResult("hi", 1.0)
    assert adapter._parse_language([185, 3], [4]) == AudioLIDResult("ta", 1.0)

    adapter._model.tokenizer.prompt_format = "canary2"
    prompts: list[str] = []
    adapter._model.tokenizer.encode = lambda prompt: prompts.append(prompt) or [7, 4, 18]
    assert adapter._lid_prompt_ids() == [7, 4, 18]
    assert prompts == ["<|startofcontext|> <|startoftranscript|> <|emo:undefined|>"]


def test_empty_prompt_text_uses_the_reference_default_prompt() -> None:
    adapter = _loaded_adapter()
    adapter.prompt_text = ""

    assert adapter._lid_prompt_ids() == [4]


def test_candidate_filter_makes_other_language_unknown() -> None:
    adapter = _loaded_adapter(candidate_languages=["hi"])

    assert adapter._parse_language([4, 185, 3], [4]) == AudioLIDResult("", 0.0)


def test_identify_batch_preserves_short_rows_and_truncates_long_rows() -> None:
    adapter = _loaded_adapter()
    adapter.max_duration_sec = 2.0
    captured: list[int] = []

    def identify(waveforms):  # noqa: ANN001, ANN202
        captured.extend(waveform.numel() for waveform in waveforms)
        return [AudioLIDResult("hi", 1.0) for _ in waveforms]

    adapter._identify_waveforms = identify
    results = adapter.identify_batch(
        [
            {"waveform": np.zeros(8_000, dtype=np.float32)},
            {"waveform": np.zeros(16_000, dtype=np.float32)},
            {"waveform": np.zeros(48_000, dtype=np.float32)},
        ]
    )

    assert results == [AudioLIDResult("", 0.0), AudioLIDResult("hi", 1.0), AudioLIDResult("hi", 1.0)]
    assert captured == [16_000, 32_000]


def test_identify_batch_respects_engine_capacity_and_runtime_contract() -> None:
    adapter = _loaded_adapter()
    adapter.max_duration_sec = 2.0
    adapter.model_batch_size = 3
    adapter.num_beams = 2
    adapter.max_new_tokens = 3
    runtime = adapter._model
    runtime.max_batch_size = 2
    runtime.preprocessor = SimpleNamespace(
        get_feats=MagicMock(side_effect=lambda audio, lengths: (torch.stack(audio), torch.tensor(lengths)))
    )
    runtime.encoder = SimpleNamespace(infer=MagicMock(side_effect=lambda mel, lengths, _stream: (mel, lengths)))
    runtime.decoder.generate = MagicMock(
        side_effect=[
            [[[4, 89, 3]], [[4, 185, 3]]],
            [[[4, 185, 3]], [[4, 89, 3]]],
            [[[4, 185, 3]]],
        ]
    )
    stream = object()
    with patch("torch.cuda.current_stream", return_value=stream):
        results = adapter.identify_batch(
            [
                {"waveform": np.ones(length, dtype=np.float32)}
                for length in (16_000, 0, 48_000, 24_000, 8_000, 16_000, 32_000)
            ]
        )

    assert results == [
        AudioLIDResult("hi", 1.0),
        AudioLIDResult("", 0.0),
        AudioLIDResult("ta", 1.0),
        AudioLIDResult("ta", 1.0),
        AudioLIDResult("", 0.0),
        AudioLIDResult("hi", 1.0),
        AudioLIDResult("ta", 1.0),
    ]
    for preprocessing, encoding, decoding, expected_lengths in zip(
        runtime.preprocessor.get_feats.call_args_list,
        runtime.encoder.infer.call_args_list,
        runtime.decoder.generate.call_args_list,
        ([16_000, 32_000], [24_000, 16_000], [32_000]),
        strict=True,
    ):
        audio, lengths = preprocessing.args
        assert lengths == expected_lengths
        for waveform, length in zip(audio, lengths, strict=True):
            assert waveform.shape == (max(lengths),)
            torch.testing.assert_close(waveform[:length], torch.ones(length))
            torch.testing.assert_close(waveform[length:], torch.zeros(max(lengths) - length))
        torch.testing.assert_close(encoding.args[0], torch.stack(audio))
        torch.testing.assert_close(encoding.args[1], torch.tensor(lengths))
        assert encoding.args[2] is stream
        prompt_ids, encoded, encoded_lengths = decoding.args
        torch.testing.assert_close(prompt_ids, torch.full((len(lengths), 1), 4, dtype=torch.int64))
        assert encoded is encoding.args[0]
        assert encoded_lengths is encoding.args[1]
        assert decoding.kwargs == {"max_new_tokens": 3, "num_beams": 2}


def test_default_duration_floor_matches_the_reference_stage_and_engine_profile() -> None:
    adapter = IndicCanaryLIDAdapter(engine_dir="unused")

    assert adapter.min_duration_sec == 1.0
    assert adapter.min_samples == 16_000


@pytest.mark.parametrize("num_gpus", [0, 2, -1, True])
def test_load_requires_exactly_one_gpu(tmp_path: Path, num_gpus: object) -> None:
    adapter = IndicCanaryLIDAdapter(engine_dir=str(_engine_bundle(tmp_path)), runtime_class=_Runtime)

    with pytest.raises(ValueError, match="exactly 1"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]
