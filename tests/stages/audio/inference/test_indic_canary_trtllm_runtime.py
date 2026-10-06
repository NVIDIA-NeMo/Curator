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

"""CPU-only unit tests for the vendored Indic Canary TensorRT-LLM runtime."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference import indic_canary_trtllm_runtime as runtime


def _write_decoder_bundle(tmp_path: Path, *, prompt_format: str = "canary2", country_codes: bool = True) -> Path:
    decoder_dir = tmp_path / "decoder"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    languages = ["en-US", "hi-IN", "ja-JP"] if country_codes else ["en", "hi", "ja"]
    language_tokens = languages if country_codes else ["en", "hi", "ja"]
    special_values = [
        "▁",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|endoftext|>",
        "<|emo:undefined|>",
        "<|transcribe|>",
        "<|translate|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|itn|>",
        "<|noitn|>",
        "<|romanized|>",
        "<|noromanized|>",
        "<|timestamp|>",
        "<|notimestamp|>",
        "<|diarize|>",
        "<|nodiarize|>",
        *[f"<|{language}|>" for language in language_tokens],
    ]
    tokens: dict[str, dict[str, str]] = {
        "spl_tokens": {str(index): token for index, token in enumerate(special_values)},
        languages[0]: {"100": "▁hello", "101": "▁world"},
        languages[1]: {"200": "▁namaste", "201": "!"},
        languages[2]: {"300": "日", "301": ",", "302": "本"},
    }
    vocab = {
        "offsets": {"spl_tokens": 0, languages[0]: 100, languages[1]: 200, languages[2]: 300},
        "tokens": tokens,
        "bos_id": 2,
        "eos_id": 3,
        "nospeech_id": 99,
        "pad_id": 0,
    }
    (decoder_dir / "vocab.json").write_text(json.dumps(vocab))
    (decoder_dir / "config.json").write_text(
        json.dumps(
            {
                "pretrained_config": {
                    "dtype": "float32",
                    "max_seq_len": 32,
                    "max_input_len": 16,
                    "prompt_format": prompt_format,
                },
                "build_config": {"max_batch_size": 4, "max_beam_width": 2},
            }
        )
    )
    return tmp_path


@pytest.mark.parametrize("array_type", [np.asarray, torch.as_tensor])
def test_pad_or_trim_truncates_and_pads(
    array_type: Callable[[list[float]], np.ndarray | torch.Tensor],
) -> None:
    values = array_type([1.0, 2.0, 3.0])

    truncated = runtime.pad_or_trim(values, length=2)
    padded = runtime.pad_or_trim(values, length=5)

    np.testing.assert_array_equal(np.asarray(truncated), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(padded), [1.0, 2.0, 3.0, 0.0, 0.0])


def test_pad_or_trim_honors_nonfinal_axis() -> None:
    values = np.arange(6).reshape(2, 3)

    result = runtime.pad_or_trim(values, length=3, axis=0)

    np.testing.assert_array_equal(result, [[0, 1, 2], [3, 4, 5], [0, 0, 0]])


def test_unpack_tensors_uses_each_row_length() -> None:
    values = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result = runtime.unpack_tensors(values, torch.tensor([1, 2]))

    assert [row.tolist() for row in result] == [[1], [4, 5]]


def test_read_config_merges_decoder_sections_in_order(tmp_path: Path) -> None:
    decoder_dir = tmp_path / "decoder"
    decoder_dir.mkdir()
    (decoder_dir / "config.json").write_text(
        json.dumps(
            {
                "pretrained_config": {"dtype": "float16", "shared": "pretrained"},
                "build_config": {"max_batch_size": 8, "shared": "build"},
            }
        )
    )

    config = runtime.read_config("decoder", tmp_path)

    assert list(config) == ["dtype", "shared", "max_batch_size"]
    assert config == {"dtype": "float16", "shared": "build", "max_batch_size": 8}


def test_read_config_preserves_encoder_config(tmp_path: Path) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(json.dumps({"max_batch_size": 4, "precision": "fp16"}))

    config = runtime.read_config("encoder", tmp_path, mode="encoder")

    assert config == {"max_batch_size": 4, "precision": "fp16"}


def test_engine_max_batch_size_uses_the_smaller_static_engine_limit() -> None:
    assert runtime.engine_max_batch_size({"max_batch_size": 8}, {"max_batch_size": 2}) == 2


@pytest.mark.parametrize(("encoder_limit", "decoder_limit"), [(0, 2), (8, 0), (-1, 2)])
def test_engine_max_batch_size_rejects_nonpositive_limits(encoder_limit: int, decoder_limit: int) -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        runtime.engine_max_batch_size(
            {"max_batch_size": encoder_limit},
            {"max_batch_size": decoder_limit},
        )


def test_missing_optional_runtime_is_reported_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime, "_TRTLLM_IMPORT_ERROR", ImportError("not installed"))

    with pytest.raises(ImportError, match="audio_canary_trtllm"):
        runtime.CanaryTRTLLM(tmp_path)


def test_prepare_native_runtime_loads_present_wheel_libraries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "environment"
    (prefix / "share/openmpi").mkdir(parents=True)
    fake_torch_file = tmp_path / "site-packages/torch/__init__.py"
    fake_torch_file.parent.mkdir(parents=True)
    fake_torch_file.touch()
    cudart = tmp_path / "site-packages/nvidia/cu13/lib/libcudart.so.13"
    cudart.parent.mkdir(parents=True)
    cudart.touch()
    loaded: list[tuple[str, int]] = []

    monkeypatch.setattr(runtime, "_NATIVE_RUNTIME_PREPARED", False)
    monkeypatch.setattr(runtime, "_NATIVE_LIBRARY_HANDLES", [])
    monkeypatch.setattr(runtime.sys, "prefix", str(prefix))
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.torch, "__file__", str(fake_torch_file))
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda path, mode: loaded.append((path, mode)) or object())
    monkeypatch.delenv("OPAL_PREFIX", raising=False)

    runtime._prepare_tensorrt_llm_native_runtime()
    runtime._prepare_tensorrt_llm_native_runtime()

    assert runtime._NATIVE_RUNTIME_PREPARED
    assert runtime.os.environ["OPAL_PREFIX"] == str(prefix)
    assert loaded == [(str(cudart), runtime.ctypes.RTLD_GLOBAL)]
    assert len(runtime._NATIVE_LIBRARY_HANDLES) == 1


def test_require_tensorrt_llm_populates_lazy_runtime_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    trt_module = ModuleType("tensorrt")
    trtllm_module = ModuleType("tensorrt_llm")
    trtllm_module.__path__ = []  # type: ignore[attr-defined]
    utils_module = ModuleType("tensorrt_llm._utils")
    bindings_module = ModuleType("tensorrt_llm.bindings")
    runtime_module = ModuleType("tensorrt_llm.runtime")
    runtime_module.__path__ = []  # type: ignore[attr-defined]
    session_module = ModuleType("tensorrt_llm.runtime.session")
    sentinels = [object() for _ in range(6)]
    utils_module.str_dtype_to_torch, utils_module.trt_dtype_to_torch = sentinels[:2]
    bindings_module.KVCacheType = sentinels[2]
    runtime_module.ModelRunnerCpp = sentinels[3]
    session_module.Session, session_module.TensorInfo = sentinels[4:]
    for name, module in {
        "tensorrt": trt_module,
        "tensorrt_llm": trtllm_module,
        "tensorrt_llm._utils": utils_module,
        "tensorrt_llm.bindings": bindings_module,
        "tensorrt_llm.runtime": runtime_module,
        "tensorrt_llm.runtime.session": session_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in (
        "trt",
        "tensorrt_llm",
        "str_dtype_to_torch",
        "trt_dtype_to_torch",
        "KVCacheType",
        "ModelRunnerCpp",
        "Session",
        "TensorInfo",
    ):
        monkeypatch.setattr(runtime, name, None)
    monkeypatch.setattr(runtime, "_TRTLLM_IMPORT_ERROR", None)
    monkeypatch.setattr(runtime, "_prepare_tensorrt_llm_native_runtime", lambda: None)

    runtime._require_tensorrt_llm()
    runtime._require_tensorrt_llm()

    assert runtime.trt is trt_module
    assert runtime.tensorrt_llm is trtllm_module
    assert runtime.str_dtype_to_torch is sentinels[0]
    assert runtime.TensorInfo is sentinels[5]


@pytest.mark.parametrize("window_type", ["hann", "hamming", "bartlett", "blackman", "rectangular"])
def test_mel_filterbank_constructs_supported_windows(window_type: str) -> None:
    mel_basis = torch.ones((1, 2, 5))

    preprocessor = runtime.MelFilterBankFeats(
        mel_basis,
        window_size=0.25,
        window_stride=0.125,
        window_type=window_type,
        preemp=0.0,
        fs=16,
        device="cpu",
    )

    assert preprocessor.nfft == 8
    assert preprocessor.nfilt == 2
    assert preprocessor.preemp is None
    assert preprocessor.window.shape == (4,)


def test_mel_filterbank_normalization_modes_and_sequence_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    values = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]]])
    lengths = torch.tensor([3])

    per_feature = runtime.MelFilterBankFeats.normalize_batch(values, lengths, "per_feature")
    all_features = runtime.MelFilterBankFeats.normalize_batch(values, lengths, "all_features")
    unchanged = runtime.MelFilterBankFeats.normalize_batch(values, lengths, "none")

    torch.testing.assert_close(per_feature.mean(dim=2), torch.zeros((1, 2)), atol=2e-5, rtol=0)
    torch.testing.assert_close(all_features.mean(), torch.tensor(0.0), atol=2e-5, rtol=0)
    assert unchanged is values

    preprocessor = runtime.MelFilterBankFeats(
        torch.ones((1, 2, 5)), nfft=8, window_size=0.25, window_stride=0.125, fs=16, device="cpu"
    )
    assert preprocessor.get_feat_seq_len(torch.tensor([16, 8])).tolist() == [9, 5]

    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(ValueError, match="tensor of length 1"):
        runtime.MelFilterBankFeats.normalize_batch(values[:, :, :1], torch.tensor([1]), "per_feature")


@pytest.mark.parametrize(("as_list", "preemp", "mag_power"), [(True, 0.97, 2.0), (False, 0.0, 1.0)])
def test_mel_filterbank_extracts_cpu_features(as_list: bool, preemp: float, mag_power: float) -> None:
    preprocessor = runtime.MelFilterBankFeats(
        torch.ones((1, 2, 5)),
        nfft=8,
        window_size=0.25,
        window_stride=0.125,
        preemp=preemp,
        fs=16,
        mag_power=mag_power,
        normalize="none",
        device="cpu",
    )
    rows = [torch.linspace(0, 1, 16), torch.linspace(0, 1, 12)]
    audio: Any = rows if as_list else torch.stack([rows[0], torch.nn.functional.pad(rows[1], (0, 4))])
    supplied_lengths = None if as_list else [16, 12]

    features, feature_lengths = preprocessor.get_feats(audio, supplied_lengths)

    assert features.shape == (2, 2, 9)
    assert feature_lengths.tolist() == [9, 7]
    assert torch.isfinite(features).all()


def test_canary_tokenizer_builds_prompts_and_decodes_text(tmp_path: Path) -> None:
    tokenizer = runtime.CanaryTokenizer(_write_decoder_bundle(tmp_path))

    assert tokenizer.prompt_format == "canary2"
    assert tokenizer.has_country_code
    assert tokenizer.nospeech_id == 99
    assert tokenizer.supports_prompt_language("hi-IN")
    assert not tokenizer.supports_prompt_language("xx")
    assert tokenizer.token_to_id("missing", "hi-IN") == 200
    assert tokenizer.tokens_to_ids("<|startoftranscript|> <|hi-IN|>") == [2, 18]
    assert tokenizer.tokens_to_ids(["<|pnc|>"]) == [7]
    assert tokenizer.ids_to_tokens([200, 201]) == ["▁namaste", "!"]
    assert tokenizer.ids_to_text([2, 200, 201, 3, 200]) == "namaste!"
    assert tokenizer.ids_to_text([200, 999], "hi-IN") == "namaste <unk>"
    assert tokenizer.ids_to_text([300, 301, 302], "ja-JP") == "日, 本"
    assert tokenizer.word_separator("ja-JP") == ""
    assert tokenizer.word_separator("hi-IN") == " "

    prompt = tokenizer.get_prompt_v2(
        pnc=False,
        src_lang="hi-IN",
        tgt_lang="en-US",
        itn=True,
        romanized=True,
        timestamp=True,
        diarize=True,
    )
    assert prompt.endswith("<|nopnc|> <|itn|> <|romanized|> <|timestamp|> <|diarize|>")
    assert "<|hi-IN|> <|en-US|>" in prompt
    config = {
        "task": "transcribe",
        "pnc": False,
        "source_language": "hi-IN",
        "target_language": "hi-IN",
        "itn": False,
        "romanized": False,
        "timestamp": False,
        "diarize": False,
    }
    assert tokenizer.get_prompt_ids_from_cfg(config) == tokenizer.encode(tokenizer.get_prompt_v2(False, "hi-IN"))
    tokenizer.ids_to_text = lambda _ids, _lang=None: "<|pnc|> transcript"  # type: ignore[method-assign]
    assert tokenizer.decode([7, 200]) == " transcript"


def test_canary_tokenizer_validates_v2_languages(tmp_path: Path) -> None:
    tokenizer = runtime.CanaryTokenizer(_write_decoder_bundle(tmp_path, country_codes=False))

    prompt = tokenizer.get_prompt_v2(src_lang="hi-IN", tgt_lang="en-US")
    assert "<|hi|> <|en|>" in prompt
    assert "<|pnc|> <|noitn|> <|noromanized|> <|notimestamp|> <|nodiarize|>" in prompt
    with pytest.raises(ValueError, match="src_lang='xx'"):
        tokenizer.get_prompt_v2(src_lang="xx")
    with pytest.raises(ValueError, match="tgt_lang='xx'"):
        tokenizer.get_prompt_v2(src_lang="hi", tgt_lang="xx")


def test_canary_tokenizer_legacy_prompt_paths(tmp_path: Path) -> None:
    tokenizer = runtime.CanaryTokenizer(_write_decoder_bundle(tmp_path, prompt_format="canary1", country_codes=False))

    assert tokenizer.get_prompt_legacy("asr", False, "hi-IN").endswith("<|transcribe|> <|hi|> <|nopnc|>")
    assert tokenizer.get_prompt_legacy("translate", True, "hi", "en-US").endswith("<|translate|> <|en|> <|pnc|>")
    config = {
        "task": "transcribe",
        "pnc": True,
        "source_language": "hi",
        "target_language": "hi",
    }
    assert tokenizer.get_prompt_ids_from_cfg(config) == tokenizer.encode(tokenizer.get_prompt_legacy(src_lang="hi"))
    tokenizer.set_prompt_format("canary2")
    assert tokenizer.prompt_format == "canary2"
    with pytest.raises(ValueError, match="src_lang='xx'"):
        tokenizer.get_prompt_legacy(src_lang="xx")
    with pytest.raises(ValueError, match="tgt_lang='xx'"):
        tokenizer.get_prompt_legacy("translate", src_lang="hi", tgt_lang="xx")
    with pytest.raises(ValueError, match="task_type='invalid'"):
        tokenizer.get_prompt_legacy("invalid", src_lang="hi")


def _ignore_cuda_device(function: Callable[..., object]) -> Callable[..., object]:
    def wrapped(*args: object, **kwargs: object) -> object:
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return function(*args, **kwargs)

    return wrapped


def test_canary_encoder_loads_engine_and_masks_padding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(json.dumps({"max_batch_size": 4}))
    (encoder_dir / "encoder.plan").write_bytes(b"serialized-engine")
    session = SimpleNamespace()
    fake_session_type = SimpleNamespace(from_serialized_engine=lambda _payload: session)
    monkeypatch.setattr(runtime, "_require_tensorrt_llm", lambda: None)
    monkeypatch.setattr(runtime, "Session", fake_session_type)
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.torch, "arange", _ignore_cuda_device(torch.arange))

    encoder = runtime.CanaryEncoder(tmp_path)
    masked, lengths = encoder.get_masked_emb(
        {
            "outputs": torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]),
            "encoded_lengths": torch.tensor([2, 1]),
        }
    )

    assert encoder.session_conformer is session
    assert encoder.encoder_config == {"max_batch_size": 4}
    assert encoder.device == "cpu"
    torch.testing.assert_close(masked, torch.tensor([[[1.0], [2.0], [0.0]], [[4.0], [0.0], [0.0]]]))
    assert lengths.tolist() == [2, 1]


def test_canary_encoder_infer_executes_static_session(monkeypatch: pytest.MonkeyPatch) -> None:
    output_info = [
        SimpleNamespace(name="encoded_outputs", shape=(2, 3, 1), dtype="float32"),
        SimpleNamespace(name="encoded_lengths", shape=(2,), dtype="int64"),
    ]

    class FakeSession:
        def infer_shapes(self, tensor_info: list[Any]) -> list[Any]:
            assert [item.name for item in tensor_info] == ["audio_signal", "length"]
            return output_info

        def run(self, inputs: dict[str, Any], outputs: dict[str, Any], cuda_stream: int) -> bool:
            assert set(inputs) == {"audio_signal", "length"}
            assert cuda_stream == 123
            outputs["encoded_outputs"].copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3, 1))
            outputs["encoded_lengths"].copy_(torch.tensor([4, 1]))
            return True

    class FakeTensorInfo:
        def __init__(self, name: str, dtype: object, shape: tuple[int, ...]):
            self.name = name
            self.dtype = dtype
            self.shape = shape

    stream = SimpleNamespace(cuda_stream=123, synchronize=lambda: None)
    encoder = object.__new__(runtime.CanaryEncoder)
    encoder.session_conformer = FakeSession()
    monkeypatch.setattr(runtime, "TensorInfo", FakeTensorInfo)
    monkeypatch.setattr(
        runtime,
        "trt",
        SimpleNamespace(DataType=SimpleNamespace(FLOAT=object(), INT64=object())),
    )
    monkeypatch.setattr(
        runtime, "trt_dtype_to_torch", lambda dtype: torch.int64 if dtype == "int64" else torch.float32
    )
    monkeypatch.setattr(runtime.torch, "empty", _ignore_cuda_device(torch.empty))
    monkeypatch.setattr(runtime.torch, "arange", _ignore_cuda_device(torch.arange))

    encoded, lengths = encoder.infer(torch.ones((2, 4, 3)), torch.tensor([4, 3]), stream)

    assert encoded.shape == (2, 3, 1)
    assert lengths.tolist() == [3, 1]
    assert encoded[1, 1:].eq(0).all()


def test_canary_decoder_builds_runner_and_generates_on_mocked_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_decoder_bundle(tmp_path)
    runner_calls: list[dict[str, Any]] = []

    class FakeRunner:
        def generate(self, **kwargs: object) -> dict[str, torch.Tensor]:
            assert len(kwargs["batch_input_ids"]) == 2
            assert kwargs["max_new_tokens"] == 5
            assert kwargs["num_beams"] == 2
            assert [mask.shape for mask in kwargs["cross_attention_masks"]] == [torch.Size([8, 3])] * 2
            return {"output_ids": torch.tensor([[[200, 3]], [[100, 3]]])}

    def from_dir(**kwargs: object) -> FakeRunner:
        runner_calls.append(kwargs)
        return FakeRunner()

    tokenizer = SimpleNamespace(set_prompt_format=lambda _value: None, eos_id=3, pad_id=0)
    monkeypatch.setattr(runtime, "_require_tensorrt_llm", lambda: None)
    monkeypatch.setattr(runtime, "str_dtype_to_torch", lambda _value: torch.float32)
    monkeypatch.setattr(runtime, "KVCacheType", object())
    monkeypatch.setattr(runtime, "ModelRunnerCpp", SimpleNamespace(from_dir=from_dir))
    decoder = runtime.CanaryDecoding(tmp_path, tokenizer, debug_mode=True, device="cpu")
    monkeypatch.setattr(runtime.torch, "tensor", _ignore_cuda_device(torch.tensor))
    monkeypatch.setattr(runtime.torch, "ones", _ignore_cuda_device(torch.ones))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    output = decoder.generate(
        torch.tensor([[2, 18, 7], [2, 17, 7]]),
        torch.ones((2, 3, 4)),
        torch.tensor([3, 3]),
        max_new_tokens=5,
        num_beams=2,
    )

    assert output == [[[200, 3]], [[100, 3]]]
    assert runner_calls[0]["engine_dir"] == str(tmp_path / "decoder")
    assert runner_calls[0]["max_output_len"] == 16
    assert runner_calls[0]["debug_mode"] is True


def test_canary_decoder_rejects_prompts_without_output_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = object.__new__(runtime.CanaryDecoding)
    decoder.dtype = torch.float32
    decoder.max_input_len = 2
    decoder.max_seq_len = 2
    monkeypatch.setattr(runtime.torch, "tensor", _ignore_cuda_device(torch.tensor))

    with pytest.raises(ValueError, match="leaves no output capacity"):
        decoder.generate(torch.ones((1, 2)), torch.ones((1, 2, 2)), torch.tensor([2]), max_new_tokens=1)


def test_canary_runtime_constructs_components_from_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_decoder_bundle(tmp_path)
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(json.dumps({"max_batch_size": 3}))
    preprocessor_dir = tmp_path / "preprocessor"
    preprocessor_dir.mkdir()
    (preprocessor_dir / "config.json").write_text(
        json.dumps(
            {
                "features": 2,
                "n_fft": 8,
                "window_size": 0.25,
                "window_stride": 0.125,
                "window": "hann",
                "sample_rate": 16,
                "preemp": 0.0,
            }
        )
    )
    torch.save(torch.ones((1, 2, 5)), preprocessor_dir / "mel_basis.pt")
    component_calls: list[tuple[str, Any]] = []

    class FakeMapping:
        gpus_per_node = 1

    class FakeComponent:
        def __init__(self, *args: object, **kwargs: object):
            component_calls.append((type(self).__name__, (args, kwargs)))

    class FakePreprocessor(FakeComponent):
        pass

    class FakeTokenizer(FakeComponent):
        pass

    class FakeEncoder(FakeComponent):
        pass

    class FakeDecoder(FakeComponent):
        pass

    monkeypatch.setattr(runtime, "_require_tensorrt_llm", lambda: None)
    monkeypatch.setattr(
        runtime,
        "tensorrt_llm",
        SimpleNamespace(mpi_rank=lambda: 0, Mapping=lambda _world_size, _rank: FakeMapping()),
    )
    monkeypatch.setattr(runtime.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(runtime, "MelFilterBankFeats", FakePreprocessor)
    monkeypatch.setattr(runtime, "CanaryTokenizer", FakeTokenizer)
    monkeypatch.setattr(runtime, "CanaryEncoder", FakeEncoder)
    monkeypatch.setattr(runtime, "CanaryDecoding", FakeDecoder)

    model = runtime.CanaryTRTLLM(
        tmp_path,
        debug_mode=True,
        device="cpu",
        kv_cache_free_gpu_memory_fraction=0.3,
        cross_kv_cache_fraction=0.4,
    )

    assert model.encoder_max_batch_size == 3
    assert model.decoder_max_batch_size == 4
    assert model.max_batch_size == 3
    assert model.num_feats == 2
    assert model.n_fft == 8
    assert [name for name, _ in component_calls] == [
        "FakePreprocessor",
        "FakeTokenizer",
        "FakeEncoder",
        "FakeDecoder",
    ]
    decoder_kwargs = component_calls[-1][1][1]
    assert decoder_kwargs["debug_mode"] is True
    assert decoder_kwargs["kv_cache_free_gpu_memory_fraction"] == 0.3
    assert decoder_kwargs["cross_kv_cache_fraction"] == 0.4


@pytest.mark.parametrize(("max_new_tokens", "expected"), [(None, 24), (5, 5)])
def test_canary_runtime_processes_and_decodes_mocked_batch(
    monkeypatch: pytest.MonkeyPatch,
    max_new_tokens: int | None,
    expected: int,
) -> None:
    calls: dict[str, Any] = {}

    class FakeTokenizer:
        pad_id = 0

        @staticmethod
        def get_prompt_ids_from_cfg(config: dict[str, Any]) -> list[int]:
            return config["ids"]

        @staticmethod
        def decode(ids: list[int]) -> str:
            return "  ...transcript  " if ids[0] == 1 else "!second"

    class FakePreprocessor:
        @staticmethod
        def get_feats(audio: list[Any], lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
            calls["audio"] = (audio, lengths)
            return torch.ones((2, 2, 3)), torch.tensor([3, 2])

    class FakeEncoder:
        @staticmethod
        def infer(mel: torch.Tensor, lengths: torch.Tensor, stream: object) -> tuple[torch.Tensor, torch.Tensor]:
            calls["encoder"] = (mel.shape, lengths.tolist(), stream)
            return torch.ones((2, 2, 4)), torch.tensor([2, 2])

    class FakeDecoder:
        @staticmethod
        def generate(
            prompt_ids: torch.Tensor,
            encoded: torch.Tensor,
            lengths: torch.Tensor,
            **kwargs: object,
        ) -> list[list[list[int]]]:
            calls["decoder"] = (prompt_ids.tolist(), encoded.shape, lengths.tolist(), kwargs)
            return [[[1]], [[2]]]

    model = object.__new__(runtime.CanaryTRTLLM)
    model.device = "cpu"
    model.max_seq_len = 24
    model.tokenizer = FakeTokenizer()
    model.preprocessor = FakePreprocessor()
    model.encoder = FakeEncoder()
    model.decoder = FakeDecoder()
    stream = object()
    monkeypatch.setattr(runtime.torch.cuda, "current_stream", lambda _device: stream)

    texts = model.process_batch(
        [torch.ones(4), torch.ones(3)],
        [4, 3],
        [{"ids": [1, 2]}, {"ids": [3]}],
        num_beams=2,
        max_new_tokens=max_new_tokens,
    )

    assert texts == ["transcript", "second"]
    assert calls["audio"][1] == [4, 3]
    assert calls["decoder"][3] == {"max_new_tokens": expected, "num_beams": 2}
