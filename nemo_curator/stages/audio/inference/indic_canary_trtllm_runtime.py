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

# This module is a faithful, trimmed copy of the upstream TensorRT-LLM
# ``canary-indic`` example runtime; ``Any``-typed tensor args and ``assert``-based
# runtime checks are kept close to upstream on purpose.
# ruff: noqa: ANN401, S101

"""Vendored, trimmed TensorRT-LLM runtime for (Indic) Canary ASR.

This is a slimmed-down copy of the reference ``run_ifb.py`` shipped with the
TensorRT-LLM ``canary-indic`` example. Only the **static (C++ session) batch
inference path** is kept: the CLI, HuggingFace-dataset/WER helpers, the Python
session and the in-flight-batching threading machinery are removed.

The optional ``tensorrt`` / ``tensorrt_llm`` imports are guarded so CPU-only
code can import utility functions and inspect the stage. Constructing an engine
runtime still fails immediately with an actionable dependency error.

The engine directory layout it expects (produced by the example's
``convert_checkpoint.py`` + ``conformer_onnx_trt.py`` + ``trtllm-build``)::

    engine_dir/
      encoder/encoder.plan
      encoder/config.json
      decoder/rank0.engine            # trtllm-build output
      decoder/config.json
      decoder/vocab.json
      preprocessor/config.json
      preprocessor/mel_basis.pt
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from string import punctuation
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

trt: Any = None
tensorrt_llm: Any = None
str_dtype_to_torch: Any = None
trt_dtype_to_torch: Any = None
KVCacheType: Any = None
ModelRunnerCpp: Any = None
Session: Any = None
TensorInfo: Any = None
_TRTLLM_IMPORT_ERROR: ImportError | None = None


def _require_tensorrt_llm() -> None:
    """Load the optional runtime on first engine construction."""
    global KVCacheType, ModelRunnerCpp, Session, TensorInfo  # noqa: PLW0603
    global _TRTLLM_IMPORT_ERROR, str_dtype_to_torch, tensorrt_llm, trt, trt_dtype_to_torch  # noqa: PLW0603

    if tensorrt_llm is not None:
        return
    if _TRTLLM_IMPORT_ERROR is None:
        try:
            import tensorrt as trt_module
            import tensorrt_llm as tensorrt_llm_module
            from tensorrt_llm._utils import str_dtype_to_torch as str_dtype_to_torch_fn
            from tensorrt_llm._utils import trt_dtype_to_torch as trt_dtype_to_torch_fn
            from tensorrt_llm.bindings import KVCacheType as KVCacheTypeImport
            from tensorrt_llm.runtime import ModelRunnerCpp as ModelRunnerCppImport
            from tensorrt_llm.runtime.session import Session as SessionImport
            from tensorrt_llm.runtime.session import TensorInfo as TensorInfoImport
        except ImportError as exc:  # pragma: no cover - runtime-only optional dependency
            _TRTLLM_IMPORT_ERROR = exc
        else:
            trt = trt_module
            tensorrt_llm = tensorrt_llm_module
            str_dtype_to_torch = str_dtype_to_torch_fn
            trt_dtype_to_torch = trt_dtype_to_torch_fn
            KVCacheType = KVCacheTypeImport
            ModelRunnerCpp = ModelRunnerCppImport
            Session = SessionImport
            TensorInfo = TensorInfoImport
            return

    # tensorrt_llm is intentionally NOT in Curator's uv.lock: its
    # cuda-python>=13 / transformers<4.52 / fsspec<=2024.9.0 pins conflict with
    # the audio_cuda12 stack (cudf-cu12, nemo_toolkit), so it cannot be
    # co-resolved. Install it on top of the synced venv instead.
    msg = (
        "tensorrt_llm is required for the Indic Canary ASR runtime but is not "
        "installed. Install it into the Curator venv (Linux x86_64 only), on top "
        "of `uv sync --extra audio_cuda12`:\n"
        "    uv pip install --index-strategy unsafe-best-match \\\n"
        "        --extra-index-url https://pypi.nvidia.com tensorrt_llm==1.2.1"
    )
    raise ImportError(msg) from _TRTLLM_IMPORT_ERROR


CONSTANT = 1e-5
SAMPLE_RATE = 16000
CHUNK_LENGTH = 40
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 640000 samples in a 40-second chunk


def pad_or_trim(array: Any, length: int = N_SAMPLES, *, axis: int = -1) -> Any:
    """Pad or trim an audio array/tensor to ``length`` along ``axis``."""
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array


def unpack_tensors(input_tensors: Any, input_tensor_lengths: Any) -> list:
    return [input_tensors[i, : input_tensor_lengths[i]] for i in range(len(input_tensors))]


def read_config(component: str, engine_dir: Path, mode: str = "decoder") -> OrderedDict:
    config_path = engine_dir / component / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_config: OrderedDict = OrderedDict()
    if mode == "decoder":
        model_config.update(config["pretrained_config"])
        model_config.update(config["build_config"])
    else:
        model_config.update(config)
    return model_config


class MelFilterBankFeats:
    """Log-mel filterbank feature extractor matching the exported preprocessor."""

    def __init__(  # noqa: PLR0913
        self,
        mel_basis: Any,
        nfft: int | None = None,
        window_size: float = 0.025,
        window_stride: float = 0.010,
        window_type: str = "hann",
        preemp: float = 0.97,
        fs: int = 16000,
        mag_power: float = 2.0,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 2**-24,
        normalize: str = "per_feature",
        device: str = "cuda",
    ):
        self.device = device
        self.mel_basis = mel_basis
        self.nfilt = self.mel_basis.shape[1]
        self.normalize = normalize
        if preemp == 0.0:
            preemp = None
        self.preemp = preemp

        self.nfft = (self.mel_basis.shape[2] - 1) * 2 if nfft is None else nfft

        self.log_zero_guard_value = log_zero_guard_value
        self.log_zero_guard_type = log_zero_guard_type
        self.log = log

        assert self.nfft / 2 + 1 == self.mel_basis.shape[2]

        self.window_size = int(fs * window_size)
        self.window_stride = int(fs * window_stride)
        self.mag_power = mag_power
        if window_type == "hann":
            self.window = torch.hann_window(self.window_size, dtype=torch.float, device=self.device)
        elif window_type == "hamming":
            self.window = torch.hamming_window(self.window_size, dtype=torch.float, device=self.device)
        elif window_type == "bartlett":
            self.window = torch.bartlett_window(self.window_size, dtype=torch.float, device=self.device)
        elif window_type == "blackman":
            self.window = torch.blackman_window(self.window_size, dtype=torch.float, device=self.device)
        else:
            self.window = torch.ones(self.window_size, dtype=torch.float, device=self.device)

    def stft(self, audio: Any) -> Any:
        return torch.stft(
            audio,
            n_fft=self.nfft,
            hop_length=self.window_stride,
            win_length=self.window_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
            onesided=True,
        )

    @staticmethod
    def normalize_batch(x: Any, seq_len: Any, normalize_type: str) -> Any:
        if normalize_type == "per_feature":
            batch_size = x.shape[0]
            max_time = x.shape[2]
            if (
                torch.cuda.is_available()
                and not torch.cuda.is_current_stream_capturing()
                and torch.any(seq_len == 1).item()
            ):
                msg = (
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. "
                    "This will result in torch.std() returning nan."
                )
                raise ValueError(msg)
            time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
            valid_mask = time_steps < seq_len.unsqueeze(1)
            x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
            x_mean_denominator = valid_mask.sum(axis=1)
            x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)
            x_std = torch.sqrt(
                torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
                / (x_mean_denominator.unsqueeze(1) - 1.0)
            )
            x_std += CONSTANT
            return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        if normalize_type == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, : seq_len[i].item()].mean()
                x_std[i] = x[i, :, : seq_len[i].item()].std()
            x_std += CONSTANT
            return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        return x

    def get_feat_seq_len(self, seq_len: Any) -> Any:
        pad_amount = self.nfft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.nfft), self.window_stride) + 1
        return seq_len.to(dtype=torch.int64)

    def get_feats(self, audio: Any, seq_len: Any = None) -> tuple[Any, Any]:
        if seq_len is None:
            seq_len = [len(a) for a in audio]
        seq_len = torch.Tensor(seq_len).to(dtype=torch.int32, device=self.device)
        if isinstance(audio, list):
            audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0).to(device=self.device)
        if self.preemp is not None:
            audio = torch.cat((audio[:, 0].unsqueeze(1), audio[:, 1:] - self.preemp * audio[:, :-1]), dim=1)
        spec = torch.view_as_real(self.stft(audio))
        spec = torch.sqrt(spec.pow(2).sum(-1))
        if self.mag_power != 1.0:
            spec = spec.pow(self.mag_power)
        spec = torch.matmul(self.mel_basis.to(spec.dtype), spec)
        spec = torch.log(spec + self.log_zero_guard_value)
        seq_len = self.get_feat_seq_len(seq_len)
        spec = self.normalize_batch(spec, seq_len, normalize_type=self.normalize)
        return spec, seq_len


class CanaryTokenizer:
    """Canary vocabulary + Canary1/Canary2 control-prompt builder."""

    def __init__(self, engine_dir: Path, prompt_format: str = "canary1"):  # noqa: C901, PLR0912
        vocab_file = os.path.join(engine_dir, "decoder/vocab.json")
        decoder_config = read_config("decoder", engine_dir)
        self.prompt_format = decoder_config.get("prompt_format", prompt_format)
        self.blank = "▁"
        self.has_country_code = False

        with open(vocab_file) as jfp:
            vocab = json.load(jfp)
        self.token_id_offset = vocab["offsets"]
        self.langs = list(vocab["tokens"])
        self.__id_to_token__ = {lang: {} for lang in vocab["tokens"]}

        self.spl_tokens = self.__id_to_token__["spl_tokens"]

        for lang in vocab["tokens"]:
            self.__id_to_token__[lang] = {int(k): v for k, v in vocab["tokens"][lang].items()}
        self.__token_to_id__ = {}
        for lang in self.__id_to_token__:
            self.__token_to_id__[lang] = {v: k for k, v in self.__id_to_token__[lang].items()}

        for lang in self.langs:
            if lang == "spl_tokens":
                continue
            lang_token = f"<|{lang.split('-')[0]}|>"
            if "-" in lang:
                lang_country_token = f"<|{lang}|>"
                self.has_country_code = True
            else:
                lang_country_token = lang_token
            if lang_country_token in self.__token_to_id__["spl_tokens"]:
                continue
            if (
                lang_country_token not in self.__token_to_id__["spl_tokens"]
                and lang_token in self.__token_to_id__["spl_tokens"]
            ):
                self.__token_to_id__["spl_tokens"][lang_country_token] = self.__token_to_id__["spl_tokens"][lang_token]
                if lang_country_token != lang_token:
                    self.langs.append(lang.split("-")[0])

        dpl = "en-US" if self.has_country_code else "en"
        if self.prompt_format == "canary2":
            self.default_prompt = (
                f"<|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|{dpl}|> <|{dpl}|> "
                f"<|nopnc|> <|noitn|> <|noromanized|> <|notimestamp|> <|nodiarize|>"
            )
        else:
            self.default_prompt = f"<|startoftranscript|> <|{dpl}|> <|transcribe|> <|{dpl}|> <|pnc|>"

        self.id_to_token = {}
        for lang in self.__id_to_token__:
            self.id_to_token.update(self.__id_to_token__[lang])
        self.task = {
            "transcribe": "<|transcribe|>",
            "translate": "<|translate|>",
            "asr": "<|transcribe|>",
            "ast": "<|translate|>",
        }

        self.bos_id = vocab.get("bos_id", self.spl_tokens.get("<|startoftranscript|>"))
        self.eos_id = vocab.get("eos_id", self.spl_tokens.get("<|endoftext|>"))
        if "nospeech_id" in vocab:
            self.nospeech_id = vocab["nospeech_id"]
        self.pad_id = vocab["pad_id"]
        self.blank_id = self.__token_to_id__["spl_tokens"][self.blank]

    def set_prompt_format(self, prompt_format: str = "canary1") -> None:
        self.prompt_format = prompt_format

    @staticmethod
    def word_separator(lang: str) -> str:
        if lang in ["ja-JP", "ko-KR", "zh-CN", "th-TH", "km-KH", "my-MM", "lo-LA"]:
            return ""
        return " "

    def ids_to_tokens(self, token_ids: list) -> list:
        return [self.id_to_token[k] for k in token_ids]

    def token_to_id(self, token: str, lang: str = "spl_tokens") -> int:
        return self.__token_to_id__[lang].get(token, self.token_id_offset[lang])

    def supports_prompt_language(self, lang: str) -> bool:
        """Whether the special-token vocabulary can encode a language."""
        return f"<|{lang}|>" in self.__token_to_id__["spl_tokens"]

    def tokens_to_ids(self, tokens: list | str, lang: str = "spl_tokens") -> list:
        if isinstance(tokens, str):
            tokens = tokens.split(" ")
        return [self.token_to_id(k, lang) for k in tokens]

    def ids_to_text(self, ids: list, lang: str | None = None) -> str:
        max_repeat = 10
        clean_ids = []
        prev_id = 0
        id_count = 0
        for i in ids:
            if prev_id == i:
                id_count += 1
                if id_count >= max_repeat:
                    continue
            else:
                id_count = 0
                prev_id = i
            if i == self.eos_id:
                break
            if i not in self.__id_to_token__["spl_tokens"]:
                clean_ids.append(i)

        if lang is None:
            return "".join(self.ids_to_tokens(clean_ids)).replace("▁", " ").strip()
        ws = self.word_separator(lang)
        tokens = [self.__id_to_token__[lang].get(k, " <unk> ").replace("▁", ws) for k in clean_ids]
        if ws == "":
            return re.sub(r"(?<=[.,;:])(?=[^\s])", r" ", "".join(tokens).strip())
        return "".join(tokens).strip()

    def get_prompt_v2(  # noqa: PLR0913
        self,
        pnc: bool = True,
        src_lang: str = "en",
        tgt_lang: str | None = None,
        itn: bool = False,
        romanized: bool = False,
        timestamp: bool = False,
        diarize: bool = False,
    ) -> str:
        prompt = "<|startofcontext|> <|startoftranscript|> <|emo:undefined|>"
        if not self.has_country_code and "-" in src_lang:
            src_lang = src_lang.split("-")[0]
        if not self.supports_prompt_language(src_lang):
            msg = f"Invalid language {src_lang=} specified"
            raise ValueError(msg)
        pnc_t = "<|pnc|>" if pnc else "<|nopnc|>"
        itn_t = "<|itn|>" if itn else "<|noitn|>"
        rom_t = "<|romanized|>" if romanized else "<|noromanized|>"
        diar_t = "<|diarize|>" if diarize else "<|nodiarize|>"
        ts_t = "<|timestamp|>" if timestamp else "<|notimestamp|>"
        if tgt_lang is None:
            tgt_lang = src_lang
        if not self.has_country_code and "-" in tgt_lang:
            tgt_lang = tgt_lang.split("-")[0]
        if not self.supports_prompt_language(tgt_lang):
            msg = f"Invalid language {tgt_lang=} specified"
            raise ValueError(msg)
        prompt += f" <|{src_lang}|> <|{tgt_lang}|> {pnc_t} {itn_t} {rom_t} {ts_t} {diar_t}"
        return prompt

    def get_prompt_legacy(
        self, task_type: str = "transcribe", pnc: bool = True, src_lang: str = "en", tgt_lang: str | None = None
    ) -> str:
        prompt = "<|startoftranscript|>"
        if src_lang not in self.langs and "-" in src_lang and src_lang.split("-")[0] in self.langs:
            src_lang = src_lang.split("-")[0]
        if src_lang not in self.langs:
            msg = f"Invalid language {src_lang=} specified"
            raise ValueError(msg)
        prompt += f" <|{src_lang}|>"
        pnc_t = "<|pnc|>" if pnc else "<|nopnc|>"
        if task_type in ("translate", "ast"):
            if tgt_lang is None:
                tgt_lang = src_lang
            if tgt_lang not in self.langs and "-" in tgt_lang and tgt_lang.split("-")[0] in self.langs:
                tgt_lang = tgt_lang.split("-")[0]
            if tgt_lang not in self.langs:
                msg = f"Invalid language {tgt_lang=} specified"
                raise ValueError(msg)
            prompt += f" {self.task[task_type]} <|{tgt_lang}|> {pnc_t}"
        elif task_type in ("transcribe", "asr"):
            prompt += f" {self.task[task_type]} <|{src_lang}|> {pnc_t}"
        else:
            msg = f"Invalid task {task_type=} specified"
            raise ValueError(msg)
        return prompt

    def get_prompt_ids_from_cfg(self, cfg: dict) -> list:
        if self.prompt_format == "canary2":
            return self.tokens_to_ids(
                self.get_prompt_v2(
                    cfg["pnc"],
                    cfg["source_language"],
                    cfg["target_language"],
                    cfg["itn"],
                    cfg.get("romanized", False),
                    cfg["timestamp"],
                    cfg["diarize"],
                )
            )
        return self.tokens_to_ids(
            self.get_prompt_legacy(cfg["task"], cfg["pnc"], cfg["source_language"], cfg["target_language"])
        )

    def encode(self, prompt: str) -> list:
        return self.tokens_to_ids(prompt.split())

    def decode(self, ids: list, lang: str | None = None) -> str:
        text = self.ids_to_text(ids, lang)
        return re.sub(r"<\|.*?\|>", "", text)


class CanaryEncoder:
    """FastConformer encoder TensorRT engine (``encoder/encoder.plan``)."""

    def __init__(self, engine_dir: Path):
        _require_tensorrt_llm()
        engine_path = os.path.join(engine_dir, "encoder/encoder.plan")
        with open(os.path.join(engine_dir, "encoder/config.json")) as f:
            self.encoder_config = json.load(f)
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()
        self.session_conformer = Session.from_serialized_engine(engine_buffer)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_masked_emb(enc_outputs: dict) -> tuple[Any, Any]:
        enc_emb = enc_outputs.get("encoded_outputs", enc_outputs.get("outputs"))
        enc_len = enc_outputs["encoded_lengths"]
        batch_size = enc_len.shape[0]
        max_length = enc_emb.shape[1]
        mask = torch.arange(max_length, device="cuda").unsqueeze(0).expand(batch_size, max_length) < enc_len.unsqueeze(
            1
        )
        enc_mask = torch.where(mask.unsqueeze(2), enc_emb, 0.0)
        return enc_mask, enc_len

    def infer(self, audio_signal: Any, lengths: Any, stream: Any) -> tuple[Any, Any]:
        audio_inputs = {"audio_signal": audio_signal, "length": lengths}
        outputs_info = self.session_conformer.infer_shapes(
            [
                TensorInfo("audio_signal", trt.DataType.FLOAT, audio_signal.shape),
                TensorInfo("length", trt.DataType.INT64, lengths.shape),
            ]
        )
        enc_outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda:0")
            for t in outputs_info
        }
        is_ok = self.session_conformer.run(audio_inputs, enc_outputs, stream.cuda_stream)
        assert is_ok, "Runtime execution failed for Conformer Encoder session"
        stream.synchronize()
        enc_mask, emb_len = self.get_masked_emb(enc_outputs)
        emb_len = torch.clip(emb_len, max=enc_mask.shape[1])
        return enc_mask, emb_len


class CanaryDecoding:
    """Transformer decoder TensorRT-LLM engine (C++ static-batch session)."""

    def __init__(  # noqa: PLR0913
        self,
        engine_dir: Path,
        tokenizer: CanaryTokenizer,
        debug_mode: bool = False,
        device: str = "cuda:0",
        kv_cache_free_gpu_memory_fraction: float = 0.2,
        cross_kv_cache_fraction: float = 0.2,
    ):
        _require_tensorrt_llm()
        self.tokenizer = tokenizer
        self.decoder_config = read_config("decoder", engine_dir)
        self.prompt_format = self.decoder_config["prompt_format"]
        self.tokenizer.set_prompt_format(self.prompt_format)
        self.dtype = str_dtype_to_torch(self.decoder_config["dtype"])
        self.max_seq_len = self.decoder_config["max_seq_len"]
        self.max_input_len = self.decoder_config["max_input_len"]
        self.device = device
        self.kv_cache_free_gpu_memory_fraction = kv_cache_free_gpu_memory_fraction
        self.cross_kv_cache_fraction = cross_kv_cache_fraction
        self.decoder_generation_session = self._get_cpp_session(engine_dir, debug_mode)

    def _get_cpp_session(self, engine_dir: Path, debug_mode: bool = False) -> Any:
        runner_kwargs = {
            "engine_dir": os.path.join(engine_dir, "decoder"),
            "is_enc_dec": False,
            "max_batch_size": self.decoder_config["max_batch_size"],
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_seq_len - self.max_input_len,
            "max_beam_width": self.decoder_config["max_beam_width"],
            "debug_mode": debug_mode,
            "kv_cache_free_gpu_memory_fraction": self.kv_cache_free_gpu_memory_fraction,
            "cross_kv_cache_fraction": self.cross_kv_cache_fraction,
        }
        # KVCacheType is imported for parity with the reference runner's config path.
        _ = KVCacheType
        return ModelRunnerCpp.from_dir(**runner_kwargs)

    def generate(
        self,
        decoder_input_ids: Any,
        encoder_outputs: Any,
        encoder_input_lengths: Any,
        max_new_tokens: int,
        num_beams: int = 1,
    ) -> list:
        encoder_outputs = encoder_outputs.to(dtype=self.dtype)
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(batch_size)], dtype=torch.int32, device="cuda"
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()
        assert decoder_max_input_length <= self.max_input_len, (
            f"Decoder input length {decoder_max_input_length} exceeds max input length {self.max_input_len}"
        )
        available_output_tokens = self.max_seq_len - decoder_max_input_length
        if available_output_tokens <= 0:
            msg = (
                f"Decoder prompt length {decoder_max_input_length} leaves no output capacity "
                f"in max_seq_len {self.max_seq_len}"
            )
            raise ValueError(msg)
        max_new_tokens = min(max_new_tokens, available_output_tokens)

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        cross_attention_masks = [
            torch.ones(
                [decoder_input_lengths[i] + max_new_tokens, encoder_input_lengths[i]],
                dtype=torch.bool,
                device="cuda",
            )
            for i in range(batch_size)
        ]
        decoder_input_ids = unpack_tensors(decoder_input_ids, decoder_input_lengths)
        encoder_outputs = unpack_tensors(encoder_outputs, encoder_input_lengths)
        out = self.decoder_generation_session.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_features=encoder_outputs,
            encoder_output_lengths=encoder_input_lengths,
            cross_attention_masks=cross_attention_masks,
            max_new_tokens=max_new_tokens,
            end_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            num_beams=num_beams,
            output_sequence_lengths=True,
            return_dict=True,
        )
        return out["output_ids"].cpu().numpy().tolist()


class CanaryTRTLLM:
    """End-to-end static-batch Canary TRT-LLM pipeline: mel → encoder → decoder."""

    def __init__(
        self,
        engine_dir: str | Path,
        debug_mode: bool = False,
        device: str = "cuda:0",
        kv_cache_free_gpu_memory_fraction: float = 0.2,
        cross_kv_cache_fraction: float = 0.2,
    ):
        _require_tensorrt_llm()
        self.device = device
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        with open(os.path.join(engine_dir, "preprocessor/config.json")) as f:
            preprocessor_config = json.load(f)

        self.encoder_config = read_config("encoder", engine_dir, mode="encoder")
        self.decoder_config = read_config("decoder", engine_dir)
        self.max_seq_len = self.decoder_config["max_seq_len"]
        self.max_input_len = self.decoder_config["max_input_len"]
        self.max_batch_size = self.encoder_config["max_batch_size"]

        self.num_feats = preprocessor_config["features"]
        self.n_fft = preprocessor_config["n_fft"]
        mel_basis_file = engine_dir / "preprocessor/mel_basis.pt"
        self.mel_basis = torch.load(mel_basis_file, weights_only=True, map_location=torch.device(self.device))

        self.preprocessor = MelFilterBankFeats(
            self.mel_basis,
            window_size=preprocessor_config.get("window_size", 0.025),
            window_stride=preprocessor_config.get("window_stride", 0.010),
            window_type=preprocessor_config.get("window", "hann"),
            fs=preprocessor_config.get("sample_rate", 16000),
            preemp=preprocessor_config.get("preemp", False),
            device=self.device,
        )
        self.tokenizer = CanaryTokenizer(engine_dir)
        self.encoder = CanaryEncoder(engine_dir)
        self.decoder = CanaryDecoding(
            engine_dir,
            tokenizer=self.tokenizer,
            debug_mode=debug_mode,
            device=self.device,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=cross_kv_cache_fraction,
        )

    def process_batch(
        self,
        audio: list,
        audio_input_lengths: list,
        prompts_cfg: list[dict],
        num_beams: int = 1,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        batch_size = len(audio_input_lengths)
        prompt_ids = [torch.tensor(self.tokenizer.get_prompt_ids_from_cfg(cfg)) for cfg in prompts_cfg]
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_ids, batch_first=True, padding_value=self.tokenizer.pad_id
        ).to(self.device)

        stream = torch.cuda.current_stream("cuda")
        if max_new_tokens is None:
            max_new_tokens = self.max_seq_len

        mel, mel_input_lengths = self.preprocessor.get_feats(audio, audio_input_lengths)
        encoder_output, encoder_output_lengths = self.encoder.infer(mel, mel_input_lengths, stream)
        output_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            encoder_output_lengths,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        _ = batch_size
        return self.decode(output_ids)

    def decode(self, output_ids: list) -> list[str]:
        texts = []
        for row in output_ids:
            text = self.tokenizer.decode(row[0]).strip()
            texts.append(text.lstrip(punctuation))
        return texts
