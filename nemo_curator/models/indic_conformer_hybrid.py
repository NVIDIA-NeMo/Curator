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

# ruff: noqa: ANN401, I001, N806, PLR0913, PLR1714, PLW0603

"""AI4Bharat IndicConformer *hybrid* (CTC+RNNT) per-language ``.nemo`` ASR.

This adapter loads the per-language
``ai4bharat/indicconformer_stt_<lang>_hybrid_ctc_rnnt_large`` ``.nemo``
checkpoints and runs waveform-to-text inference behind the shared ``ASRStage``.

These checkpoints were trained with AI4Bharat's NeMo fork
(https://github.com/AI4Bharat/NeMo, ``nemo-v2`` branch), which adds a *multi-softmax*
head to the standard NeMo ASR models: one shared Conformer encoder + shared RNNT
prediction network, and a **per-language output head** selected at inference time by
``language_id``.

The stock ``nemo-toolkit`` (2.7.x) installed in this container does NOT know those
config keys, so ``ASRModel.restore_from`` fails out of the box:

    * ``RNNTDecoder(multisoftmax=...)``      -> unexpected kwarg
    * ``RNNTJoint(multilingual=..., language_keys=...)`` -> unexpected kwargs +
      a per-language ``ModuleDict`` final layer instead of a single ``Linear``
    * ``ConvASRDecoder(multisoftmax=...)``   -> unexpected kwarg

Rather than installing the fork (which is pinned to NeMo 1.23 and would break the
rest of the pipeline), :func:`_apply_multisoftmax_patches` **monkeypatches just those
three module classes** on top of the installed NeMo so the checkpoint loads, and the
model then runs a **compact greedy CTC / RNNT decode** that mirrors the fork's decode
semantics (per-language blank index ``V/num_langs``, per-language joint head, local-id
feedback to the prediction network). Decoding maps the per-language local token ids
back to text through the model's own ``AggregateTokenizer`` (which already ships the
per-language tokenizers and offset tables in 2.7.x).

The patches are idempotent and additive: when ``multisoftmax`` / ``multilingual`` are
absent (a normal NeMo model), every patched path falls back to the original behaviour,
so importing this module does not change ordinary NeMo usage.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.audio_chunking import (
    has_audio_longer_than,
    merge_chunk_texts,
    split_waveforms,
)

_TARGET_SR = 16_000
_MAX_CHUNK_DURATION_SEC = 40.0

# Set once ``_apply_multisoftmax_patches`` has run.
_PATCHED = False
# Scratch space used to pass ``multilingual`` / ``language_keys`` from the patched
# ``RNNTJoint.__init__`` into the patched ``_joint_net_modules`` that the original
# ``__init__`` body calls before we get a chance to set the instance attributes.
# Safe because checkpoint restore instantiates modules single-threaded.
_JOINT_CTX: dict[str, Any] = {}

# The 22 languages carried by every IndicConformer hybrid checkpoint's multi-softmax head.
INDIC_CONFORMER_HYBRID_LANGS: frozenset[str] = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)


class _LanguageRNNTDecoder:
    """Route a per-language blank to the aggregate predictor's SOS token."""

    def __init__(self, decoder: Any, blank_index: int):
        self._decoder = decoder
        self._blank_index = blank_index

    def __getattr__(self, name: str) -> Any:
        return getattr(self._decoder, name)

    def predict(self, y: Any = None, state: Any = None, **kwargs: Any) -> Any:
        if y is not None:
            y = y.masked_fill(y == self._blank_index, self._decoder.blank_idx)
        return self._decoder.predict(y, state=state, **kwargs)


class _LanguageRNNTJoint:
    """Bind the multilingual joint network to one language head."""

    def __init__(self, joint: Any, language: str, num_classes_with_blank: int):
        self._joint = joint
        self._language = language
        self._num_classes_with_blank = num_classes_with_blank

    def __getattr__(self, name: str) -> Any:
        return getattr(self._joint, name)

    @property
    def num_classes_with_blank(self) -> int:
        return self._num_classes_with_blank

    def project_encoder(self, encoder_output: Any) -> Any:
        project = getattr(self._joint, "project_encoder", self._joint.enc)
        return project(encoder_output)

    def project_prednet(self, prednet_output: Any) -> Any:
        project = getattr(self._joint, "project_prednet", self._joint.pred)
        return project(prednet_output)

    def joint_after_projection(self, f: Any, g: Any) -> Any:
        language_ids = [self._language] * f.shape[0]
        return self._joint.joint_after_projection(f, g, language_ids=language_ids)


def _apply_multisoftmax_patches() -> None:  # noqa: C901, PLR0915
    """Idempotently patch ConvASRDecoder / RNNTJoint / RNNTDecoder for multi-softmax."""
    global _PATCHED
    if _PATCHED:
        return

    import torch
    from nemo.collections.asr.modules import conv_asr, rnnt
    from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin

    # ------------------------------------------------------------------
    # Tokenizer routing: the fork tags the aggregate tokenizer ``type:
    # multilingual``; stock NeMo only routes ``agg`` to the aggregate path and
    # sends everything else to the monolingual path (which needs a top-level
    # ``dir`` key and fails). Treat ``multilingual`` as an aggregate tokenizer.
    # ------------------------------------------------------------------
    _orig_setup_tokenizer = ASRBPEMixin._setup_tokenizer

    def _setup_tokenizer(self: Any, tokenizer_cfg: Any) -> None:
        ttype = tokenizer_cfg.get("type")
        if ttype is not None and str(ttype).lower() == "multilingual":
            self._setup_aggregate_tokenizer(tokenizer_cfg)
            # Stock NeMo keys its aggregate-tokenizer handling (vocabulary as a
            # list, CTC vocab wiring) off ``tokenizer_type == "agg"``; the fork
            # used "multilingual" for the same thing. Normalise so the model's
            # own __init__ takes the aggregate branch.
            self.tokenizer_type = "agg"
            self._derive_tokenizer_properties()
            return
        _orig_setup_tokenizer(self, tokenizer_cfg)

    ASRBPEMixin._setup_tokenizer = _setup_tokenizer

    # ------------------------------------------------------------------
    # ConvASRDecoder (auxiliary CTC head)
    # ------------------------------------------------------------------
    _ConvASRDecoder = conv_asr.ConvASRDecoder
    _conv_orig_init = _ConvASRDecoder.__init__

    def _conv_init(
        self: Any, *args: Any, multisoftmax: bool = False, language_masks: Any = None, **kwargs: Any
    ) -> None:
        # Structure is identical to stock NeMo; the extra kwargs only gate the
        # per-language masking applied in forward(). Drop them before delegating.
        _conv_orig_init(self, *args, **kwargs)
        self.multisoftmax = multisoftmax
        self.language_masks = language_masks

    def _conv_forward(self: Any, encoder_output: Any, language_ids: Any = None) -> Any:
        # Mirrors AI4Bharat fork conv_asr.ConvASRDecoder.forward (no @typecheck so
        # language_ids is accepted). decoder_layers -> [B, T, C]; optional mask to
        # the language's contiguous token block + blank, then log_softmax.
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)

        if self.temperature != 1.0:
            decoder_output = self.decoder_layers(encoder_output).transpose(1, 2) / self.temperature
        else:
            decoder_output = self.decoder_layers(encoder_output).transpose(1, 2)

        if language_ids is not None:
            sample_mask = torch.tensor([self.language_masks[lang] for lang in language_ids], dtype=torch.bool)
            mask = sample_mask.unsqueeze(1).repeat(1, decoder_output.shape[1], 1).to(decoder_output.device)
            decoder_output = torch.masked_select(decoder_output, mask).view(
                decoder_output.shape[0], decoder_output.shape[1], -1
            )
        return torch.nn.functional.log_softmax(decoder_output, dim=-1)

    _ConvASRDecoder.__init__ = _conv_init
    _ConvASRDecoder.forward = _conv_forward

    # ------------------------------------------------------------------
    # RNNTDecoder (shared prediction network) — only absorbs the extra kwargs.
    # ------------------------------------------------------------------
    _RNNTDecoder = rnnt.RNNTDecoder
    _dec_orig_init = _RNNTDecoder.__init__

    def _dec_init(
        self: Any, *args: Any, multisoftmax: bool = False, language_masks: Any = None, **kwargs: Any
    ) -> None:
        _dec_orig_init(self, *args, **kwargs)
        self.multisoftmax = multisoftmax
        self.language_masks = language_masks

    _RNNTDecoder.__init__ = _dec_init

    # ------------------------------------------------------------------
    # RNNTJoint — per-language ModuleDict final layer + language routing.
    # ------------------------------------------------------------------
    _RNNTJoint = rnnt.RNNTJoint
    _joint_orig_init = _RNNTJoint.__init__
    _joint_orig_jnm = _RNNTJoint._joint_net_modules

    def _joint_init(
        self: Any,
        *args: Any,
        multilingual: bool = False,
        language_keys: Any = None,
        language_masks: Any = None,
        token_id_offsets: Any = None,
        offset_token_ids_by_token_id: Any = None,
        **kwargs: Any,
    ) -> None:
        # _joint_net_modules runs *inside* the original __init__ before we can set
        # instance attrs, so stash what it needs in module-level scratch.
        _JOINT_CTX["multilingual"] = multilingual
        _JOINT_CTX["language_keys"] = list(language_keys) if language_keys is not None else None
        try:
            _joint_orig_init(self, *args, **kwargs)
        finally:
            _JOINT_CTX.clear()
        self.multilingual = multilingual
        self.language_keys = list(language_keys) if language_keys is not None else None
        self.language_masks = language_masks
        self.token_id_offsets = token_id_offsets
        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id

    def _joint_net_modules(
        self: Any,
        num_classes: int,
        pred_n_hidden: int,
        enc_n_hidden: int,
        joint_n_hidden: int,
        activation: str,
        dropout: float,
    ) -> Any:
        if not _JOINT_CTX.get("multilingual"):
            return _joint_orig_jnm(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout)
        language_keys = _JOINT_CTX["language_keys"]
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)
        act = activation.lower()
        if act == "relu":
            act_mod: Any = torch.nn.ReLU(inplace=True)
        elif act == "sigmoid":
            act_mod = torch.nn.Sigmoid()
        elif act == "tanh":
            act_mod = torch.nn.Tanh()
        else:
            msg = f"Unsupported activation for joint step: {activation}"
            raise ValueError(msg)
        # Per-language head: V/num_langs (+1 for blank). self._vocab_size is the
        # full aggregate vocab; it is set before this method is called.
        per_lang = self._vocab_size // len(language_keys) + 1
        final_layer = torch.nn.ModuleDict({lang: torch.nn.Linear(joint_n_hidden, per_lang) for lang in language_keys})
        logger.info(f"Multilingual RNNT joint: {len(language_keys)} heads x {per_lang} classes")
        layers = [act_mod] + ([torch.nn.Dropout(p=dropout)] if dropout else []) + [final_layer]
        return pred, enc, torch.nn.Sequential(*layers)

    def _joint_after_projection(self: Any, f: Any, g: Any, language_ids: Any = None) -> Any:
        # Mirrors fork RNNTJoint.joint_after_projection with language routing.
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)
        inp = f + g  # (B, T, U, H)
        del f, g
        if self.is_adapter_available():
            inp = self.forward_enabled_adapters(inp)

        if language_ids is not None:
            for module in self.joint_net[:-1]:
                inp = module(inp)
            if len(set(language_ids)) == 1:
                res = self.joint_net[-1][language_ids[0]](inp)
            else:
                res = torch.stack(
                    [self.joint_net[-1][lang](single) for single, lang in zip(inp, language_ids, strict=True)]
                )
        else:
            res = self.joint_net(inp)
        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()
        if self.log_softmax is None:
            if not res.is_cuda:
                res = (
                    (res / self.temperature).log_softmax(dim=-1)
                    if self.temperature != 1.0
                    else res.log_softmax(dim=-1)
                )
        elif self.log_softmax:
            res = (res / self.temperature).log_softmax(dim=-1) if self.temperature != 1.0 else res.log_softmax(dim=-1)
        return res

    _RNNTJoint.__init__ = _joint_init
    _RNNTJoint._joint_net_modules = _joint_net_modules
    _RNNTJoint.joint_after_projection = _joint_after_projection

    _PATCHED = True
    logger.info("Applied AI4Bharat multi-softmax patches to NeMo ConvASRDecoder/RNNTDecoder/RNNTJoint")


class IndicConformerHybridASR:
    """AI4Bharat IndicConformer hybrid adapter for #1967's generic ASR stage."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        decode_mode: Literal["ctc", "rnnt"] = "rnnt",
        *,
        max_symbols_per_step: int = 10,
        rnnt_precision: Literal["fp32", "fp16", "bf16"] = "fp32",
        empty_audio_marks_skip: bool = True,
    ):
        if not model_id:
            msg = "IndicConformerHybridASR.model_id must be non-empty"
            raise ValueError(msg)
        if revision is not None:
            msg = "IndicConformerHybridASR does not support revision pinning"
            raise ValueError(msg)
        if decode_mode not in {"ctc", "rnnt"}:
            msg = f"Unsupported IndicConformer decode mode: {decode_mode!r}"
            raise ValueError(msg)
        if rnnt_precision not in {"fp32", "fp16", "bf16"}:
            msg = f"Unsupported IndicConformer RNNT precision: {rnnt_precision!r}"
            raise ValueError(msg)
        if max_symbols_per_step < 1:
            msg = "max_symbols_per_step must be at least 1"
            raise ValueError(msg)
        self.model_id = model_id
        self.revision = revision
        self.decode_mode = decode_mode
        self.max_symbols_per_step = max_symbols_per_step
        self.rnnt_precision = rnnt_precision
        self.empty_audio_marks_skip = empty_audio_marks_skip
        self._model: Any = None
        self._device: Any = None
        self._num_langs: int = 0
        self._per_lang_classes: int = 0  # V / num_langs (blank index within a head)
        self._rnnt_decoders: dict[str, Any] = {}
        self._chunk_duration_sec: float | None = _MAX_CHUNK_DURATION_SEC

    @staticmethod
    def _offline() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "0").strip().lower() not in {"0", "", "false", "no"}

    @classmethod
    def _resolve_nemo_path(cls, model_id: str) -> str:
        """Resolve a local checkpoint or an already-cached Hugging Face repo ID."""
        if model_id.endswith(".nemo"):
            if not Path(model_id).is_file():
                msg = f"Local NeMo checkpoint not found: {model_id}"
                raise FileNotFoundError(msg)
            return model_id
        if os.path.exists(model_id):
            return model_id

        from huggingface_hub import snapshot_download

        try:
            snapshot_dir = Path(snapshot_download(model_id, local_files_only=True))
            cached = sorted(snapshot_dir.rglob("*.nemo"))
            if cached:
                return str(cached[0])
        except Exception:  # noqa: BLE001, S110
            pass

        if cls._offline():
            msg = f"No cached .nemo file found for HuggingFace repo '{model_id}' while HF_HUB_OFFLINE is set"
            raise FileNotFoundError(msg)
        msg = (
            f"No cached .nemo file found for HuggingFace repo '{model_id}'; "
            "run download_weights_on_node() during node setup before loading the worker model"
        )
        raise FileNotFoundError(msg)

    def download_weights_on_node(self) -> None:
        """Resolve the configured checkpoint into the node-local cache without loading it."""
        if self.model_id.endswith(".nemo"):
            if not Path(self.model_id).is_file():
                msg = f"Local NeMo checkpoint not found: {self.model_id}"
                raise FileNotFoundError(msg)
            return
        if os.path.exists(self.model_id):
            return
        if self._offline():
            self._resolve_nemo_path(self.model_id)
            return

        from huggingface_hub import HfApi, hf_hub_download

        files = [f for f in HfApi().list_repo_files(self.model_id) if f.endswith(".nemo")]
        if not files:
            msg = f"No .nemo file found in HuggingFace repo '{self.model_id}'"
            raise RuntimeError(msg)
        hf_hub_download(self.model_id, files[0])

    def load_model(self, *, num_gpus: int) -> None:
        if self._model is not None:
            return
        if num_gpus < 0:
            msg = "num_gpus must be non-negative"
            raise ValueError(msg)

        import torch
        import nemo.collections.asr as nemo_asr

        _apply_multisoftmax_patches()
        self._device = torch.device("cuda" if num_gpus else "cpu")
        nemo_path = self._resolve_nemo_path(self.model_id)
        logger.info(f"Loading IndicConformer hybrid model={nemo_path} device={self._device}")

        self._model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=self._device)
        self._model.to(self._device)
        self._model.eval()
        self._chunk_duration_sec = _MAX_CHUNK_DURATION_SEC
        self._configure_rnnt_precision()

        tok = self._model.tokenizer
        if not hasattr(tok, "langs_by_token_id"):
            msg = "Loaded model does not use an AggregateTokenizer; this wrapper expects the multilingual checkpoint."
            raise RuntimeError(msg)
        self._num_langs = len(tok.tokenizers_dict)
        self._per_lang_classes = self._model.joint._vocab_size // self._num_langs

        # Build the per-language CTC masks (token belongs to lang) + blank, then
        # hand them to the (patched) CTC decoder for masked decoding.
        masks: dict[str, list[bool]] = {}
        for lang in tok.tokenizers_dict:
            m = [tok.langs_by_token_id[i] == lang for i in range(len(tok.langs_by_token_id))]
            m.append(True)  # blank
            masks[lang] = m
        self._model.ctc_decoder.language_masks = masks
        logger.info(f"IndicConformer hybrid ready: {self._num_langs} langs, {self._per_lang_classes} tokens/lang")

    def unload_model(self) -> None:
        for decoder in self._rnnt_decoders.values():
            decoding_computer = getattr(decoder, "decoding_computer", None)
            reset_cuda_graphs = getattr(decoding_computer, "reset_cuda_graphs_state", None)
            if callable(reset_cuda_graphs):
                reset_cuda_graphs()
        self._rnnt_decoders.clear()
        self._model = None
        self._device = None
        self._chunk_duration_sec = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _rnnt_dtype(self) -> Any:
        import torch

        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.rnnt_precision]

    def _configure_rnnt_precision(self) -> None:
        if self.rnnt_precision == "fp32":
            return
        import torch

        if self._device.type != "cuda":
            msg = f"IndicConformer {self.rnnt_precision.upper()} RNNT inference requires CUDA"
            raise RuntimeError(msg)
        if self.rnnt_precision == "bf16" and not torch.cuda.is_bf16_supported():
            msg = "IndicConformer BF16 RNNT inference is not supported by this GPU"
            raise RuntimeError(msg)
        dtype = self._rnnt_dtype()
        self._model.decoder.to(dtype=dtype)
        self._model.joint.to(dtype=dtype)

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
        decode_mode: str | None = None,
    ) -> tuple[list[str], list[str]]:
        if self._model is None:
            msg = "Model not initialized. Call load_model() first."
            raise RuntimeError(msg)
        mode = (decode_mode or self.decode_mode).lower()
        if len(waveforms) != len(sample_rates) or len(waveforms) != len(lang_codes):
            msg = "waveforms, sample_rates, and lang_codes must have the same length"
            raise ValueError(msg)
        if self._chunk_duration_sec is None:
            msg = "IndicConformer chunk duration is not initialized"
            raise RuntimeError(msg)

        original_languages = [str(language).strip().lower() for language in lang_codes]
        requires_merge = has_audio_longer_than(waveforms, sample_rates, self._chunk_duration_sec)
        chunks, chunk_sample_rates, owners = split_waveforms(
            waveforms,
            sample_rates,
            self._chunk_duration_sec,
        )
        if not chunks:
            return [""] * len(waveforms), original_languages
        chunk_languages = [original_languages[owner] for owner in owners]
        texts, _ = self._generate_chunks(chunks, chunk_sample_rates, chunk_languages, mode)

        if requires_merge:
            return merge_chunk_texts(texts, owners, len(original_languages)), original_languages
        restored = [""] * len(original_languages)
        for text, owner in zip(texts, owners, strict=True):
            restored[owner] = text
        return restored, original_languages

    def _generate_chunks(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
        mode: str,
    ) -> tuple[list[str], list[str]]:
        """Batch already bounded chunks by duration and restore chunk order."""
        import torch

        texts: list[str] = [""] * len(waveforms)
        langs_out = [str(language).strip().lower() for language in lang_codes]
        prepared: list[Any] = []
        lengths: list[int] = []
        prepared_languages: list[str] = []
        original_indices: list[int] = []
        with torch.inference_mode():
            for index, (w, sr, lang) in enumerate(zip(waveforms, sample_rates, langs_out, strict=True)):
                if w is None or np.asarray(w).size == 0:
                    continue
                samples = np.asarray(w, dtype=np.float32)
                if samples.ndim != 1:
                    msg = f"ASRStage must provide a mono 1-D waveform, got shape {samples.shape}"
                    raise ValueError(msg)
                if int(sr) != _TARGET_SR:
                    msg = f"ASRStage must provide {_TARGET_SR} Hz audio; received {sr} Hz"
                    raise ValueError(msg)
                wav = torch.from_numpy(np.ascontiguousarray(samples)).to(self._device)
                prepared.append(wav)
                lengths.append(int(wav.shape[0]))
                prepared_languages.append(lang)
                original_indices.append(index)

            duration_order = sorted(range(len(prepared)), key=lengths.__getitem__)
            prepared = [prepared[index] for index in duration_order]
            lengths = [lengths[index] for index in duration_order]
            prepared_languages = [prepared_languages[index] for index in duration_order]
            original_indices = [original_indices[index] for index in duration_order]

            if prepared:
                padded = torch.nn.utils.rnn.pad_sequence(prepared, batch_first=True)
                length_tensor = torch.tensor(lengths, dtype=torch.long, device=self._device)
                encoded, encoded_len = self._model(input_signal=padded, input_signal_length=length_tensor)
                if mode == "ctc":
                    batch_texts = self._decode_ctc_batch(encoded, encoded_len, prepared_languages)
                else:
                    encoded = encoded.to(dtype=self._rnnt_dtype())
                    batch_texts = self._decode_rnnt_batch(encoded, encoded_len, prepared_languages)
                for original_index, text in zip(original_indices, batch_texts, strict=True):
                    texts[original_index] = text
        return texts, langs_out

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe supported rows and preserve the shared one-result-per-item contract."""
        results = [
            ASRResult(
                text="",
                skipped=self.empty_audio_marks_skip,
                skip_reason="empty_audio" if self.empty_audio_marks_skip else None,
            )
            for _ in items
        ]
        valid_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        sample_rates: list[int] = []
        languages: list[str] = []
        for index, item in enumerate(items):
            language = str(item.get("language_code") or "").strip().lower()
            if language not in INDIC_CONFORMER_HYBRID_LANGS:
                results[index] = ASRResult(text="", unsupported_language=language or None)
                continue
            waveform = np.asarray(item.get("waveform"), dtype=np.float32)
            if waveform.size == 0:
                continue
            valid_indices.append(index)
            waveforms.append(waveform)
            sample_rates.append(int(item.get("sample_rate") or 0))
            languages.append(language)

        if valid_indices:
            texts, languages_out = self.generate(waveforms, sample_rates, languages)
            if len(texts) != len(valid_indices):
                msg = f"IndicConformer returned {len(texts)} transcriptions for {len(valid_indices)} inputs"
                raise RuntimeError(msg)
            for index, text, language in zip(valid_indices, texts, languages_out, strict=True):
                results[index] = ASRResult(text=text, extras={"language_code": language})
        return results

    def _ids_to_text(self, local_ids: list[int], lang: str) -> str:
        """Map per-language local token ids -> aggregate ids -> text."""
        if not local_ids:
            return ""
        offset = self._model.tokenizer.token_id_offset[lang]
        agg_ids = [int(i) + offset for i in local_ids]
        return self._model.tokenizer.ids_to_text(agg_ids).strip()

    def _decode_ctc_batch(self, encoded: Any, encoded_len: Any, lang_codes: list[str]) -> list[str]:
        log_probs = self._model.ctc_decoder(encoder_output=encoded, language_ids=lang_codes)
        return [
            self._decode_ctc_row(log_probs[index], int(encoded_len[index].item()), language)
            for index, language in enumerate(lang_codes)
        ]

    def _decode_ctc_row(self, log_probs: Any, encoded_len: int, lang: str) -> str:
        preds = log_probs[:encoded_len].argmax(dim=-1).tolist()
        blank = self._per_lang_classes  # per-language blank sits at the last index
        out: list[int] = []
        prev = None
        for p in preds:
            if p != blank and p != prev:
                out.append(p)
            prev = p
        return self._ids_to_text(out, lang)

    def _rnnt_decoder(self, lang: str) -> Any:
        decoder = self._rnnt_decoders.get(lang)
        if decoder is not None:
            return decoder

        from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer

        decoder = GreedyBatchedRNNTInfer(
            decoder_model=_LanguageRNNTDecoder(self._model.decoder, self._per_lang_classes),
            joint_model=_LanguageRNNTJoint(
                self._model.joint,
                lang,
                self._per_lang_classes + 1,
            ),
            blank_index=self._per_lang_classes,
            max_symbols_per_step=self.max_symbols_per_step,
            preserve_alignments=False,
            preserve_frame_confidence=False,
            loop_labels=True,
            use_cuda_graph_decoder=self._device.type == "cuda",
        )
        self._rnnt_decoders[lang] = decoder
        return decoder

    def _decode_rnnt_batch(self, encoded: Any, encoded_len: Any, lang_codes: list[str]) -> list[str]:
        import torch

        texts = [""] * len(lang_codes)
        language_groups: dict[str, list[int]] = {}
        for index, language in enumerate(lang_codes):
            language_groups.setdefault(language, []).append(index)

        for language, indices in language_groups.items():
            if len(indices) == len(lang_codes):
                group_encoded = encoded
                group_lengths = encoded_len
            else:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=encoded.device)
                group_encoded = encoded.index_select(0, index_tensor)
                group_lengths = encoded_len.index_select(0, index_tensor)
            hypotheses = self._rnnt_decoder(language)(
                encoder_output=group_encoded,
                encoded_lengths=group_lengths,
            )[0]
            for index, hypothesis in zip(indices, hypotheses, strict=True):
                token_ids = hypothesis.y_sequence
                if torch.is_tensor(token_ids):
                    token_ids = token_ids.tolist()
                texts[index] = self._ids_to_text(token_ids, language)
        return texts
