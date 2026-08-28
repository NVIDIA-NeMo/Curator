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
from typing import Any, Literal

import numpy as np
from loguru import logger

from nemo_curator.models.asr.base import ASRResult

_TARGET_SR = 16_000

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
    ):
        if not model_id:
            msg = "IndicConformerHybridASR.model_id must be non-empty"
            raise ValueError(msg)
        if revision is not None:
            msg = "IndicConformerHybridASR does not support revision pinning"
            raise ValueError(msg)
        self.model_id = model_id
        self.revision = revision
        self.decode_mode = decode_mode
        self.max_symbols_per_step = max_symbols_per_step
        self._model: Any = None
        self._device: Any = None
        self._num_langs: int = 0
        self._per_lang_classes: int = 0  # V / num_langs (blank index within a head)

    @staticmethod
    def _resolve_nemo_path(model_id: str) -> str:
        """Resolve ``model_id`` to a local ``.nemo`` path.

        Accepts a local ``.nemo`` file, or a HuggingFace repo id like
        ``ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large`` (downloads the
        single ``.nemo`` it contains). The HF repos are gated — set ``HF_TOKEN``.
        """
        import os

        if model_id.endswith(".nemo") or os.path.exists(model_id):
            return model_id
        from huggingface_hub import HfApi, hf_hub_download

        files = [f for f in HfApi().list_repo_files(model_id) if f.endswith(".nemo")]
        if not files:
            msg = f"No .nemo file found in HuggingFace repo '{model_id}'"
            raise RuntimeError(msg)
        return hf_hub_download(model_id, files[0])

    def download_weights_on_node(self) -> None:
        """Resolve the configured checkpoint into the node-local cache without loading it."""
        self._resolve_nemo_path(self.model_id)

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
        self._model = None
        self._device = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
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
        import torch

        texts: list[str] = []
        langs_out: list[str] = []
        with torch.inference_mode():
            for w, sr, lang in zip(waveforms, sample_rates, lang_codes, strict=True):
                if w is None or np.asarray(w).size == 0:
                    texts.append("")
                    langs_out.append(lang)
                    continue
                samples = np.asarray(w, dtype=np.float32)
                if samples.ndim != 1:
                    msg = f"ASRStage must provide a mono 1-D waveform, got shape {samples.shape}"
                    raise ValueError(msg)
                if int(sr) != _TARGET_SR:
                    msg = f"ASRStage must provide {_TARGET_SR} Hz audio; received {sr} Hz"
                    raise ValueError(msg)
                wav = torch.from_numpy(np.ascontiguousarray(samples)).to(self._device)
                length = torch.tensor([wav.shape[0]], device=self._device)
                encoded, encoded_len = self._model(
                    input_signal=wav.unsqueeze(0), input_signal_length=length
                )  # encoded: [B, D, T]
                if mode == "ctc":
                    text = self._decode_ctc(encoded, encoded_len, lang)
                else:
                    text = self._decode_rnnt(encoded, int(encoded_len[0].item()), lang)
                texts.append(text)
                langs_out.append(lang)
        return texts, langs_out

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe supported rows and preserve the shared one-result-per-item contract."""
        results = [ASRResult(text="", skipped=True, skip_reason="empty_audio") for _ in items]
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
            texts, _ = self.generate(waveforms, sample_rates, languages)
            if len(texts) != len(valid_indices):
                msg = f"IndicConformer returned {len(texts)} transcriptions for {len(valid_indices)} inputs"
                raise RuntimeError(msg)
            for index, text in zip(valid_indices, texts, strict=True):
                results[index] = ASRResult(text=text)
        return results

    def _ids_to_text(self, local_ids: list[int], lang: str) -> str:
        """Map per-language local token ids -> aggregate ids -> text."""
        if not local_ids:
            return ""
        offset = self._model.tokenizer.token_id_offset[lang]
        agg_ids = [int(i) + offset for i in local_ids]
        return self._model.tokenizer.ids_to_text(agg_ids).strip()

    def _decode_ctc(self, encoded: Any, encoded_len: Any, lang: str) -> str:
        log_probs = self._model.ctc_decoder(encoder_output=encoded, language_ids=[lang])  # [1, T, per_lang+1]
        elen = int(encoded_len[0].item())
        preds = log_probs[0, :elen].argmax(dim=-1).tolist()
        blank = self._per_lang_classes  # per-language blank sits at the last index
        out: list[int] = []
        prev = None
        for p in preds:
            if p != blank and p != prev:
                out.append(p)
            prev = p
        return self._ids_to_text(out, lang)

    def _decode_rnnt(self, encoded: Any, enc_len: int, lang: str) -> str:
        # Compact greedy transducer decode mirroring the fork's single-sample path:
        # per-language joint head, blank index = V/num_langs, local-id feedback.
        import torch

        joint = self._model.joint
        decoder = self._model.decoder
        blank = self._per_lang_classes
        x = encoded.transpose(1, 2)  # [B, T, D_enc]
        f_enc = joint.enc(x)  # project encoder once: [B, T, H]

        last_token: int | None = None
        state: Any = None
        hyp: list[int] = []
        for t in range(enc_len):
            f = f_enc[:, t : t + 1, :]  # [B, 1, H]
            not_blank = True
            symbols = 0
            while not_blank and symbols < self.max_symbols_per_step:
                if last_token is None and state is None:
                    g, new_state = decoder.predict(None, state=None, add_sos=False, batch_size=1)
                else:
                    label = torch.full([1, 1], fill_value=last_token, dtype=torch.long, device=self._device)
                    g, new_state = decoder.predict(label, state=state, add_sos=False, batch_size=1)
                g = joint.pred(g)  # [1, 1, H]
                logp = joint.joint_after_projection(f, g, language_ids=[lang])[0, 0, 0, :]
                k = int(logp.argmax(dim=-1).item())
                if k == blank:
                    not_blank = False
                else:
                    hyp.append(k)
                    last_token = k
                    state = new_state
                symbols += 1
        return self._ids_to_text(hyp, lang)
