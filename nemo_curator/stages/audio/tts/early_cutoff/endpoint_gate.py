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

"""GPU early cut-off endpoint gate.

Port of TTS Granary ``EarlyCutOffEndpointGateProcessor``. Scores each utterance
with a phoneme CTC model and a frozen endpoint head. Rows are annotated, never
dropped. MFA verification is a later step; candidates are marked ``REVIEW``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from nemo_curator.stages.audio.common import resolve_waveform_from_item
from nemo_curator.stages.audio.tts.early_cutoff.endpoint_classifier import (
    SCALAR_FEATURES,
    EndpointCheckpoint,
    endpoint_scalar_features,
    load_endpoint_checkpoint,
)
from nemo_curator.stages.audio.tts.early_cutoff.phonemization import (
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZATION_VERSION,
    ModelCompatiblePhonemizer,
    PhonemizedReference,
)
from nemo_curator.stages.audio.tts.fields import MISSING, get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

SCHEMA_VERSION = 13
CASCADE_VERSION = "endpoint_mfa_v13"
SUPPORTED_LANGUAGES = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)
_TEXT_FALLBACKS = ("itn_text", "GranaryV2.tn_raw")
_FINAL_STEPS = frozenset({"gpu_endpoint_gate", "mfa_resolver"})
_FINAL_STATUS = frozenset({"OK", "INDETERMINATE", "ERROR"})
_FINAL_DECISIONS = frozenset({"OK", "REVIEW", "DROP"})


def language_code(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace("_", "-").split("-", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle_artifact(manifest_path: Path | None, artifact: Path) -> None:
    if manifest_path is None or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = (manifest.get("artifacts") or {}).get(artifact.name)
    if not isinstance(record, dict) or not record.get("sha256"):
        msg = f"{artifact.name} is absent from production bundle manifest"
        raise ValueError(msg)
    if _sha256(artifact) != str(record["sha256"]):
        msg = f"production artifact hash mismatch: {artifact}"
        raise ValueError(msg)


def _endpoint_threshold(calibration_path: Path | None, language: str, checkpoint: EndpointCheckpoint) -> float:
    if calibration_path is None or not calibration_path.is_file():
        return float(checkpoint.threshold)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if "global_thresholds" in calibration:
        thresholds = (calibration.get("language_overrides") or {}).get(language, calibration["global_thresholds"])
    else:
        thresholds = calibration["thresholds"]
    return float(thresholds["endpoint_threshold"])


def _error(code: str, message: str | None = None) -> dict[str, str]:
    payload = {"code": code}
    if message:
        payload["message"] = message
    return payload


def empty_detection(error_code: str, message: str | None = None) -> dict[str, Any]:
    return {
        "analysis_status": "ERROR",
        "error": _error(error_code, message),
        "decision": "REVIEW",
        "is_suspicious": False,
        "cut_score": None,
        "raw_score": None,
        "reference_phone_count": None,
        "estimated_last_supported_phone_index": None,
        "missing_phone_count": None,
        "missing_phone_fraction": None,
        "estimated_cut_time_sec": None,
        "estimated_cut_time_uncertainty_sec": None,
        "word_index": None,
        "word": None,
        "phone": None,
        "position_inside_word": None,
        "supported_text": None,
        "missing_text": None,
        "prefix_advantage": None,
        "posterior_drop": None,
        "prefix_support": None,
        "suffix_unsupported_score": None,
        "suffix_recovery_score": None,
        "eof_score": None,
        "alignment_quality": None,
        "tail_blank_probability": None,
        "ctc_scores": None,
        "candidate_count": 0,
        "quality_flags": [],
    }


def feature_output_lengths(
    input_lengths: torch.Tensor,
    conv_kernels: list[int] | tuple[int, ...],
    conv_strides: list[int] | tuple[int, ...],
) -> torch.Tensor:
    lengths = input_lengths.to(dtype=torch.long)
    for kernel, stride in zip(conv_kernels, conv_strides, strict=True):
        lengths = torch.div(lengths - int(kernel), int(stride), rounding_mode="floor") + 1
    return torch.clamp_min(lengths, 0)


def _with_provenance(  # noqa: PLR0913
    annotation: dict[str, Any],
    *,
    language: str,
    model_id: str,
    model_revision: str,
    text_key: str,
    reference: PhonemizedReference | None = None,
) -> dict[str, Any]:
    annotation.update(
        {
            "schema_version": SCHEMA_VERSION,
            "cascade_version": CASCADE_VERSION,
            "cascade_step": "gpu_endpoint_gate",
            "model_id": model_id,
            "model_revision": model_revision,
            "phonemizer_version": "transformers-compatible-espeak-v1",
            "normalization_version": NORMALIZATION_VERSION,
            "language": language,
            "text_source": text_key,
            "phone_source": reference.source if reference else None,
            "phonemizer_language": reference.phonemizer_language if reference else None,
            "oov_tokens": list(reference.oov_tokens) if reference else [],
            "unsupported_phone_count": len(reference.oov_tokens) if reference else None,
            "reference_eligible_for_drop": bool(reference and reference.eligible),
            "annotation_only": True,
        }
    )
    return annotation


def _is_final(existing: Any) -> bool:  # noqa: ANN401
    return (
        isinstance(existing, dict)
        and existing.get("schema_version") == SCHEMA_VERSION
        and existing.get("cascade_version") == CASCADE_VERSION
        and existing.get("cascade_step") in _FINAL_STEPS
        and existing.get("analysis_status") in _FINAL_STATUS
        and existing.get("decision") in _FINAL_DECISIONS
        and existing.get("annotation_only") is True
    )


@dataclass
class _EndpointHead:
    model: Any
    checkpoint: EndpointCheckpoint
    calibration_path: Path | None


@dataclass
class EarlyCutOffEndpointGateStage(ProcessingStage[AudioTask, AudioTask]):
    """Annotate utterances with the CTC endpoint-gate decision.

    Reads ``text_key`` (default ``tn_raw``) and optional precomputed IPA at
    ``ipa_key``. Writes the TTS Granary early-cut-off dict at ``output_key``.
    English uses ``english_checkpoint``; every other supported language uses
    ``multilingual_checkpoint``.
    """

    text_key: str = "tn_raw"
    ipa_key: str = "ipa"
    output_key: str = "early_cut_off_detection"
    source_lang_key: str = "source_lang"
    language: str | None = None
    bundle_dir: str | None = None
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    model_path: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    dtype: str = "float16"
    english_checkpoint: str | None = None
    english_calibration: str | None = None
    multilingual_checkpoint: str | None = None
    multilingual_calibration: str | None = None
    bundle_manifest: str | None = None
    append_silence_sec: float = 0.30
    sample_rate: int = 16000
    overwrite: bool = False
    name: str = "EarlyCutOffEndpointGate"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))
    batch_size: int = 4

    def __post_init__(self) -> None:
        super().__init__()
        self._apply_bundle_dir()
        self._setup_error: str | None = None
        self._model: Any = None
        self._feature_extractor: Any = None
        self._phonemizer: ModelCompatiblePhonemizer | None = None
        self._detector: Any = None
        self._blank_id = 0
        self._frame_stride_sec = 0.02
        self._heads: dict[str, _EndpointHead] = {}

    def _apply_bundle_dir(self) -> None:
        if not self.bundle_dir:
            return
        root = Path(self.bundle_dir).expanduser()
        self.english_checkpoint = self.english_checkpoint or str(root / "english_endpoint.pt")
        self.english_calibration = self.english_calibration or str(root / "english_verifier.json")
        self.multilingual_checkpoint = self.multilingual_checkpoint or str(root / "multilingual_endpoint.pt")
        self.multilingual_calibration = self.multilingual_calibration or str(root / "multilingual_thresholds.json")
        self.bundle_manifest = self.bundle_manifest or str(root / "manifest.json")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        try:
            self._load_models()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cannot initialize early cut-off endpoint gate")
            self._setup_error = f"{type(exc).__name__}: {exc}"

    def _load_models(self) -> None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

        from nemo_curator.stages.audio.tts.early_cutoff.detector import DetectorConfig, EarlyCutOffDetector

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        model_ref = self.model_path or self.model_id
        common = {
            "revision": self.model_revision,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir,
        }
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_ref, do_phonemize=False, **common)
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_ref, **common)
        dtype_name = self.dtype.lower()
        dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name, torch.float16)
        load_dtype = dtype if device.type == "cuda" else torch.float32
        self._model = Wav2Vec2ForCTC.from_pretrained(model_ref, dtype=load_dtype, **common).to(device)
        self._model.eval()
        pad_id = self._model.config.pad_token_id
        self._blank_id = int(pad_id if pad_id is not None else tokenizer.pad_token_id)
        stride_samples = math.prod(int(value) for value in self._model.config.conv_stride)
        self._frame_stride_sec = stride_samples / self.sample_rate
        self._phonemizer = ModelCompatiblePhonemizer(
            tokenizer,
            language_map={"en": "en-us", "fr": "fr-fr"},
            isolate_languages=("da",),
        )
        self._detector = EarlyCutOffDetector(DetectorConfig.from_dict(None), None)
        manifest = Path(self.bundle_manifest).expanduser() if self.bundle_manifest else None
        hidden_size = int(self._model.config.hidden_size)
        self._heads = {}
        for family, checkpoint_path, calibration_path in (
            ("en", self.english_checkpoint, self.english_calibration),
            ("multi", self.multilingual_checkpoint, self.multilingual_calibration),
        ):
            if not checkpoint_path:
                continue
            path = Path(checkpoint_path).expanduser()
            calibration = Path(calibration_path).expanduser() if calibration_path else None
            _verify_bundle_artifact(manifest, path)
            if calibration is not None:
                _verify_bundle_artifact(manifest, calibration)
            model, checkpoint = load_endpoint_checkpoint(path, map_location=device)
            if checkpoint.config.acoustic_hidden_size != hidden_size:
                msg = "endpoint checkpoint acoustic hidden size does not match CTC model"
                raise ValueError(msg)
            required_mask = {name for name in SCALAR_FEATURES if name.startswith("mfa_")}
            missing = required_mask.difference(checkpoint.masked_scalar_features)
            if missing:
                msg = "endpoint checkpoint is not CTC-only; unmasked MFA features: " + ", ".join(sorted(missing))
                raise ValueError(msg)
            model = model.to(device)
            model.eval()
            self._heads[family] = _EndpointHead(model=model, checkpoint=checkpoint, calibration_path=calibration)
        if not self._heads:
            msg = "english_checkpoint or multilingual_checkpoint is required"
            raise ValueError(msg)
        if not math.isfinite(self.append_silence_sec) or round(self.append_silence_sec * self.sample_rate) < 1:
            msg = "append_silence_sec must be positive"
            raise ValueError(msg)

    def _resolve_language(self, payload: dict[str, Any]) -> str:
        if self.language:
            return language_code(self.language)
        raw = payload.get(self.source_lang_key) or payload.get("language") or payload.get("lang")
        return language_code(str(raw) if raw else "en")

    def _resolve_text(self, payload: dict[str, Any]) -> Any:  # noqa: ANN401
        for key in (self.text_key, *_TEXT_FALLBACKS):
            value = get_dotted(payload, key, MISSING)
            if value is not MISSING:
                return value
        return MISSING

    def _resolve_ipa(self, payload: dict[str, Any]) -> str | None:
        value = get_dotted(payload, self.ipa_key, MISSING)
        if value is MISSING:
            value = payload.get(self.ipa_key, MISSING)
        if isinstance(value, dict):
            value = value.get("ipa")
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _write(
        self,
        payload: dict[str, Any],
        annotation: dict[str, Any],
        language: str,
        reference: PhonemizedReference | None = None,
    ) -> None:
        set_dotted(
            payload,
            self.output_key,
            _with_provenance(
                annotation,
                language=language,
                model_id=self.model_id,
                model_revision=self.model_revision,
                text_key=self.text_key,
                reference=reference,
            ),
        )

    def _unavailable(self, payload: dict[str, Any], language: str) -> None:
        code = (
            "unsupported_language" if language == "mt" or language not in SUPPORTED_LANGUAGES else "model_unavailable"
        )
        self._write(payload, empty_detection(code, self._setup_error), language)

    def _prepare_item(  # noqa: C901, PLR0911
        self, payload: dict[str, Any], task_id: str
    ) -> tuple[dict[str, Any], Any, PhonemizedReference] | None:
        language = self._resolve_language(payload)
        existing = get_dotted(payload, self.output_key)
        if not self.overwrite and _is_final(existing):
            return None
        if self._setup_error is not None or self._phonemizer is None or self._detector is None:
            self._unavailable(payload, language)
            return None
        if language == "mt" or language not in SUPPORTED_LANGUAGES:
            self._write(payload, empty_detection("unsupported_language", language or "unknown"), language)
            return None
        family = "en" if language == "en" else "multi"
        if family not in self._heads:
            self._write(payload, empty_detection("model_unavailable", f"no {family} endpoint checkpoint"), language)
            return None

        text = self._resolve_text(payload)
        if text is MISSING:
            self._write(payload, empty_detection("missing_text"), language)
            return None
        if not isinstance(text, str) or not text.strip():
            code = "invalid_text" if not isinstance(text, str) else "empty_text"
            self._write(payload, empty_detection(code), language)
            return None
        try:
            reference = self._phonemizer.phonemize(text, language, precomputed_ipa=self._resolve_ipa(payload))
        except Exception as exc:  # noqa: BLE001
            self._write(payload, empty_detection("phonemization_failed", str(exc)), language)
            return None
        if not reference.eligible:
            detection = self._detector._empty_result(
                reference,
                status="INDETERMINATE",
                error=_error("unsupported_reference", "empty or OOV phone sequence"),
            )
            self._write(payload, detection, language, reference)
            return None
        waveform = self._load_waveform(payload, task_id)
        if waveform is None:
            self._write(payload, empty_detection("missing_audio"), language, reference)
            return None
        if int(waveform.numel()) <= 0:
            self._write(payload, empty_detection("audio_load_failed", "decoded empty waveform"), language, reference)
            return None
        return payload, waveform.to(dtype=torch.float32).reshape(-1).cpu(), reference

    def _load_waveform(self, payload: dict[str, Any], task_id: str) -> Any:  # noqa: ANN401
        audio = resolve_waveform_from_item(payload, task_id)
        if audio is not None:
            waveform, sample_rate = audio
            waveform = waveform.detach().cpu().reshape(-1).to(dtype=torch.float32)
            return _resample(waveform, int(sample_rate), self.sample_rate)
        return _load_tar_waveform(payload, self.sample_rate)

    def _flush(self, items: list[tuple[dict[str, Any], Any, PhonemizedReference]], language: str) -> None:
        if not items:
            return
        family = "en" if language_code(language) == "en" else "multi"
        head = self._heads[family]
        threshold = _endpoint_threshold(head.calibration_path, language_code(language), head.checkpoint)
        try:
            self._infer(items, language, head, threshold)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Acoustic inference batch failed: {}", exc)
            for payload, _, reference in items:
                self._write(payload, empty_detection("model_inference_failed", str(exc)), language, reference)

    def _infer(
        self,
        items: list[tuple[dict[str, Any], Any, PhonemizedReference]],
        language: str,
        head: _EndpointHead,
        threshold: float,
    ) -> None:
        silence_samples = round(float(self.append_silence_sec) * self.sample_rate)
        waveforms = [waveform for _, waveform, _ in items]
        originals = [waveform.numpy() for waveform in waveforms]
        appended = [torch.nn.functional.pad(waveform, (0, silence_samples)).numpy() for waveform in waveforms]
        inputs = self._feature_extractor(
            [*originals, *appended],
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_dtype = next(self._model.parameters()).dtype
        input_values = inputs.input_values.to(device=self._device, dtype=model_dtype)
        attention_mask = inputs.attention_mask.to(self._device)
        with torch.inference_mode():
            model_output = self._model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        logits = model_output.logits
        hidden = model_output.hidden_states[-1]
        log_probs = logits.float().log_softmax(dim=-1)
        all_lengths = feature_output_lengths(
            attention_mask.sum(dim=-1),
            self._model.config.conv_kernel,
            self._model.config.conv_stride,
        )
        output_lengths = all_lengths[: len(items)]
        appended_lengths = all_lengths[len(items) :]
        detections = self._detect_rows(items, log_probs, output_lengths, appended_lengths)
        probabilities = self._endpoint_probabilities(items, hidden, output_lengths, detections, head)
        for (payload, _, reference), detection, probability in zip(items, detections, probabilities, strict=True):
            candidate = detection.get("analysis_status") == "OK" and float(probability) > threshold
            gate_error = detection.get("analysis_status") != "OK"
            detection.update(
                {
                    "endpoint_probability": float(probability),
                    "endpoint_threshold": threshold,
                    "endpoint_score_version": head.checkpoint.score_version,
                    "mfa_verification_required": candidate,
                    "decision": "REVIEW" if candidate or gate_error else "OK",
                    "is_suspicious": candidate,
                    "verifier_status": "PENDING"
                    if candidate
                    else "GATE_ERROR_REVIEW"
                    if gate_error
                    else "NOT_INVOKED",
                    "verification_status": "PENDING" if candidate else "GATE_ERROR" if gate_error else "NOT_INVOKED",
                }
            )
            self._write(payload, detection, language, reference)

    def _detect_rows(
        self,
        items: list[tuple[dict[str, Any], Any, PhonemizedReference]],
        log_probs: torch.Tensor,
        output_lengths: torch.Tensor,
        appended_lengths: torch.Tensor,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for row, ((_, waveform, reference), output_length) in enumerate(
            zip(items, output_lengths.tolist(), strict=True)
        ):
            try:
                detection = self._detector.detect(
                    log_probs[row, : int(output_length)],
                    reference,
                    blank_id=self._blank_id,
                    frame_stride_sec=self._frame_stride_sec,
                    appended_silence_log_probs=log_probs[len(items) + row, : int(appended_lengths[row])],
                    waveform=waveform,
                    sample_rate=self.sample_rate,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("CTC analysis failed: {}", exc)
                detection = empty_detection("ctc_analysis_failed", str(exc))
            detections.append(detection)
        return detections

    def _endpoint_probabilities(
        self,
        items: list[tuple[dict[str, Any], Any, PhonemizedReference]],
        hidden: torch.Tensor,
        output_lengths: torch.Tensor,
        detections: list[dict[str, Any]],
        head: _EndpointHead,
    ) -> list[float]:
        config = head.checkpoint.config
        acoustic = hidden.new_zeros((len(items), config.endpoint_frames, config.acoustic_hidden_size))
        acoustic_mask = torch.zeros((len(items), config.endpoint_frames), dtype=torch.bool, device=self._device)
        phones = torch.zeros((len(items), config.reference_phones), dtype=torch.long, device=self._device)
        phone_mask = torch.zeros_like(phones, dtype=torch.bool)
        scalar_rows: list[list[float]] = []
        means = head.checkpoint.scalar_mean
        scales = head.checkpoint.scalar_scale
        masked = set(head.checkpoint.masked_scalar_features)
        for row, ((_, _, reference), output_length, detection) in enumerate(
            zip(items, output_lengths.tolist(), detections, strict=True)
        ):
            retained = min(int(output_length), config.endpoint_frames)
            acoustic[row, :retained] = hidden[row, int(output_length) - retained : int(output_length)]
            acoustic_mask[row, :retained] = True
            tail = reference.token_ids[-config.reference_phones :]
            phones[row, : len(tail)] = torch.tensor(tail, device=self._device)
            phone_mask[row, : len(tail)] = True
            values, observed = endpoint_scalar_features({"prediction": detection})
            normalized: list[float] = []
            for index, (raw_value, raw_present) in enumerate(zip(values, observed, strict=True)):
                present = 0.0 if SCALAR_FEATURES[index] in masked else raw_present
                numeric = raw_value if present else means[index]
                scale = scales[index] if scales[index] else 1.0
                normalized.append(max(-8.0, min(8.0, (numeric - means[index]) / scale)))
                observed[index] = present
            scalar_rows.append(normalized + observed)
        with torch.inference_mode():
            logits = head.model(
                acoustic_states=acoustic,
                acoustic_mask=acoustic_mask,
                reference_phones=phones,
                reference_mask=phone_mask,
                scalar_features=torch.tensor(scalar_rows, dtype=torch.float32, device=self._device),
            )
            return torch.sigmoid(logits).cpu().tolist()

    def _annotate_payload(self, payload: dict[str, Any], task_id: str, pending: dict[str, list]) -> None:
        segments = payload.get("segments")
        if isinstance(segments, list):
            for index, segment in enumerate(segments):
                if isinstance(segment, dict):
                    prepared = self._prepare_item(segment, f"{task_id}:{index}")
                    if prepared is not None:
                        pending.setdefault(self._resolve_language(segment), []).append(prepared)
            return
        prepared = self._prepare_item(payload, task_id)
        if prepared is not None:
            pending.setdefault(self._resolve_language(payload), []).append(prepared)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        pending: dict[str, list] = {}
        for task in tasks:
            self._annotate_payload(task.data, task.task_id, pending)
        for language, items in pending.items():
            self._flush(items, language)
        return tasks


def _resample(waveform: Any, sample_rate: int, target_sample_rate: int) -> Any:  # noqa: ANN401
    if sample_rate == target_sample_rate:
        return waveform
    target_len = max(1, round(waveform.shape[-1] * target_sample_rate / sample_rate))
    return torch.nn.functional.interpolate(
        waveform.view(1, 1, -1),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).view(-1)


def _load_tar_waveform(payload: dict[str, Any], target_sample_rate: int) -> Any:  # noqa: ANN401
    import io

    import soundfile as sf

    tar_key = get_dotted(payload, "GranaryHifi.AudioMetadata.tar_key")
    offset = get_dotted(payload, "GranaryHifi.AudioMetadata.tar_member_offset")
    size = get_dotted(payload, "GranaryHifi.AudioMetadata.tar_member_bytes")
    if not tar_key or offset is None or size is None:
        return None
    path = Path(str(tar_key)).expanduser()
    if not path.is_file():
        return None
    with path.open("rb") as src:
        src.seek(int(offset))
        payload_bytes = src.read(int(size))
    audio, sample_rate = sf.read(io.BytesIO(payload_bytes), dtype="float32", always_2d=False)
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)
    return _resample(waveform.reshape(-1), int(sample_rate), target_sample_rate)
