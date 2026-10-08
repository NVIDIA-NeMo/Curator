# ruff: noqa
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

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from .ctc_dp_numpy import (
    align_phones_numpy,
    ctc_prefix_forward_numpy,
    minimum_ctc_frames_numpy,
)
from .phonemization import PhonemizedReference

SCORE_VERSION = "terminal_binary_eou_v11"
SCORE_OUTPUT_DIGITS = 6


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _finite_mean(values: list[float | None], default: float) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else default


def _round(value: float | None, digits: int = SCORE_OUTPUT_DIGITS) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _bounded_log_odds(
    positive_log_probability: float,
    negative_log_probability: float,
    *,
    bound: float = 30.0,
) -> float | None:
    if math.isfinite(positive_log_probability) and math.isfinite(negative_log_probability):
        return max(
            -bound,
            min(bound, positive_log_probability - negative_log_probability),
        )
    if math.isfinite(positive_log_probability):
        return bound
    if math.isfinite(negative_log_probability):
        return -bound
    return None


@dataclass
class DetectorConfig:
    min_prefix_phones: int = 2
    confidence_window_phones: int = 3
    eof_scale_frames: float = 5.0
    prefix_advantage_center: float = 0.0
    prefix_advantage_scale: float = 1.0
    phone_margin_center: float = -2.0
    phone_margin_scale: float = 2.0
    posterior_drop_center: float = 1.0
    posterior_drop_scale: float = 2.0
    review_threshold: float = 0.55
    drop_threshold: float = 0.82
    min_prefix_support_for_drop: float = 0.45
    min_eof_score_for_drop: float = 0.35
    min_suffix_unsupported_for_drop: float = 0.55
    terminal_word_max_missing_phones: int = 3
    terminal_margin_drop_center: float = 0.5
    terminal_margin_drop_scale: float = 1.5
    min_terminal_score_for_drop: float = 0.70
    partial_final_phone_review_only: bool = True
    binary_decisions: bool = True
    allow_ctc_partial_phone_drop: bool = True
    require_appended_silence_for_partial_drop: bool = True
    min_partial_score_for_drop: float = 0.90
    min_eof_discontinuity_for_partial_drop: float = 0.55
    min_appended_closure_for_partial_drop: float = 0.45
    endpoint_duration_ratio_center: float = 0.65
    endpoint_duration_ratio_scale: float = 0.20
    eof_discontinuity_center: float = 0.25
    eof_discontinuity_scale: float = 0.15
    eou_confidence_center: float = 0.12
    eou_confidence_scale: float = 0.05
    eou_trailing_center_sec: float = 0.10
    eou_trailing_scale_sec: float = 0.04
    eou_gap_center_sec: float = 0.40
    eou_gap_scale_sec: float = 0.10
    min_eou_score_for_partial_drop: float = 0.70
    max_endpoint_bad_region_score_for_drop: float = 0.50
    terminal_probability_slope_center: float = 0.03
    terminal_probability_slope_scale: float = 0.03
    partial_ctc_min_complete_frames: int = 3
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "prefix_advantage": 0.30,
            "posterior_drop": 0.25,
            "suffix_unsupported": 0.20,
            "suffix_non_recovery": 0.10,
            "eof": 0.15,
        }
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DetectorConfig":
        payload = dict(payload or {})
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known})


class EmpiricalScoreCalibration:
    """Empirical CDF for the already maximized clip statistic."""

    def __init__(
        self,
        null_scores: list[float],
        *,
        whole_phone_null_scores: list[float] | None = None,
        partial_phone_null_scores: list[float] | None = None,
        review_threshold: float | None = None,
        drop_threshold: float | None = None,
    ) -> None:
        self.null_scores = sorted(float(score) for score in null_scores if math.isfinite(score))
        self.whole_phone_null_scores = sorted(
            float(score) for score in (whole_phone_null_scores or null_scores) if math.isfinite(score)
        )
        self.partial_phone_null_scores = sorted(
            float(score) for score in (partial_phone_null_scores or null_scores) if math.isfinite(score)
        )
        self.review_threshold = review_threshold
        self.drop_threshold = drop_threshold

    @classmethod
    def from_json(cls, path: str | Path) -> "EmpiricalScoreCalibration":
        with Path(path).open("r", encoding="utf-8") as src:
            payload = json.load(src)
        if isinstance(payload, dict) and payload.get("score_version") != SCORE_VERSION:
            raise ValueError(
                "calibration score_version is incompatible with "
                f"{SCORE_VERSION}; regenerate predictions and calibration"
            )
        values = payload.get("null_raw_scores") if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError("calibration JSON must contain null_raw_scores")
        thresholds = (payload.get("thresholds") or {}) if isinstance(payload, dict) else {}
        channel_scores = payload.get("null_scores_by_channel") or {} if isinstance(payload, dict) else {}
        return cls(
            values,
            whole_phone_null_scores=channel_scores.get("whole_phone"),
            partial_phone_null_scores=channel_scores.get("partial_phone"),
            review_threshold=(float(thresholds["review"]) if thresholds.get("review") is not None else None),
            drop_threshold=(float(thresholds["drop"]) if thresholds.get("drop") is not None else None),
        )

    def percentile(self, raw_score: float, *, channel: str = "combined") -> float:
        values = {
            "combined": self.null_scores,
            "whole_phone": self.whole_phone_null_scores,
            "partial_phone": self.partial_phone_null_scores,
        }.get(channel)
        if values is None:
            raise ValueError(f"unknown calibration channel: {channel}")
        if not values:
            return raw_score
        # Use a strict empirical rank. With bisect_right, a large tied mass
        # (notably the legitimate partial-phone score of 0) is mapped to the
        # top percentile and becomes an anomaly. Strict ranking maps ties to
        # the same non-exceedance level as their null peers, while a score
        # above the largest observed null still maps to 1.0.
        # Calibration inputs are persisted through `_round`; quantize inference
        # scores identically so a tiny positive floating-point residue does not
        # rank above a stored null mass of 0.0.
        calibrated_score = round(float(raw_score), SCORE_OUTPUT_DIGITS)
        rank = bisect.bisect_left(values, calibrated_score)
        return rank / len(values)


class EarlyCutOffDetector:
    def __init__(
        self,
        config: DetectorConfig | None = None,
        calibration: EmpiricalScoreCalibration | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.calibration = calibration
        for name in (
            "eof_scale_frames",
            "prefix_advantage_scale",
            "phone_margin_scale",
            "posterior_drop_scale",
            "terminal_margin_drop_scale",
            "endpoint_duration_ratio_scale",
            "eof_discontinuity_scale",
            "eou_confidence_scale",
            "eou_trailing_scale_sec",
            "eou_gap_scale_sec",
            "terminal_probability_slope_scale",
        ):
            if not math.isfinite(float(getattr(self.config, name))) or float(getattr(self.config, name)) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.config.weights or any(
            not math.isfinite(float(weight)) or float(weight) < 0 for weight in self.config.weights.values()
        ):
            raise ValueError("detector weights must be finite and non-negative")
        if (
            isinstance(self.config.partial_ctc_min_complete_frames, bool)
            or not isinstance(self.config.partial_ctc_min_complete_frames, int)
            or self.config.partial_ctc_min_complete_frames < 2
        ):
            raise ValueError("partial_ctc_min_complete_frames must be an integer >= 2")

    def _empty_result(
        self,
        reference: PhonemizedReference,
        *,
        status: str,
        error: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "analysis_status": status,
            "error": error,
            "decision": "REVIEW",
            "is_suspicious": False,
            "cut_score": None,
            "raw_score": None,
            "reference_phone_count": len(reference.phones),
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
            "score_version": SCORE_VERSION,
            "terminal_candidate_type": None,
            "terminal_word_score": None,
            "terminal_suffix_score": None,
            "terminal_suffix_unsupported_score": None,
            "partial_final_phone_score": None,
            "terminal_prefix_phone_count": None,
            "terminal_missing_phone_count": None,
            "terminal_right_censor_score": None,
            "terminal_censor_log_odds": None,
            "terminal_partial_ctc_log_probability": None,
            "terminal_complete_ctc_log_probability": None,
            "terminal_partial_ctc_log_likelihood_ratio": None,
            "terminal_partial_ctc_probability": None,
            "terminal_partial_ctc_min_complete_frames": None,
            "terminal_partial_ctc_consistency_error": None,
            "appended_terminal_censor_log_odds": None,
            "appended_completion_log_odds_delta": None,
            "terminal_final_phone_margin_drop": None,
            "terminal_final_phone_touches_eof": None,
            "whole_phone_score": None,
            "whole_phone_cut_score": None,
            "partial_phone_cut_score": None,
            "endpoint_phone_duration_sec": None,
            "endpoint_duration_ratio": None,
            "endpoint_short_duration_score": None,
            "eof_nonblank_probability": None,
            "eof_nonblank_jump": None,
            "eof_discontinuity_score": None,
            "appended_silence_closure_score": None,
            "last_phone_confidence": None,
            "last_two_phone_avg_confidence": None,
            "last_phone_gap_sec": None,
            "trailing_duration_sec": None,
            "trail_rms_ratio": None,
            "eou_cutoff_score": None,
            "endpoint_noise_score": None,
            "endpoint_silence_score": None,
            "endpoint_bad_region_score": None,
            "terminal_phone_probability_slope": None,
            "terminal_blank_probability_slope": None,
            "terminal_probability_rise_score": None,
            "terminal_occupancy_entropy": None,
            "appended_context_confidence_delta": None,
        }

    def detect(
        self,
        log_probs: torch.Tensor,
        reference: PhonemizedReference,
        *,
        blank_id: int,
        frame_stride_sec: float,
        appended_silence_log_probs: torch.Tensor | None = None,
        waveform: torch.Tensor | np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        if not reference.eligible:
            return self._empty_result(
                reference,
                status="INDETERMINATE",
                error={
                    "code": "unsupported_reference",
                    "message": "reference is empty or contains OOV phone tokens",
                },
            )
        if log_probs.ndim != 2 or log_probs.shape[0] < 1:
            return self._empty_result(
                reference,
                status="ERROR",
                error={"code": "invalid_logits", "message": "expected non-empty [T,V] logits"},
            )
        if not 0 <= blank_id < log_probs.shape[1]:
            return self._empty_result(
                reference,
                status="ERROR",
                error={"code": "invalid_blank_id", "message": "blank is outside vocabulary"},
            )
        if torch.isnan(log_probs).any() or torch.isposinf(log_probs).any():
            return self._empty_result(
                reference,
                status="ERROR",
                error={"code": "invalid_logits", "message": "log probabilities contain NaN/+inf"},
            )

        log_probs_numpy = log_probs.detach().float().cpu().numpy()
        targets = np.asarray(reference.token_ids, dtype=np.int64)
        if bool(np.any(targets == blank_id)):
            return self._empty_result(
                reference,
                status="INDETERMINATE",
                error={"code": "blank_in_reference", "message": "reference contains CTC blank"},
            )
        n_phones = int(targets.size)
        if n_phones <= self.config.min_prefix_phones:
            return self._empty_result(
                reference,
                status="INDETERMINATE",
                error={"code": "reference_too_short", "message": "not enough phones for a cut candidate"},
            )

        forward = ctc_prefix_forward_numpy(
            log_probs_numpy,
            targets,
            blank_id=blank_id,
            terminal_phone_min_complete_frames=(self.config.partial_ctc_min_complete_frames),
        )
        terminal_ctc = forward.terminal_phone
        if terminal_ctc is None:
            return self._empty_result(
                reference,
                status="ERROR",
                error={
                    "code": "terminal_ctc_unavailable",
                    "message": "terminal CTC partition was not computed",
                },
            )
        scores = forward.prefix_log_probs
        blank_score = float(scores[0])
        normalized: list[float] = [float("-inf")]
        feasible_candidates: list[int] = []
        for phone_count in range(1, n_phones + 1):
            prefix = targets[:phone_count]
            minimum_frames = minimum_ctc_frames_numpy(prefix)
            value = float(scores[phone_count])
            if math.isfinite(value) and minimum_frames > 0:
                normalized.append((value - blank_score) / minimum_frames)
                if phone_count >= self.config.min_prefix_phones and phone_count < n_phones:
                    feasible_candidates.append(phone_count)
            else:
                normalized.append(float("-inf"))

        if not feasible_candidates:
            return self._empty_result(
                reference,
                status="INDETERMINATE",
                error={"code": "no_feasible_prefix", "message": "no cut prefix fits available CTC frames"},
            )

        full_normalized = normalized[n_phones]
        finite_extensions = [
            float(scores[idx] - scores[idx - 1])
            for idx in range(1, n_phones + 1)
            if math.isfinite(float(scores[idx])) and math.isfinite(float(scores[idx - 1]))
        ]
        extension_floor = min(finite_extensions, default=-20.0) - 5.0
        extensions = [float("nan")]
        for idx in range(1, n_phones + 1):
            current = float(scores[idx])
            previous = float(scores[idx - 1])
            extensions.append(
                current - previous if math.isfinite(current) and math.isfinite(previous) else extension_floor
            )
        window = self.config.confidence_window_phones
        local_contrasts: dict[int, float] = {}
        extension_before_by_k: dict[int, float] = {}
        extension_after_by_k: dict[int, float] = {}
        extension_recovery_by_k: dict[int, float] = {}
        for phone_count in feasible_candidates:
            extension_before = mean(extensions[max(1, phone_count - window + 1) : phone_count + 1])
            after_values = extensions[phone_count + 1 : min(n_phones, phone_count + window) + 1]
            extension_after = mean(after_values) if after_values else extension_floor
            later_values = extensions[min(n_phones + 1, phone_count + window + 1) :]
            extension_recovery = max(later_values, default=extension_after)
            extension_before_by_k[phone_count] = extension_before
            extension_after_by_k[phone_count] = extension_after
            extension_recovery_by_k[phone_count] = extension_recovery
            local_contrasts[phone_count] = extension_before - extension_after

        advantages = {
            phone_count: (
                normalized[phone_count] - full_normalized
                if math.isfinite(full_normalized)
                else local_contrasts[phone_count]
            )
            for phone_count in feasible_candidates
        }
        final_word_idx = reference.phone_to_word[-1] if len(reference.phone_to_word) == n_phones else None
        final_word_start = next(
            (idx for idx, word_idx in enumerate(reference.phone_to_word) if word_idx == final_word_idx),
            max(0, n_phones - self.config.terminal_word_max_missing_phones),
        )
        terminal_start = max(
            final_word_start,
            n_phones - max(int(self.config.terminal_word_max_missing_phones), 1),
            self.config.min_prefix_phones,
        )
        terminal_candidates = [phone_count for phone_count in feasible_candidates if phone_count >= terminal_start]
        terminal_k = (
            max(
                terminal_candidates,
                key=lambda phone_count: (
                    local_contrasts[phone_count]
                    + (0.25 * advantages[phone_count] if math.isfinite(full_normalized) else 0.0),
                    phone_count,
                ),
            )
            if terminal_candidates
            else None
        )
        best_k = max(
            feasible_candidates,
            key=lambda phone_count: (
                local_contrasts[phone_count]
                + (0.25 * advantages[phone_count] if math.isfinite(full_normalized) else 0.0),
                phone_count,
            ),
        )
        prefix_advantage = advantages[best_k]

        prefix_alignment = align_phones_numpy(
            log_probs_numpy,
            targets[:best_k],
            blank_id=blank_id,
        )
        full_feasible = minimum_ctc_frames_numpy(targets) <= log_probs.shape[0] and math.isfinite(float(scores[-1]))
        full_alignment = align_phones_numpy(log_probs_numpy, targets, blank_id=blank_id) if full_feasible else None

        terminal_suffix_score = 0.0
        terminal_suffix_unsupported_signal = 0.0
        terminal_eof_score = 0.0
        terminal_prefix_support = 0.0
        if terminal_k is not None:
            terminal_alignment = (
                prefix_alignment
                if terminal_k == best_k
                else align_phones_numpy(
                    log_probs_numpy,
                    targets[:terminal_k],
                    blank_id=blank_id,
                )
            )
            terminal_margins = [phone.acoustic_margin for phone in terminal_alignment.phones]
            terminal_before = _finite_mean(
                terminal_margins[max(0, terminal_k - window) : terminal_k],
                self.config.phone_margin_center,
            )
            terminal_prefix_support = _sigmoid(
                (terminal_before - self.config.phone_margin_center) / self.config.phone_margin_scale
            )
            terminal_final = terminal_alignment.phones[-1]
            terminal_last_frame = terminal_final.end_frame
            terminal_eof_distance = (
                log_probs.shape[0] - 1 - terminal_last_frame if terminal_last_frame is not None else log_probs.shape[0]
            )
            terminal_proximity = math.exp(-max(terminal_eof_distance, 0) / max(self.config.eof_scale_frames, 1e-6))
            terminal_phone_state = 2 * terminal_k - 1
            terminal_blank_state = 2 * terminal_k
            terminal_phone_end = float(forward.final_state_log_probs[terminal_phone_state])
            terminal_blank_end = float(forward.final_state_log_probs[terminal_blank_state])
            terminal_censor = (
                _sigmoid(terminal_phone_end - terminal_blank_end) if math.isfinite(terminal_phone_end) else 0.0
            )
            terminal_eof_score = 0.5 * (terminal_proximity + terminal_censor)
            terminal_advantage_signal = _sigmoid(
                (advantages[terminal_k] - self.config.prefix_advantage_center) / self.config.prefix_advantage_scale
            )
            terminal_drop_signal = _sigmoid(
                (local_contrasts[terminal_k] - self.config.posterior_drop_center) / self.config.posterior_drop_scale
            )
            terminal_recovery_signal = _sigmoid(
                (
                    extension_before_by_k[terminal_k]
                    - extension_recovery_by_k[terminal_k]
                    - self.config.posterior_drop_center
                )
                / self.config.posterior_drop_scale
            )
            if full_alignment is not None:
                terminal_full_margins = [phone.acoustic_margin for phone in full_alignment.phones]
                terminal_after_margin = _finite_mean(
                    terminal_full_margins[terminal_k:],
                    self.config.phone_margin_center,
                )
                terminal_suffix_unsupported_signal = _sigmoid(
                    (self.config.phone_margin_center - terminal_after_margin) / self.config.phone_margin_scale
                )
            else:
                terminal_suffix_unsupported_signal = terminal_drop_signal
            terminal_suffix_score = terminal_suffix_unsupported_signal * (
                0.30 * terminal_advantage_signal
                + 0.30 * terminal_drop_signal
                + 0.25 * terminal_eof_score
                + 0.15 * terminal_recovery_signal
            )

        endpoint_phone = prefix_alignment.phones[-1]
        tail_frames = min(5, log_probs.shape[0])
        tail_blank = float(np.exp(log_probs_numpy[-tail_frames:, blank_id]).mean())
        duration_alignment = (
            full_alignment.phones if full_alignment is not None and full_alignment.phones else prefix_alignment.phones
        )
        duration_phone = duration_alignment[-1]
        endpoint_occupancy_spread_sec = (
            duration_phone.posterior_std_frames * frame_stride_sec
            if duration_phone.posterior_std_frames is not None
            else None
        )
        endpoint_duration_sec = (
            (duration_phone.end_frame - duration_phone.start_frame + 1) * frame_stride_sec
            if duration_phone.start_frame is not None and duration_phone.end_frame is not None
            else None
        )
        preceding_durations = [
            (phone.end_frame - phone.start_frame + 1) * frame_stride_sec
            for phone in duration_alignment[max(0, len(duration_alignment) - window - 1) : -1]
            if phone.start_frame is not None and phone.end_frame is not None
        ]
        duration_baseline = float(np.median(preceding_durations)) if preceding_durations else None
        endpoint_duration_ratio = (
            endpoint_duration_sec / duration_baseline
            if endpoint_duration_sec is not None and duration_baseline is not None and duration_baseline > 0
            else None
        )
        endpoint_short_duration_score = (
            _sigmoid(
                (self.config.endpoint_duration_ratio_center - endpoint_duration_ratio)
                / self.config.endpoint_duration_ratio_scale
            )
            if endpoint_duration_ratio is not None
            else 0.0
        )
        phone_confidences: list[float] = []
        if full_alignment is not None and full_alignment.phones:
            for phone_idx in range(max(0, n_phones - 2), n_phones):
                state = 2 * phone_idx + 1
                mask = full_alignment.state_path == state
                confidence = (
                    float(np.exp(log_probs_numpy[mask, targets[phone_idx]]).mean()) if bool(np.any(mask)) else 0.0
                )
                phone_confidences.append(confidence)
        last_phone_confidence = phone_confidences[-1] if phone_confidences else None
        last_two_phone_avg_confidence = float(np.mean(phone_confidences)) if phone_confidences else None
        last_phone_gap_sec = None
        if full_alignment is not None and len(full_alignment.phones) >= 2:
            previous_phone = full_alignment.phones[-2]
            final_phone_for_gap = full_alignment.phones[-1]
            if previous_phone.end_frame is not None and final_phone_for_gap.start_frame is not None:
                last_phone_gap_sec = max(
                    0.0,
                    (final_phone_for_gap.start_frame - previous_phone.end_frame - 1) * frame_stride_sec,
                )

        trailing_duration_sec = None
        trail_rms_ratio = None
        eou_cutoff_score = 0.0
        endpoint_noise_score = 0.0
        endpoint_silence_score = 0.0
        if (
            waveform is not None
            and sample_rate is not None
            and sample_rate > 0
            and full_alignment is not None
            and full_alignment.phones
            and full_alignment.phones[-1].end_frame is not None
        ):
            audio = (
                waveform.detach().float().cpu().numpy()
                if isinstance(waveform, torch.Tensor)
                else np.asarray(waveform, dtype=np.float32)
            ).reshape(-1)
            audio_duration_sec = audio.size / sample_rate
            speech_end_sec = (full_alignment.phones[-1].end_frame + 1) * frame_stride_sec
            trailing_duration_sec = max(
                0.0,
                audio_duration_sec - speech_end_sec,
            )
            final_phone_text = reference.phones[-1].casefold()
            is_sibilant = any(phone in final_phone_text for phone in ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ", "ts"))
            trail_start_sec = speech_end_sec + (0.15 if is_sibilant else 0.10)
            trail_start = min(
                audio.size,
                max(0, round(trail_start_sec * sample_rate)),
            )
            trailing_audio = audio[trail_start:]
            full_rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
            trailing_rms = float(np.sqrt(np.mean(np.square(trailing_audio)) + 1e-12)) if trailing_audio.size else 0.0
            trail_rms_ratio = trailing_rms / (full_rms + 1e-10)
            effective_last_confidence = (
                last_two_phone_avg_confidence
                if last_phone_confidence is not None and last_phone_confidence < 0.01
                else last_phone_confidence
            )
            confidence_signal = _sigmoid(
                (
                    self.config.eou_confidence_center
                    - (effective_last_confidence if effective_last_confidence is not None else 1.0)
                )
                / self.config.eou_confidence_scale
            )
            trailing_signal = _sigmoid(
                (self.config.eou_trailing_center_sec - trailing_duration_sec) / self.config.eou_trailing_scale_sec
            )
            gap_signal = _sigmoid(
                (self.config.eou_gap_center_sec - (last_phone_gap_sec if last_phone_gap_sec is not None else 0.0))
                / self.config.eou_gap_scale_sec
            )
            eou_cutoff_score = min(
                confidence_signal,
                trailing_signal,
                gap_signal,
            )
            endpoint_noise_score = max(
                min(
                    _sigmoid((trailing_duration_sec - 0.20) / 0.05),
                    _sigmoid((trail_rms_ratio - 0.40) / 0.10),
                ),
                min(
                    _sigmoid(
                        (
                            (last_phone_gap_sec if last_phone_gap_sec is not None else 0.0)
                            - self.config.eou_gap_center_sec
                        )
                        / self.config.eou_gap_scale_sec
                    ),
                    confidence_signal,
                ),
            )
            endpoint_silence_score = _sigmoid((trailing_duration_sec - 1.40) / 0.20)
        endpoint_bad_region_score = max(
            endpoint_noise_score,
            endpoint_silence_score,
        )
        terminal_window = min(5, log_probs_numpy.shape[0])
        terminal_phone_probabilities = np.exp(log_probs_numpy[-terminal_window:, targets[-1]])
        terminal_blank_probabilities = np.exp(log_probs_numpy[-terminal_window:, blank_id])
        terminal_phone_probability_slope = (
            float((terminal_phone_probabilities[-1] - terminal_phone_probabilities[0]) / (terminal_window - 1))
            if terminal_window > 1
            else 0.0
        )
        terminal_blank_probability_slope = (
            float((terminal_blank_probabilities[-1] - terminal_blank_probabilities[0]) / (terminal_window - 1))
            if terminal_window > 1
            else 0.0
        )
        terminal_probability_rise_score = min(
            _sigmoid(
                (terminal_phone_probability_slope - self.config.terminal_probability_slope_center)
                / self.config.terminal_probability_slope_scale
            ),
            _sigmoid(
                (-terminal_blank_probability_slope - self.config.terminal_probability_slope_center)
                / self.config.terminal_probability_slope_scale
            ),
        )
        terminal_occupancy_entropy = None
        if full_alignment is not None and full_alignment.phones:
            final_state = 2 * n_phones - 1
            occupancy = np.clip(
                full_alignment.state_posteriors[-terminal_window:, final_state],
                1e-8,
                1.0 - 1e-8,
            )
            terminal_occupancy_entropy = float(
                np.mean(-occupancy * np.log(occupancy) - (1.0 - occupancy) * np.log(1.0 - occupancy)) / math.log(2.0)
            )
        nonblank = 1.0 - np.exp(log_probs_numpy[:, blank_id])
        eof_nonblank_probability = float(nonblank[-tail_frames:].mean())
        previous_start = max(0, log_probs.shape[0] - 2 * tail_frames)
        previous_end = max(previous_start + 1, log_probs.shape[0] - tail_frames)
        previous_nonblank = float(nonblank[previous_start:previous_end].mean())
        eof_nonblank_jump = eof_nonblank_probability - previous_nonblank
        eof_discontinuity_score = _sigmoid(
            (eof_nonblank_jump - self.config.eof_discontinuity_center) / self.config.eof_discontinuity_scale
        )

        final_phone_state = 2 * n_phones - 1
        final_blank_state = 2 * n_phones
        original_phone_end = float(forward.final_state_log_probs[final_phone_state])
        original_blank_end = float(forward.final_state_log_probs[final_blank_state])
        terminal_censor_log_odds = _bounded_log_odds(
            original_phone_end,
            original_blank_end,
        )
        appended_terminal_censor_log_odds = None
        appended_completion_log_odds_delta = None
        appended_silence_closure_score = None
        appended_context_confidence_delta = None
        if appended_silence_log_probs is not None:
            appended_numpy = appended_silence_log_probs.detach().float().cpu().numpy()
            if (
                appended_numpy.ndim == 2
                and appended_numpy.shape[1] == log_probs_numpy.shape[1]
                and appended_numpy.shape[0] > log_probs_numpy.shape[0]
                and np.isfinite(appended_numpy).all()
            ):
                appended_forward = ctc_prefix_forward_numpy(
                    appended_numpy,
                    targets,
                    blank_id=blank_id,
                )
                appended_phone_end = float(appended_forward.final_state_log_probs[final_phone_state])
                appended_blank_end = float(appended_forward.final_state_log_probs[final_blank_state])
                appended_terminal_censor_log_odds = _bounded_log_odds(
                    appended_phone_end,
                    appended_blank_end,
                )
                appended_right_censor = (
                    _sigmoid(appended_terminal_censor_log_odds)
                    if appended_terminal_censor_log_odds is not None
                    else 0.0
                )
                original_right_censor = (
                    _sigmoid(terminal_censor_log_odds) if terminal_censor_log_odds is not None else 0.0
                )
                appended_silence_closure_score = max(
                    0.0,
                    min(1.0, original_right_censor - appended_right_censor),
                )
                if terminal_censor_log_odds is not None and appended_terminal_censor_log_odds is not None:
                    appended_completion_log_odds_delta = terminal_censor_log_odds - appended_terminal_censor_log_odds
                shared_tail_start = max(
                    0,
                    log_probs_numpy.shape[0] - terminal_window,
                )
                appended_context_confidence_delta = float(
                    np.exp(
                        appended_numpy[
                            shared_tail_start : log_probs_numpy.shape[0],
                            targets[-1],
                        ]
                    ).mean()
                    - terminal_phone_probabilities.mean()
                )

        partial_final_phone_score = 0.0
        terminal_right_censor = 0.0
        terminal_margin_drop = None
        terminal_touches_eof = 0.0
        if full_alignment is not None and full_alignment.phones:
            full_margins_for_terminal = [phone.acoustic_margin for phone in full_alignment.phones]
            final_phone_alignment = full_alignment.phones[-1]
            final_margin = _finite_mean(
                [final_phone_alignment.acoustic_margin],
                self.config.phone_margin_center,
            )
            preceding_margin = _finite_mean(
                full_margins_for_terminal[max(0, n_phones - window - 1) : n_phones - 1],
                self.config.phone_margin_center,
            )
            terminal_margin_drop = preceding_margin - final_margin
            margin_drop_signal = _sigmoid(
                (terminal_margin_drop - self.config.terminal_margin_drop_center)
                / self.config.terminal_margin_drop_scale
            )
            extension_before = mean(extensions[max(1, n_phones - window) : n_phones])
            extension_drop = extension_before - extensions[n_phones]
            extension_drop_signal = _sigmoid(
                (extension_drop - self.config.posterior_drop_center) / self.config.posterior_drop_scale
            )
            final_phone_end_score = float(forward.final_state_log_probs[final_phone_state])
            final_blank_end_score = float(forward.final_state_log_probs[final_blank_state])
            terminal_right_censor = _sigmoid(terminal_censor_log_odds) if terminal_censor_log_odds is not None else 0.0
            final_end_frame = final_phone_alignment.end_frame
            final_eof_distance = (
                log_probs.shape[0] - 1 - final_end_frame if final_end_frame is not None else log_probs.shape[0]
            )
            terminal_touches_eof = math.exp(-max(final_eof_distance, 0) / max(self.config.eof_scale_frames, 1e-6))
            censor_signal = _sigmoid((terminal_right_censor - 0.999) / 0.002)
            tail_nonblank_signal = _sigmoid((0.95 - tail_blank) / 0.03)
            short_endpoint_signal = _sigmoid(
                (0.08 - (endpoint_occupancy_spread_sec if endpoint_occupancy_spread_sec is not None else 1.0)) / 0.04
            )
            original_partial_final_phone_score = min(
                censor_signal,
                tail_nonblank_signal,
                short_endpoint_signal,
                margin_drop_signal,
            )
            eou_partial_score = min(
                censor_signal,
                margin_drop_signal,
                eou_cutoff_score,
            )
            temporal_partial_score = min(
                censor_signal,
                margin_drop_signal,
                terminal_probability_rise_score,
            )
            partial_final_phone_score = max(
                original_partial_final_phone_score,
                eou_partial_score,
                temporal_partial_score,
            )

        prefix_margins = [phone.acoustic_margin for phone in prefix_alignment.phones]
        before = _finite_mean(
            prefix_margins[max(0, best_k - window) : best_k],
            self.config.phone_margin_center,
        )
        if full_alignment is not None:
            full_margins = [phone.acoustic_margin for phone in full_alignment.phones]
            after_values = full_margins[best_k : min(n_phones, best_k + window)]
            later_values = full_margins[min(n_phones, best_k + window) :]
            after = _finite_mean(after_values, self.config.phone_margin_center)
            recovery = max(
                [float(value) for value in later_values if value is not None and math.isfinite(value)],
                default=after,
            )
            alignment_quality = _finite_mean(
                full_margins[:best_k],
                self.config.phone_margin_center,
            )
            posterior_drop = before - after
            suffix_unsupported = _sigmoid((self.config.phone_margin_center - after) / self.config.phone_margin_scale)
            suffix_non_recovery = _sigmoid(
                (self.config.phone_margin_center - recovery) / self.config.phone_margin_scale
            )
        else:
            extension_before = extension_before_by_k[best_k]
            extension_after = extension_after_by_k[best_k]
            extension_recovery = extension_recovery_by_k[best_k]
            posterior_drop = extension_before - extension_after
            after = None
            recovery = None
            alignment_quality = _finite_mean(
                prefix_margins,
                self.config.phone_margin_center,
            )
            suffix_unsupported = _sigmoid(
                (posterior_drop - self.config.posterior_drop_center) / self.config.posterior_drop_scale
            )
            suffix_non_recovery = _sigmoid(
                (extension_before - extension_recovery - self.config.posterior_drop_center)
                / self.config.posterior_drop_scale
            )

        prefix_support = _sigmoid((before - self.config.phone_margin_center) / self.config.phone_margin_scale)
        prefix_advantage_signal = _sigmoid(
            (prefix_advantage - self.config.prefix_advantage_center) / self.config.prefix_advantage_scale
        )
        posterior_drop_signal = _sigmoid(
            (posterior_drop - self.config.posterior_drop_center) / self.config.posterior_drop_scale
        )

        final_phone = endpoint_phone
        last_frame = final_phone.end_frame
        eof_distance = log_probs.shape[0] - 1 - last_frame if last_frame is not None else log_probs.shape[0]
        proximity = math.exp(-max(eof_distance, 0) / max(self.config.eof_scale_frames, 1e-6))
        phone_state = 2 * best_k - 1
        blank_state = 2 * best_k
        phone_end = float(forward.final_state_log_probs[phone_state])
        blank_end = float(forward.final_state_log_probs[blank_state])
        right_censor = _sigmoid(phone_end - blank_end) if math.isfinite(phone_end) else 0.0
        eof_score = 0.5 * (proximity + right_censor)

        components = {
            "prefix_advantage": prefix_advantage_signal,
            "posterior_drop": posterior_drop_signal,
            "suffix_unsupported": suffix_unsupported,
            "suffix_non_recovery": suffix_non_recovery,
            "eof": eof_score,
        }
        weight_total = sum(max(float(weight), 0.0) for weight in self.config.weights.values())
        generic_raw_score = sum(
            max(float(self.config.weights.get(name, 0.0)), 0.0) * value for name, value in components.items()
        ) / max(weight_total, 1e-12)
        terminal_word_score = max(
            terminal_suffix_score,
            partial_final_phone_score,
        )
        whole_phone_score = max(generic_raw_score, terminal_suffix_score)
        raw_score = max(whole_phone_score, partial_final_phone_score)
        if not math.isfinite(raw_score):
            return self._empty_result(
                reference,
                status="ERROR",
                error={"code": "non_finite_score", "message": "detector score is not finite"},
            )
        whole_phone_cut_score = (
            self.calibration.percentile(
                whole_phone_score,
                channel="whole_phone",
            )
            if self.calibration
            else whole_phone_score
        )
        partial_phone_cut_score = (
            self.calibration.percentile(
                partial_final_phone_score,
                channel="partial_phone",
            )
            if self.calibration
            else partial_final_phone_score
        )
        cut_score = max(whole_phone_cut_score, partial_phone_cut_score)
        review_threshold = (
            self.calibration.review_threshold
            if self.calibration and self.calibration.review_threshold is not None
            else self.config.review_threshold
        )
        drop_threshold = (
            self.calibration.drop_threshold
            if self.calibration and self.calibration.drop_threshold is not None
            else self.config.drop_threshold
        )

        generic_mandatory_gates = (
            prefix_support >= self.config.min_prefix_support_for_drop
            and eof_score >= self.config.min_eof_score_for_drop
            and suffix_unsupported >= self.config.min_suffix_unsupported_for_drop
        )
        terminal_mandatory_gates = (
            terminal_k is not None
            and terminal_suffix_score >= self.config.min_terminal_score_for_drop
            and terminal_prefix_support >= self.config.min_prefix_support_for_drop
            and terminal_eof_score >= self.config.min_eof_score_for_drop
        )
        partial_dominates = partial_phone_cut_score > review_threshold
        partial_drop_gates = (
            self.config.allow_ctc_partial_phone_drop
            and partial_phone_cut_score
            > max(
                drop_threshold,
                self.config.min_partial_score_for_drop,
            )
            and (
                (
                    eof_discontinuity_score >= self.config.min_eof_discontinuity_for_partial_drop
                    and endpoint_short_duration_score >= 0.5
                )
                or eou_cutoff_score >= self.config.min_eou_score_for_partial_drop
            )
            and endpoint_bad_region_score <= self.config.max_endpoint_bad_region_score_for_drop
            and (
                (not self.config.require_appended_silence_for_partial_drop and appended_silence_closure_score is None)
                or appended_silence_closure_score is not None
                and appended_silence_closure_score >= self.config.min_appended_closure_for_partial_drop
            )
        )
        if (
            whole_phone_cut_score > drop_threshold
            and (generic_mandatory_gates or terminal_mandatory_gates)
            and not partial_dominates
        ):
            decision = "DROP"
        elif partial_drop_gates:
            decision = "DROP"
        elif self.config.binary_decisions:
            decision = "KEEP"
        elif cut_score > review_threshold:
            decision = "REVIEW"
        else:
            decision = "KEEP"

        last_phone_idx = best_k - 1
        missing_count = n_phones - best_k
        supported_text, missing_text = reference.text_spans(last_phone_idx)
        word_idx = reference.phone_to_word[last_phone_idx] if last_phone_idx < len(reference.phone_to_word) else None
        word = reference.words[word_idx] if word_idx is not None and 0 <= word_idx < len(reference.words) else None
        phones_in_word = [idx for idx, mapped_word in enumerate(reference.phone_to_word) if mapped_word == word_idx]
        position_inside_word = (
            phones_in_word.index(last_phone_idx) / max(len(phones_in_word), 1)
            if last_phone_idx in phones_in_word
            else None
        )
        cut_time = (last_frame + 1) * frame_stride_sec if last_frame is not None else None
        occupancy_spread = (
            final_phone.posterior_std_frames * frame_stride_sec
            if final_phone.posterior_std_frames is not None
            else None
        )
        flags: list[str] = []
        if not full_feasible:
            flags.append("full_reference_ctc_infeasible")
        if missing_count == 1:
            flags.append("single_missing_phone")
        if partial_dominates:
            flags.append("possible_partial_final_phone")
        if eof_distance > self.config.eof_scale_frames * 2:
            flags.append("candidate_far_from_eof")
        if recovery is not None and after is not None and recovery > after + self.config.posterior_drop_scale:
            flags.append("suffix_support_recovers")

        return {
            "analysis_status": "OK",
            "error": None,
            "decision": decision,
            "is_suspicious": decision in {"REVIEW", "DROP"},
            "cut_score": _round(cut_score),
            "raw_score": _round(raw_score),
            "reference_phone_count": n_phones,
            "estimated_last_supported_phone_index": last_phone_idx,
            "missing_phone_count": missing_count,
            "missing_phone_fraction": _round(missing_count / n_phones),
            "estimated_cut_time_sec": _round(cut_time),
            "estimated_cut_time_uncertainty_sec": None,
            "final_phone_occupancy_spread_sec": _round(occupancy_spread),
            "word_index": word_idx,
            "word": word,
            "phone": reference.phones[last_phone_idx],
            "position_inside_word": _round(position_inside_word),
            "supported_text": supported_text,
            "missing_text": missing_text,
            "prefix_advantage": _round(prefix_advantage),
            "posterior_drop": _round(posterior_drop),
            "prefix_support": _round(prefix_support),
            "suffix_unsupported_score": _round(suffix_unsupported),
            "suffix_recovery_score": _round(1.0 - suffix_non_recovery),
            "eof_score": _round(eof_score),
            "alignment_quality": _round(alignment_quality),
            "tail_blank_probability": _round(tail_blank),
            "ctc_scores": {
                "blank_log_probability": _round(blank_score),
                "full_log_probability": _round(float(scores[-1])),
                "best_prefix_log_probability": _round(float(scores[best_k])),
                "full_normalized": _round(full_normalized),
                "best_prefix_normalized": _round(normalized[best_k]),
            },
            "candidate_count": len(feasible_candidates),
            "quality_flags": flags,
            "score_version": SCORE_VERSION,
            "terminal_candidate_type": (
                "partial_final_phone"
                if partial_dominates
                else (
                    "missing_terminal_phones"
                    if terminal_k is not None and terminal_suffix_score >= generic_raw_score
                    else "generic_prefix"
                )
            ),
            "terminal_word_score": _round(terminal_word_score),
            "terminal_suffix_score": _round(terminal_suffix_score),
            "terminal_suffix_unsupported_score": _round(terminal_suffix_unsupported_signal),
            "partial_final_phone_score": _round(partial_final_phone_score),
            "whole_phone_score": _round(whole_phone_score),
            "whole_phone_cut_score": _round(whole_phone_cut_score),
            "partial_phone_cut_score": _round(partial_phone_cut_score),
            "endpoint_phone_duration_sec": _round(endpoint_duration_sec),
            "endpoint_duration_ratio": _round(endpoint_duration_ratio),
            "endpoint_short_duration_score": _round(endpoint_short_duration_score),
            "eof_nonblank_probability": _round(eof_nonblank_probability),
            "eof_nonblank_jump": _round(eof_nonblank_jump),
            "eof_discontinuity_score": _round(eof_discontinuity_score),
            "appended_silence_closure_score": _round(appended_silence_closure_score),
            "last_phone_confidence": _round(last_phone_confidence),
            "last_two_phone_avg_confidence": _round(last_two_phone_avg_confidence),
            "last_phone_gap_sec": _round(last_phone_gap_sec),
            "trailing_duration_sec": _round(trailing_duration_sec),
            "trail_rms_ratio": _round(trail_rms_ratio),
            "eou_cutoff_score": _round(eou_cutoff_score),
            "endpoint_noise_score": _round(endpoint_noise_score),
            "endpoint_silence_score": _round(endpoint_silence_score),
            "endpoint_bad_region_score": _round(endpoint_bad_region_score),
            "terminal_phone_probability_slope": _round(terminal_phone_probability_slope),
            "terminal_blank_probability_slope": _round(terminal_blank_probability_slope),
            "terminal_probability_rise_score": _round(terminal_probability_rise_score),
            "terminal_occupancy_entropy": _round(terminal_occupancy_entropy),
            "appended_context_confidence_delta": _round(appended_context_confidence_delta),
            "terminal_prefix_phone_count": terminal_k,
            "terminal_missing_phone_count": (n_phones - terminal_k if terminal_k is not None else None),
            "terminal_right_censor_score": _round(terminal_right_censor),
            "terminal_censor_log_odds": _round(terminal_censor_log_odds),
            "terminal_partial_ctc_log_probability": _round(terminal_ctc.partial_log_probability),
            "terminal_complete_ctc_log_probability": _round(terminal_ctc.complete_log_probability),
            "terminal_partial_ctc_log_likelihood_ratio": _round(terminal_ctc.log_likelihood_ratio),
            "terminal_partial_ctc_probability": _round(terminal_ctc.partial_posterior),
            "terminal_partial_ctc_min_complete_frames": (terminal_ctc.min_complete_frames),
            "terminal_partial_ctc_consistency_error": _round(terminal_ctc.consistency_error),
            "appended_terminal_censor_log_odds": _round(appended_terminal_censor_log_odds),
            "appended_completion_log_odds_delta": _round(appended_completion_log_odds_delta),
            "terminal_final_phone_margin_drop": _round(terminal_margin_drop),
            "terminal_final_phone_touches_eof": _round(terminal_touches_eof),
            "components": {name: _round(value) for name, value in components.items()},
            "detector_config": {
                "review_threshold": review_threshold,
                "drop_threshold": drop_threshold,
                "calibrated": self.calibration is not None,
            },
        }

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
