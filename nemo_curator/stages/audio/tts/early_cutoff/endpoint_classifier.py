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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


SCALAR_FEATURES = (
    "terminal_partial_ctc_log_likelihood_ratio",
    "terminal_censor_log_odds",
    "appended_completion_log_odds_delta",
    "suffix_non_recovery",
    "last_phone_confidence",
    "last_two_phone_avg_confidence",
    "eou_cutoff_score",
    "endpoint_bad_region_score",
    "terminal_phone_probability_slope",
    "terminal_blank_probability_slope",
    "terminal_occupancy_entropy",
    "appended_silence_closure_score",
    "mfa_alignment_log_likelihood_per_frame",
    "mfa_final_vs_local_loglike",
    "mfa_final_vs_alignment_loglike",
    "mfa_final_phone_duration_sec",
    "mfa_end_to_audio_eof_sec",
    "mfa_max_running_short_interval",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def endpoint_scalar_features(row: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Extract fixed-order scalar values and explicit observed-value masks."""
    prediction = row.get("prediction")
    prediction = prediction if isinstance(prediction, Mapping) else {}
    components = prediction.get("components")
    components = components if isinstance(components, Mapping) else {}
    mfa = row.get("mfa")
    mfa = mfa if isinstance(mfa, Mapping) else {}
    raw = {
        "terminal_partial_ctc_log_likelihood_ratio": prediction.get("terminal_partial_ctc_log_likelihood_ratio"),
        "terminal_censor_log_odds": prediction.get("terminal_censor_log_odds"),
        "appended_completion_log_odds_delta": prediction.get("appended_completion_log_odds_delta"),
        "suffix_non_recovery": components.get("suffix_non_recovery"),
        "last_phone_confidence": prediction.get("last_phone_confidence"),
        "last_two_phone_avg_confidence": prediction.get("last_two_phone_avg_confidence"),
        "eou_cutoff_score": prediction.get("eou_cutoff_score"),
        "endpoint_bad_region_score": prediction.get("endpoint_bad_region_score"),
        "terminal_phone_probability_slope": prediction.get("terminal_phone_probability_slope"),
        "terminal_blank_probability_slope": prediction.get("terminal_blank_probability_slope"),
        "terminal_occupancy_entropy": prediction.get("terminal_occupancy_entropy"),
        "appended_silence_closure_score": prediction.get("appended_silence_closure_score"),
        "mfa_alignment_log_likelihood_per_frame": mfa.get("alignment_log_likelihood_per_frame"),
        "mfa_final_vs_local_loglike": mfa.get("final_vs_local_loglike"),
        "mfa_final_vs_alignment_loglike": mfa.get("final_vs_alignment_loglike"),
        "mfa_final_phone_duration_sec": mfa.get("final_phone_duration_sec"),
        "mfa_end_to_audio_eof_sec": mfa.get("end_to_audio_eof_sec"),
        "mfa_max_running_short_interval": mfa.get("max_running_short_interval"),
    }
    values: list[float] = []
    observed: list[float] = []
    for name in SCALAR_FEATURES:
        value = _finite(raw[name])
        values.append(value if value is not None else 0.0)
        observed.append(1.0 if value is not None else 0.0)
    return values, observed


@dataclass(frozen=True)
class EndpointClassifierConfig:
    acoustic_hidden_size: int
    vocab_size: int
    scalar_feature_count: int = len(SCALAR_FEATURES)
    endpoint_frames: int = 40
    reference_phones: int = 8
    model_width: int = 128
    phone_embedding_size: int = 64
    scalar_width: int = 64
    dropout: float = 0.15

    @property
    def scalar_input_size(self) -> int:
        return 2 * self.scalar_feature_count


class TextConditionedEndpointClassifier(nn.Module):
    """Small endpoint head over frozen acoustic states and reference phones."""

    def __init__(self, config: EndpointClassifierConfig) -> None:
        super().__init__()
        self.config = config
        width = config.model_width
        phone_width = config.phone_embedding_size
        self.acoustic_normalization = nn.LayerNorm(
            config.acoustic_hidden_size,
            elementwise_affine=False,
        )
        self.acoustic_projection = nn.Linear(config.acoustic_hidden_size, width)
        self.phone_embedding = nn.Embedding(
            config.vocab_size,
            phone_width,
            padding_idx=0,
        )
        self.phone_position = nn.Embedding(config.reference_phones, phone_width)
        self.phone_projection = nn.Sequential(
            nn.Linear(2 * phone_width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(config.scalar_input_size, config.scalar_width),
            nn.GELU(),
            nn.LayerNorm(config.scalar_width),
        )
        fused_size = 4 * width + config.scalar_width
        self.output = nn.Sequential(
            nn.Linear(fused_size, width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        acoustic_states: torch.Tensor,
        acoustic_mask: torch.Tensor,
        reference_phones: torch.Tensor,
        reference_mask: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        if acoustic_states.ndim != 3:
            raise ValueError("acoustic_states must have shape [batch, frames, hidden]")
        acoustic_mask = acoustic_mask.bool()
        reference_mask = reference_mask.bool()
        if not torch.all(acoustic_mask.any(dim=1)):
            raise ValueError("every sample must contain at least one acoustic frame")
        if not torch.all(reference_mask.any(dim=1)):
            raise ValueError("every sample must contain at least one reference phone")

        acoustic = self.acoustic_projection(self.acoustic_normalization(acoustic_states.float()))
        positions = torch.arange(
            reference_phones.shape[1],
            device=reference_phones.device,
        )
        phones = self.phone_embedding(reference_phones) + self.phone_position(positions)
        phone_weights = reference_mask.unsqueeze(-1).to(phones.dtype)
        phone_mean = (phones * phone_weights).sum(dim=1) / phone_weights.sum(dim=1)
        last_phone_index = reference_mask.long().sum(dim=1) - 1
        last_phone = phones[
            torch.arange(phones.shape[0], device=phones.device),
            last_phone_index,
        ]
        phone_context = self.phone_projection(torch.cat((phone_mean, last_phone), dim=-1))

        attention_logits = torch.einsum("btd,bd->bt", acoustic, phone_context)
        attention_logits = attention_logits / math.sqrt(acoustic.shape[-1])
        attention_logits = attention_logits.masked_fill(~acoustic_mask, float("-inf"))
        attention = torch.softmax(attention_logits, dim=1)
        attended = torch.einsum("bt,btd->bd", attention, acoustic)

        acoustic_weights = acoustic_mask.unsqueeze(-1).to(acoustic.dtype)
        acoustic_mean = (acoustic * acoustic_weights).sum(dim=1) / acoustic_weights.sum(dim=1)
        last_index = acoustic_mask.long().sum(dim=1) - 1
        acoustic_last = acoustic[
            torch.arange(acoustic.shape[0], device=acoustic.device),
            last_index,
        ]
        acoustic_first = acoustic[:, 0]
        acoustic_slope = acoustic_last - acoustic_first
        scalar_context = self.scalar_projection(scalar_features.float())
        fused = torch.cat(
            (
                attended,
                acoustic_mean,
                acoustic_last,
                acoustic_slope,
                scalar_context,
            ),
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


@dataclass(frozen=True)
class EndpointCheckpoint:
    config: EndpointClassifierConfig
    scalar_mean: tuple[float, ...]
    scalar_scale: tuple[float, ...]
    threshold: float
    target_fpr: float
    score_version: str = "text_conditioned_endpoint_v12"
    masked_scalar_features: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        payload["scalar_features"] = list(SCALAR_FEATURES)
        return payload


def empirical_drop_threshold(
    clean_probabilities: Sequence[float],
    *,
    target_fpr: float,
) -> float:
    """Choose a strict-`>` threshold with empirical FPR no larger than target."""
    if not clean_probabilities:
        raise ValueError("clean_probabilities must not be empty")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    values = sorted(float(value) for value in clean_probabilities)
    allowed = math.floor(len(values) * target_fpr)
    index = max(0, len(values) - allowed - 1)
    return values[index]


def empirical_percentile(value: float, null_values: Sequence[float]) -> float:
    if not null_values:
        raise ValueError("null_values must not be empty")
    return bisect.bisect_left(null_values, float(value)) / len(null_values)


def save_endpoint_checkpoint(
    path: Path,
    model: TextConditionedEndpointClassifier,
    checkpoint: EndpointCheckpoint,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": checkpoint.metadata(),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_endpoint_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TextConditionedEndpointClassifier, EndpointCheckpoint]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    metadata = payload["metadata"]
    config = EndpointClassifierConfig(**metadata["config"])
    checkpoint = EndpointCheckpoint(
        config=config,
        scalar_mean=tuple(float(value) for value in metadata["scalar_mean"]),
        scalar_scale=tuple(float(value) for value in metadata["scalar_scale"]),
        threshold=float(metadata["threshold"]),
        target_fpr=float(metadata["target_fpr"]),
        score_version=str(metadata["score_version"]),
        masked_scalar_features=tuple(metadata.get("masked_scalar_features", ())),
    )
    model = TextConditionedEndpointClassifier(config)
    model.load_state_dict(payload["state_dict"])
    return model, checkpoint
