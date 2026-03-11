# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Configuration for Audio Data Filter Stage."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Literal, Optional, Tuple

SUPPORTED_AUDIO_FORMATS: Tuple[str, ...] = (
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".webm",
)

DEFAULT_OUTPUT_FORMAT: str = "wav"


@dataclass
class AudioDataFilterConfig:
    """
    Configuration for the complete Audio Data Filter pipeline.

    Example:
        config = AudioDataFilterConfig(enable_nisqa=True)
    """

    # General
    sample_rate: int = 48000
    strict_sample_rate: bool = True
    output_format: str = DEFAULT_OUTPUT_FORMAT

    # VAD
    enable_vad: bool = True
    vad_min_duration_sec: float = 2.0
    vad_max_duration_sec: float = 60.0

    # Concatenation
    silence_duration_ms: int = 500

    # Band Filter
    enable_band_filter: bool = True
    band_value: Literal["full_band", "narrow_band"] = "full_band"

    # NISQA Filter
    enable_nisqa: bool = True
    nisqa_mos_threshold: Optional[float] = 4.5
    nisqa_noi_threshold: Optional[float] = 4.3
    nisqa_col_threshold: Optional[float] = None
    nisqa_dis_threshold: Optional[float] = None
    nisqa_loud_threshold: Optional[float] = None

    # SIGMOS Filter
    enable_sigmos: bool = True
    sigmos_noise_threshold: Optional[float] = 4.0
    sigmos_ovrl_threshold: Optional[float] = 3.5
    sigmos_sig_threshold: Optional[float] = None
    sigmos_col_threshold: Optional[float] = None
    sigmos_disc_threshold: Optional[float] = None
    sigmos_loud_threshold: Optional[float] = None
    sigmos_reverb_threshold: Optional[float] = None

    # Speaker Separation
    enable_speaker_separation: bool = True
    speaker_exclude_overlaps: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioDataFilterConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def get_enabled_filters(self) -> List[str]:
        filters = []
        if self.enable_band_filter:
            filters.append("band")
        if self.enable_nisqa:
            filters.append("nisqa")
        if self.enable_sigmos:
            filters.append("sigmos")
        return filters
