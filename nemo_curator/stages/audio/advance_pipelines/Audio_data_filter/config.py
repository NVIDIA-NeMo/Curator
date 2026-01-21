# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Configuration for Audio Data Filter Stage."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Literal, Optional


@dataclass
class AudioDataFilterConfig:
    """
    Configuration for the complete Audio Data Filter pipeline.
    
    This config combines all settings for:
    - Mono conversion
    - VAD segmentation
    - Quality filtering (NISQA, SIGMOS, Band)
    - Speaker separation
    - Timestamp tracking
    - Resource allocation for parallel processing
    
    Resource Allocation:
        cpus: Number of CPU cores for parallel processing
        gpus: Number of GPUs (0.0-1.0 for fractional, >1 for multiple)
        
        When cpus > 1 and gpus == 0: Uses CPU parallel processing (ThreadPoolExecutor)
        When gpus > 0: Uses GPU for model inference
    
    Example:
        # GPU processing (default)
        config = AudioDataFilterConfig(enable_nisqa=True)
        
        # CPU parallel processing with 4 workers
        config = AudioDataFilterConfig(cpus=4.0, gpus=0.0)
    """
    
    # Resource allocation
    cpus: float = 1.0
    gpus: float = 1.0
    
    # General settings
    sample_rate: int = 48000
    strict_sample_rate: bool = True
    
    # VAD settings
    enable_vad: bool = True
    vad_min_duration_sec: float = 2.0
    vad_max_duration_sec: float = 60.0
    vad_min_interval_ms: int = 500
    vad_threshold: float = 0.5
    vad_speech_pad_ms: int = 300
    
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
    speaker_min_duration: float = 0.8
    speaker_gap_threshold: float = 0.1
    speaker_buffer_time: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioDataFilterConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def get_enabled_filters(self) -> List[str]:
        """Return list of enabled filter names."""
        filters = []
        if self.enable_band_filter:
            filters.append("band")
        if self.enable_nisqa:
            filters.append("nisqa")
        if self.enable_sigmos:
            filters.append("sigmos")
        return filters
