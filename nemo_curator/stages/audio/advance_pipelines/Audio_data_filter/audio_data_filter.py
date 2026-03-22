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

"""
Audio Data Filter Stage - CompositeStage that decomposes into independent
pipeline stages for extracting clean single-speaker segments.

Pipeline (when all filters + speaker separation enabled):
    1. MonoConversion (1:1)
    2. VAD batch mode (1:1, items=N segments)
    3. BandFilter (1:1, filter items)
    4. UTMOS (1:1, filter items)
    5. SIGMOS (1:1, filter items)
    6. SegmentConcatenation (1:1, M items -> 1 item + timestamp mappings)
    7. SpeakerSeparation (1:N fan-out)
    8-11. Per-speaker: VAD + Band + UTMOS + SIGMOS
    12. TimestampMapper (1:1, resolve to original file positions)

Usage:
    pipeline.add_stage(AudioDataFilterStage(config=config))
"""

from dataclasses import dataclass, field
from typing import Any, List

from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources

from nemo_curator.stages.audio import (
    MonoConversionStage,
    VADSegmentationStage,
    SIGMOSFilterStage,
    UTMOSFilterStage,
    BandFilterStage,
    SpeakerSeparationStage,
    SegmentConcatenationStage,
    TimestampMapperStage,
)
from nemo_curator.stages.audio.configs import (
    VADConfig,
    SIGMOSConfig,
    UTMOSConfig,
    BandFilterConfig,
    SpeakerSeparationConfig,
)

from .config import AudioDataFilterConfig


@dataclass
class AudioDataFilterStage(CompositeStage):
    """
    Complete audio data filtering and curation pipeline (CompositeStage).

    Decomposes into independent stages that the executor can schedule with
    cross-file parallelism. Each stage has its own resource allocation.

    Args:
        config: AudioDataFilterConfig with all pipeline settings.
        gpu_resources: Resources for GPU stages. Defaults to Resources(gpus=1.0).
    """

    config: AudioDataFilterConfig = field(default_factory=AudioDataFilterConfig)
    gpu_resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    name: str = "AudioDataFilter"

    def __post_init__(self):
        super().__init__()

    def decompose(self) -> List[ProcessingStage]:
        cfg = self.config
        gpu_res = self.gpu_resources
        cpu_res = Resources(cpus=1.0)
        band_res = Resources(cpus=4.0)

        stages: List[ProcessingStage] = []

        # 1. Mono conversion (CPU)
        stages.append(MonoConversionStage(
            output_sample_rate=cfg.sample_rate,
            strict_sample_rate=cfg.strict_sample_rate,
            name="MonoConversion"))

        # 2. VAD (batch mode)
        if cfg.enable_vad:
            stages.append(VADSegmentationStage(
                config=VADConfig(min_duration_sec=cfg.vad_min_duration_sec,
                                 max_duration_sec=cfg.vad_max_duration_sec),
                mode="batch", name="VAD").with_(resources=gpu_res))

        # 3. Band filter (CPU-only, sklearn classifier)
        if cfg.enable_band_filter:
            stages.append(BandFilterStage(
                config=BandFilterConfig(band_value=cfg.band_value),
                name="BandFilter").with_(resources=band_res))

        # 4. UTMOS
        if cfg.enable_utmos:
            stages.append(UTMOSFilterStage(
                config=UTMOSConfig(mos_threshold=cfg.utmos_mos_threshold),
                name="UTMOS").with_(resources=gpu_res))

        # 5. SIGMOS
        if cfg.enable_sigmos:
            stages.append(SIGMOSFilterStage(
                config=SIGMOSConfig(noise_threshold=cfg.sigmos_noise_threshold,
                                    ovrl_threshold=cfg.sigmos_ovrl_threshold,
                                    sig_threshold=cfg.sigmos_sig_threshold,
                                    col_threshold=cfg.sigmos_col_threshold,
                                    disc_threshold=cfg.sigmos_disc_threshold,
                                    loud_threshold=cfg.sigmos_loud_threshold,
                                    reverb_threshold=cfg.sigmos_reverb_threshold),
                name="SIGMOS").with_(resources=gpu_res))

        if cfg.enable_speaker_separation:
            # 6. Concatenation (CPU)
            stages.append(SegmentConcatenationStage(
                silence_duration_sec=cfg.silence_duration_ms / 1000.0,
                name="SegmentConcat"))

            # 7. Speaker separation (GPU, fan-out)
            stages.append(SpeakerSeparationStage(
                config=SpeakerSeparationConfig(
                    exclude_overlaps=cfg.speaker_exclude_overlaps),
                name="SpeakerSeparation").with_(resources=gpu_res))

            # 8-11. Per-speaker stages
            if cfg.enable_vad:
                stages.append(VADSegmentationStage(
                    config=VADConfig(min_duration_sec=cfg.vad_min_duration_sec,
                                     max_duration_sec=cfg.vad_max_duration_sec),
                    mode="batch", name="VAD_Speaker").with_(resources=gpu_res))

            if cfg.enable_band_filter:
                stages.append(BandFilterStage(
                    config=BandFilterConfig(band_value=cfg.band_value),
                    name="BandFilter_Speaker").with_(resources=band_res))

            if cfg.enable_utmos:
                stages.append(UTMOSFilterStage(
                    config=UTMOSConfig(mos_threshold=cfg.utmos_mos_threshold),
                    name="UTMOS_Speaker").with_(resources=gpu_res))

            if cfg.enable_sigmos:
                stages.append(SIGMOSFilterStage(
                    config=SIGMOSConfig(noise_threshold=cfg.sigmos_noise_threshold,
                                        ovrl_threshold=cfg.sigmos_ovrl_threshold,
                                        sig_threshold=cfg.sigmos_sig_threshold,
                                        col_threshold=cfg.sigmos_col_threshold,
                                        disc_threshold=cfg.sigmos_disc_threshold,
                                        loud_threshold=cfg.sigmos_loud_threshold,
                                        reverb_threshold=cfg.sigmos_reverb_threshold),
                    name="SIGMOS_Speaker").with_(resources=gpu_res))

        # 12. Timestamp mapper (CPU)
        stages.append(TimestampMapperStage(
            passthrough_keys=cfg.passthrough_keys,
            name="TimestampMapper",
        ))

        logger.info(f"AudioDataFilterStage decomposed into {len(stages)} stages "
                    f"(filters: {cfg.get_enabled_filters()}, "
                    f"speaker_sep: {cfg.enable_speaker_separation})")
        return stages
