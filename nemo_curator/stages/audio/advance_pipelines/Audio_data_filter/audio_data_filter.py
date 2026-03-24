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
Audio Data Filter Stage -- CompositeStage that decomposes into independent
pipeline stages for extracting clean single-speaker segments.

Pipeline (when all filters + speaker separation enabled)::

    1. MonoConversion (1:1)
    2. VAD batch mode (1:1, items = N segments)
    3. BandFilter (1:1, filter items)
    4. UTMOS (1:1, filter items)
    5. SIGMOS (1:1, filter items)
    6. SegmentConcatenation (1:1, M items -> 1 item + timestamp mappings)
    7. SpeakerSeparation (1:N fan-out)
    8-11. Per-speaker: VAD + Band + UTMOS + SIGMOS
    12. TimestampMapper (1:1, resolve to original file positions)

Usage::

    # Using default config
    pipeline.add_stage(AudioDataFilterStage())

    # Using custom YAML config
    pipeline.add_stage(AudioDataFilterStage(config_path="/path/to/config.yaml"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.filtering import BandFilterStage, SIGMOSFilterStage, UTMOSFilterStage
from nemo_curator.stages.audio.postprocessing import TimestampMapperStage
from nemo_curator.stages.audio.preprocessing import MonoConversionStage, SegmentConcatenationStage
from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage, VADSegmentationStage
from nemo_curator.stages.base import CompositeStage, ProcessingStage

from .config import get_enabled_stages, load_config


class AudioDataFilterStage(CompositeStage):
    """Complete audio data filtering and curation pipeline (CompositeStage).

    Decomposes into independent stages that the executor can schedule with
    cross-file parallelism.  Each stage owns its own default resource
    allocation.  Use ``.with_()`` to override individual stage resources.

    Args:
        config_path: Path to a YAML config file.  When *None* the
            built-in ``default_config.yaml`` is used.
        config: Pre-loaded config dict (alternative to *config_path*).
            When both are given, *config* values override the YAML file.
        name: Name for this composite stage instance.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        name: str = "AudioDataFilter",
    ) -> None:
        super().__init__()
        self.name = name
        self._cfg = load_config(config_path)
        if config:
            from .config import _deep_merge
            self._cfg = _deep_merge(self._cfg, config)

    def decompose(self) -> list[ProcessingStage]:
        cfg = self._cfg
        stages: list[ProcessingStage] = []

        mc = cfg.get("mono_conversion", {})
        stages.append(MonoConversionStage(
            output_sample_rate=mc.get("output_sample_rate", 48000),
            strict_sample_rate=mc.get("strict_sample_rate", True),
            name="MonoConversion",
        ))

        vad = cfg.get("vad", {})
        band = cfg.get("band_filter", {})
        utmos = cfg.get("utmos", {})
        sigmos = cfg.get("sigmos", {})
        speaker = cfg.get("speaker_separation", {})
        concat = cfg.get("concatenation", {})
        ts = cfg.get("timestamp_mapper", {})

        enable_vad = vad.get("enable", True)
        enable_band = band.get("enable", True)
        enable_utmos = utmos.get("enable", True)
        enable_sigmos = sigmos.get("enable", True)
        enable_speaker = speaker.get("enable", True)

        self._append_filter_stages(
            stages, vad, band, utmos, sigmos,
            enable_vad, enable_band, enable_utmos, enable_sigmos,
            suffix="",
        )

        if enable_speaker:
            if enable_vad:
                stages.append(SegmentConcatenationStage(
                    silence_duration_sec=concat.get("silence_duration_sec", 0.5),
                    name="SegmentConcat",
                ))

            stages.append(SpeakerSeparationStage(
                exclude_overlaps=speaker.get("exclude_overlaps", True),
                min_duration=speaker.get("min_duration", 0.8),
                gap_threshold=speaker.get("gap_threshold", 0.1),
                buffer_time=speaker.get("buffer_time", 0.5),
                name="SpeakerSeparation",
            ))

            self._append_filter_stages(
                stages, vad, band, utmos, sigmos,
                enable_vad, enable_band, enable_utmos, enable_sigmos,
                suffix="_Speaker",
            )

        if enable_vad or enable_speaker:
            stages.append(TimestampMapperStage(
                passthrough_keys=ts.get("passthrough_keys"),
                name="TimestampMapper",
            ))

        enabled = get_enabled_stages(cfg)
        logger.info(
            f"AudioDataFilterStage decomposed into {len(stages)} stages "
            f"(enabled: {enabled}, speaker_sep: {enable_speaker})"
        )
        return stages

    @staticmethod
    def _append_filter_stages(
        stages: list[ProcessingStage],
        vad: dict, band: dict, utmos: dict, sigmos: dict,
        enable_vad: bool, enable_band: bool,
        enable_utmos: bool, enable_sigmos: bool,
        *,
        suffix: str,
    ) -> None:
        """Append VAD + quality filter stages to *stages* list."""
        if enable_vad:
            stages.append(VADSegmentationStage(
                mode=vad.get("mode", "batch"),
                min_duration_sec=vad.get("min_duration_sec", 2.0),
                max_duration_sec=vad.get("max_duration_sec", 60.0),
                threshold=vad.get("threshold", 0.5),
                min_interval_ms=vad.get("min_interval_ms", 500),
                speech_pad_ms=vad.get("speech_pad_ms", 300),
                name=f"VAD{suffix}",
            ))

        if enable_band:
            stages.append(BandFilterStage(
                band_value=band.get("band_value", "full_band"),
                name=f"BandFilter{suffix}",
            ))

        if enable_utmos:
            stages.append(UTMOSFilterStage(
                mos_threshold=utmos.get("mos_threshold", 3.5),
                name=f"UTMOS{suffix}",
            ))

        if enable_sigmos:
            stages.append(SIGMOSFilterStage(
                noise_threshold=sigmos.get("noise_threshold", 4.0),
                ovrl_threshold=sigmos.get("ovrl_threshold", 3.5),
                sig_threshold=sigmos.get("sig_threshold"),
                col_threshold=sigmos.get("col_threshold"),
                disc_threshold=sigmos.get("disc_threshold"),
                loud_threshold=sigmos.get("loud_threshold"),
                reverb_threshold=sigmos.get("reverb_threshold"),
                name=f"SIGMOS{suffix}",
            ))
