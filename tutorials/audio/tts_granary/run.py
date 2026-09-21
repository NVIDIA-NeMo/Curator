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

"""TTS Granary annotation pipeline over JSONL manifests with ``tn_raw``.

Reads JSONL whose rows carry inverse-text-normalized transcript ``tn_raw``
(or ``itn_text``). Annotates rows with the same sequence as the ameister
TTS Granary pipeline, then writes a JSONL manifest.

Architecture (flags control optional GPU/audio stages)::

    ManifestReader (CPU)
        → JSONL with tn_raw / itn_text
    [if --enable_speaker_id] InferenceSortformerStage (GPU)
        → diar_segments  (TTS Granary speaker_id)
    [if --enable_mos] UTMOSFilterStage (GPU, mos_threshold=None)
        → utmos_mos  (TTS Granary mos; annotation only, no drop)
    [if --enable_bandwidth] BandwidthAnnotationStage (CPU)
        → bandwidth dict  (TTS Granary bandwidth)
    [if --enable_sed] SEDInferenceStage + SEDPostprocessingStage (GPU+CPU)
        → sed_events  (TTS Granary sound_event_detection)
    [unless --disable_ipa] ManifestIpaStage (CPU)
        → ipa from tn_raw via espeak-ng  (TTS Granary manifest_ipa)
    ManifestWriterStage (CPU)

Requires ``espeak-ng`` on PATH for IPA. Audio stages need ``audio_filepath``
or in-memory ``waveform`` + ``sample_rate`` on each row.

Usage (from the Curator repo root)::

    python tutorials/audio/tts_granary/run.py \\
        --input_manifest /path/to/text_pipeline.jsonl \\
        --output_manifest /path/to/tts_out.jsonl
"""

from __future__ import annotations

import argparse

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader, ManifestWriterStage
from nemo_curator.stages.audio.tts.bandwidth import BandwidthAnnotationStage
from nemo_curator.stages.audio.tts.ipa import ManifestIpaStage

_PANNS_ADAPTER = "nemo_curator.models.sed.panns.PANNsSEDAdapter"


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TTS Granary annotation pipeline over JSONL (tn_raw).")
    ap.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="JSONL manifest(s) with tn_raw (or itn_text). File, directory, or glob.",
    )
    ap.add_argument("--output_manifest", type=str, required=True, help="Output JSONL path.")
    ap.add_argument(
        "--text_key",
        type=str,
        default="tn_raw",
        help="Input text field. itn_text / GranaryV2.tn_raw are automatic fallbacks.",
    )
    ap.add_argument(
        "--ipa_output_key",
        type=str,
        default="ipa",
        help="Where to write the IPA annotation dict {ipa, error}. Dotted paths allowed.",
    )
    ap.add_argument("--source_lang_key", type=str, default="source_lang")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force espeak voice for IPA (e.g. en, de). Default: per-row source_lang.",
    )
    ap.add_argument("--overwrite_ipa", action="store_true", default=False)
    ap.add_argument(
        "--disable_ipa",
        action="store_true",
        default=False,
        help="Skip ManifestIpaStage (enabled by default; this is the tn_raw consumer).",
    )
    ap.add_argument(
        "--enable_speaker_id",
        action="store_true",
        default=False,
        help="Streaming Sortformer diarization (TTS Granary speaker_id). Needs audio.",
    )
    ap.add_argument(
        "--enable_mos",
        action="store_true",
        default=False,
        help="UTMOS scoring without dropping rows (TTS Granary mos). Needs audio.",
    )
    ap.add_argument(
        "--enable_bandwidth",
        action="store_true",
        default=False,
        help="CPU bandwidth estimate (TTS Granary bandwidth). Needs audio.",
    )
    ap.add_argument(
        "--enable_sed",
        action="store_true",
        default=False,
        help="PANNs CNN14 sound-event detection (TTS Granary SED). Needs audio.",
    )
    ap.add_argument(
        "--sed_checkpoint",
        type=str,
        default=None,
        help="Optional PANNs .pth path. When omitted, the adapter uses its default cache file.",
    )
    ap.add_argument(
        "--sortformer_model",
        type=str,
        default="nvidia/diar_streaming_sortformer_4spk-v2.1",
        help="Sortformer model id or local .nemo path.",
    )
    ap.add_argument("--bandwidth_output_key", type=str, default="bandwidth")
    ap.add_argument("--executor", type=str, default="xenna", choices=["xenna", "ray_data"])
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    enable_ipa = not args.disable_ipa
    audio_stages = any((args.enable_speaker_id, args.enable_mos, args.enable_bandwidth, args.enable_sed))
    if not enable_ipa and not audio_stages:
        logger.warning("No stages enabled. IPA is on by default; pass at least one --enable_* or omit --disable_ipa.")
        return

    stages: list = [ManifestReader(manifest_path=args.input_manifest)]

    if args.enable_speaker_id:
        from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage

        stages.append(InferenceSortformerStage(model_name=args.sortformer_model))
        logger.info("speaker_id enabled: InferenceSortformerStage → diar_segments")

    if args.enable_mos:
        from nemo_curator.stages.audio.filtering import UTMOSFilterStage

        stages.append(UTMOSFilterStage(mos_threshold=None))
        logger.info("mos enabled: UTMOSFilterStage(mos_threshold=None) → utmos_mos (no drop)")

    if args.enable_bandwidth:
        stages.append(BandwidthAnnotationStage(output_key=args.bandwidth_output_key))
        logger.info(f"bandwidth enabled: BandwidthAnnotationStage → {args.bandwidth_output_key}")

    if args.enable_sed:
        from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
        from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage

        stages.append(
            SEDInferenceStage(
                adapter_target=_PANNS_ADAPTER,
                checkpoint_path=args.sed_checkpoint,
            )
        )
        stages.append(SEDPostprocessingStage())
        logger.info("sed enabled: SEDInferenceStage + SEDPostprocessingStage → sed_events")

    if enable_ipa:
        stages.append(
            ManifestIpaStage(
                text_key=args.text_key,
                output_key=args.ipa_output_key,
                source_lang_key=args.source_lang_key,
                language=args.language,
                overwrite=args.overwrite_ipa,
            )
        )
        logger.info(f"IPA enabled: {args.text_key} (+ itn_text / GranaryV2.tn_raw fallback) → {args.ipa_output_key}")

    stages.append(ManifestWriterStage(output_path=args.output_manifest))

    pipeline = Pipeline(name="tts_granary_annotation", stages=stages)
    if args.executor == "ray_data":
        from nemo_curator.backends.ray_data import RayDataExecutor

        executor = RayDataExecutor()
    else:
        executor = XennaExecutor()

    logger.info(f"Running TTS pipeline: {len(stages)} stages, executor={args.executor}")
    pipeline.run(executor=executor)
    logger.info("TTS pipeline complete.")


if __name__ == "__main__":
    main()
