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

"""GranaryHiFi: turn Granary ASR manifests into contextual TTS training shards.

Implements every step of the GranaryHiFi pipeline (ICLR submission "GranaryHiFi: A 203K-Hour Open Dataset for
Contextual Speech Synthesis in 24 European Languages", Sec. 4) as two Curator pipelines.

``annotate`` — one JSONL row per source recording (``orig_sample_id``, ``orig_sample_rate``, ``segments``), segment
audio addressed as tar members (``GranaryHifi.AudioMetadata``)::

    ManifestReader
    4.2 SegmentDiarizationStage      streaming Sortformer -> diarization.{num_speakers, diarized_speech_rate}
        SpeakerClusteringStage       TitaNet + agglomerative clustering (0.4) -> speaker_cluster.{speaker_id, ...}
        SegmentSEDStage              CNN14 framewise + GBM postprocessor -> sed.{weighted_*}
        UTMOSv2AnnotationStage       -> utmosv2.score
        BandwidthAnnotationStage     -> bandwidth.bandwidth
        [StudioVoiceEnhancementStage] Maxine StudioVoice via Triton (optional, --studio_voice_url)
    4.3 SegmentFilterStage           common filter, per-source preset (drops failing segments)
    4.4 ContextSelectionStage        same-speaker context: >=3 s, similarity >=0.6, text overlap <=0.5
    4.6 ManifestIpaStage             espeak IPA from tn_raw
    4.5 EarlyCutOffEndpointGateStage wav2vec2 CTC + endpoint heads (proposes candidates)
        EarlyCutOffMFAResolverStage  MFA resolves candidates -> OK / DROP / REVIEW
    4.6 TrainingFlagsStage           valid_tn_raw_transcript, valid_ipa, valid_context_audio_text, early_cut_off
    ManifestWriterStage

``build_shar`` — one task per annotated manifest file (Sec. 4.6, 4.1)::

    LhotseSharBuildStage   cuts / target_audio / context_audio SHAR shards (<= 4096 cuts, speaker-packed)
    CodecCodesStage        25 fps spectral codec (BWE) target/context codes
    SharValidationStage    count / id / tar / cross-stream alignment checks
    publish_shars          global renumbering + hard-link publication (--publish_root)

Steps can be skipped with ``--skip`` (e.g. when a manifest already carries an annotation). The early cut-off branch
needs the v13 endpoint bundle (``--early_cut_off_bundle_dir``) and Montreal Forced Aligner (``--mfa_command``).

Usage (from the Curator repo root)::

    python tutorials/audio/granary_hifi/run.py annotate --preset ytc \\
        --input_manifest in.jsonl --output_manifest annotated.jsonl \\
        --sed_classifier gbt_multiclass_model.joblib --early_cut_off_bundle_dir v13/
    python tutorials/audio/granary_hifi/run.py build_shar \\
        --input_manifest annotated.jsonl --shar_dir shars/ --dataset "GranaryHiFi_YTC (orig_audio)" --language it \\
        --codec_model 25fps_spectral_codec_with_bandwidth_extension.nemo --publish_root published/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader, ManifestWriterStage

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage

ANNOTATE_STEPS = (
    "enhancement",
    "diarization",
    "speaker_clustering",
    "sed",
    "utmos",
    "bandwidth",
    "filter",
    "context",
    "ipa",
    "early_cut_off",
    "mfa_resolver",
    "training_flags",
)
TN_RAW = "GranaryV2.tn_raw"


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--input_manifest", required=True, help="JSONL file, directory or glob.")
    ap.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    ap.add_argument(
        "--ray_temp_dir",
        default=None,
        help="Private Ray temp dir (keep it short). Isolates this run from other Ray clusters sharing /tmp/ray.",
    )
    ap.add_argument(
        "--execution_mode",
        default="streaming",
        choices=["streaming", "batch"],
        help="Xenna mode. 'batch' runs stages one after another (needed when the GPU stages don't all fit at once, "
        "e.g. a 2-GPU workstation). Xenna counts GPUs from CUDA_VISIBLE_DEVICES, so set it (e.g. 0,1) locally.",
    )
    ap.add_argument("--tar_root", default=None, help="Root for relative GranaryHifi.AudioMetadata.tar_key values.")
    ap.add_argument("--strip_tar_key_prefix", action="append", default=[], help="Prefix removed from tar_key.")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    an = sub.add_parser("annotate", help="Annotate, filter and select contexts (Sec. 4.2-4.6).")
    _add_common(an)
    an.add_argument("--output_manifest", required=True)
    an.add_argument("--preset", default="default", choices=["default", "ytc", "yodas", "voxpopuli", "librilight"])
    an.add_argument("--skip", nargs="*", default=[], choices=ANNOTATE_STEPS, help="Steps to leave out.")
    an.add_argument("--source_lang_key", default="source_lang")
    an.add_argument("--language", default=None, help="Force the language (espeak voice, filter charset, MFA).")
    an.add_argument("--sortformer_model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    an.add_argument(
        "--sortformer_path", default=None, help="Local .nemo (reference: diar_streaming_sortformer_4spk-v2)."
    )
    an.add_argument("--titanet_path", default=None, help="Local speakerverification_en_titanet_large.nemo.")
    an.add_argument(
        "--sed_checkpoint", default=None, help="Cnn14_DecisionLevelMax_mAP=0.385.pth (downloaded if unset)."
    )
    an.add_argument("--sed_classifier", default=None, help="gbt_multiclass_model.joblib (GBM postprocessor).")
    an.add_argument("--utmos_checkpoint", default=None, help="UTMOSv2 checkpoint (e.g. fold0_s42_best_model.pth).")
    an.add_argument(
        "--enhancement",
        default="none",
        choices=["none", "studio_voice", "reuse"],
        help="Speech enhancement: Maxine StudioVoice (Triton) or ReUse (Resemble-Enhance). Writes enhanced_audio.",
    )
    an.add_argument("--enhanced_dir", default="enhanced_audio", help="Where enhanced WAVs are written.")
    an.add_argument("--studio_voice_url", default=None, help="StudioVoice Triton gRPC host:port.")
    an.add_argument("--resemble_python", default=None, help="Python of the resemble-enhance venv (torch 2.1).")
    an.add_argument("--resemble_run_dir", default=None, help="Pre-staged resemble enhancer_stage2 weights dir.")
    an.add_argument("--resemble_nfe", type=int, default=32)
    an.add_argument(
        "--enhanced_fallback", action="store_true", help="Use the original audio where no enhanced copy exists."
    )
    an.add_argument("--early_cut_off_bundle_dir", default=None, help="v13 endpoint bundle (gate + MFA thresholds).")
    an.add_argument("--mfa_command", default="mfa", help="Montreal Forced Aligner 3.4.1 executable.")
    an.add_argument("--mfa_root_dir", default=None, help="MFA_ROOT_DIR with downloaded acoustic models/dicts.")
    an.add_argument("--mfa_num_jobs", type=int, default=None, help="MFA parallel jobs (default: stage CPUs).")
    an.add_argument("--utmos_repetitions", type=int, default=1, help="UTMOSv2 random-crop repetitions to average.")

    sh = sub.add_parser("build_shar", help="Materialize Lhotse SHAR, codec codes, validate, publish (Sec. 4.6).")
    _add_common(sh)
    sh.add_argument("--shar_dir", required=True, help="Working directory for per-manifest shard sets.")
    sh.add_argument("--preset", default="ytc", choices=["ytc", "yodas", "voxpopuli", "librilight"])
    sh.add_argument(
        "--dataset", default=None, help="Speaker-string dataset (default: GranaryHiFi_<PRESET> (orig_audio))."
    )
    sh.add_argument("--language", default=None, help="Overrides the row source_lang; required if rows have none.")
    sh.add_argument("--max_cuts_per_shard", type=int, default=4096)
    sh.add_argument("--codec_model", default=None, help="25fps_spectral_codec_with_bandwidth_extension.nemo")
    sh.add_argument("--skip_codec", action="store_true")
    sh.add_argument("--enhanced_fallback", action="store_true", help="Original audio where no enhanced copy exists.")
    sh.add_argument("--publish_root", default=None, help="Hard-link published layout here (same filesystem).")
    return ap


def _abs(path: str | None) -> str | None:
    return str(Path(path).expanduser().resolve()) if path else None


def _check_annotate_args(args: argparse.Namespace, want: list[str]) -> None:
    """Refuse configurations that would silently drop or mislabel the whole dataset."""
    problems = []
    if "sed" in want and "filter" in want and not args.sed_classifier:
        problems.append(
            "--sed_classifier is required: without the GBM, weighted_classifier_confidence is 0 and the "
            "common filter drops every segment (or --skip sed filter)"
        )
    if ("early_cut_off" in want or "mfa_resolver" in want) and not args.early_cut_off_bundle_dir:
        problems.append(
            "--early_cut_off_bundle_dir is required for the end-of-word check; without it every segment "
            "would be flagged clean (or --skip early_cut_off mfa_resolver)"
        )
    if "enhancement" in want and args.enhancement == "studio_voice" and not args.studio_voice_url:
        problems.append("--enhancement studio_voice needs --studio_voice_url (a running StudioVoice Triton server)")
    if "mfa_resolver" in want and "early_cut_off" not in want:
        problems.append("mfa_resolver resolves the gate's candidates: it needs the early_cut_off step")
    if problems:
        raise SystemExit("annotate: " + "; ".join(problems))


def _annotate_stages(args: argparse.Namespace) -> list[ProcessingStage]:  # noqa: C901, PLR0912
    from nemo_curator.stages.audio import tts

    source = {"tar_root": _abs(args.tar_root), "strip_tar_key_prefixes": tuple(args.strip_tar_key_prefix)}
    # One switch: with enhancement on, every later stage reads the enhanced copy; otherwise the original audio.
    audio_source = "original" if args.enhancement == "none" else "enhanced"
    audio = {**source, "audio_source": audio_source, "enhanced_fallback": args.enhanced_fallback}
    want = [s for s in ANNOTATE_STEPS if s not in set(args.skip)]
    if args.enhancement == "none" and "enhancement" in want:
        want.remove("enhancement")
    if args.preset == "librilight":  # LibriLight carries a sample-level speaker_id (no clustering)
        want = [s for s in want if s != "speaker_clustering"]
    _check_annotate_args(args, want)
    stages: list[ProcessingStage] = [ManifestReader(manifest_path=args.input_manifest)]
    for step in want:
        if step == "enhancement":
            if args.enhancement == "studio_voice":
                stages.append(
                    tts.StudioVoiceEnhancementStage(
                        server_url=args.studio_voice_url, output_dir=_abs(args.enhanced_dir), **source
                    )
                )
            else:
                stages.append(
                    tts.ResembleEnhanceStage(
                        output_dir=_abs(args.enhanced_dir),
                        enhancer_python=args.resemble_python,
                        run_dir=_abs(args.resemble_run_dir),
                        nfe=args.resemble_nfe,
                        **source,
                    )
                )
        elif step == "diarization":
            stages.append(
                tts.SegmentDiarizationStage(model_name=args.sortformer_model, model_path=args.sortformer_path, **audio)
            )
        elif step == "speaker_clustering":
            stages.append(tts.SpeakerClusteringStage(model_path=args.titanet_path, **audio))
        elif step == "sed":
            stages.append(
                tts.SegmentSEDStage(
                    checkpoint_path=args.sed_checkpoint,
                    classifier_path=args.sed_classifier,
                    **audio,
                )
            )
        elif step == "utmos":
            stages.append(
                tts.UTMOSv2AnnotationStage(
                    checkpoint_path=args.utmos_checkpoint, num_repetitions=args.utmos_repetitions, **audio
                )
            )
        elif step == "bandwidth":
            stages.append(tts.BandwidthAnnotationStage(**audio))
        elif step == "filter":
            stages.append(tts.SegmentFilterStage(preset=args.preset, language=args.language))
        elif step == "context":
            scope = "sample" if args.preset == "librilight" else "segment"
            speaker_key = "speaker_id" if scope == "sample" else "speaker_cluster.speaker_id"
            stages.append(
                tts.ContextSelectionStage(
                    model_path=args.titanet_path, speaker_id_key=speaker_key, speaker_id_scope=scope, **audio
                )
            )
        elif step == "ipa":
            stages.append(
                tts.ManifestIpaStage(text_key=TN_RAW, source_lang_key=args.source_lang_key, language=args.language)
            )
        elif step == "early_cut_off":
            stages.append(
                tts.EarlyCutOffEndpointGateStage(
                    text_key=TN_RAW,
                    source_lang_key=args.source_lang_key,
                    language=args.language,
                    bundle_dir=_abs(args.early_cut_off_bundle_dir),
                    **audio,
                )
            )
        elif step == "mfa_resolver":
            mfa_kwargs = {"num_jobs": args.mfa_num_jobs} if args.mfa_num_jobs else {}
            stages.append(
                tts.EarlyCutOffMFAResolverStage(
                    text_key=TN_RAW,
                    bundle_dir=_abs(args.early_cut_off_bundle_dir),
                    mfa_command=args.mfa_command,
                    mfa_root_dir=_abs(args.mfa_root_dir),
                    **mfa_kwargs,
                    **audio,
                )
            )
        elif step == "training_flags":
            stages.append(tts.TrainingFlagsStage(text_key=TN_RAW))
    stages.append(ManifestWriterStage(output_path=args.output_manifest))
    return stages


def _manifest_audio_source(manifest: str) -> str:
    """'enhanced' if the annotate run enhanced the audio (segments carry ``enhanced_audio``), else 'original'.

    Keeps build_shar consistent with annotate: the shards contain exactly the audio the annotations were made on.
    """
    import glob
    import json

    paths = sorted(glob.glob(str(Path(manifest) / "*.jsonl")) if Path(manifest).is_dir() else glob.glob(manifest))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line) if line.strip() else {}
                for segment in row.get("segments") or [row]:
                    if isinstance(segment, dict) and "enhanced_audio" in segment:
                        return "enhanced"
                if row:
                    return "original"
    return "original"


_DATASET_NAMES = {"ytc": "YTC", "yodas": "YODAS", "voxpopuli": "VoxPopuli", "librilight": "LibriLight"}


def _build_shar_stages(args: argparse.Namespace) -> list[ProcessingStage]:
    from nemo_curator.stages.audio import tts
    from nemo_curator.stages.file_partitioning import FilePartitioningStage

    if not args.skip_codec and not args.codec_model:
        msg = "build_shar: --codec_model is required (or --skip_codec)"
        raise SystemExit(msg)
    stages: list[ProcessingStage] = [
        FilePartitioningStage(file_paths=args.input_manifest, files_per_partition=1, file_extensions=[".jsonl"]),
        tts.LhotseSharBuildStage(
            output_dir=_abs(args.shar_dir),
            dataset=args.dataset or f"GranaryHiFi_{_DATASET_NAMES[args.preset]} (orig_audio)",
            language=args.language,
            preset="librilight" if args.preset == "librilight" else "default",
            max_samples_per_shard=args.max_cuts_per_shard,
            tar_root=_abs(args.tar_root),
            strip_tar_key_prefixes=tuple(args.strip_tar_key_prefix),
            audio_source=_manifest_audio_source(args.input_manifest),
            enhanced_fallback=args.enhanced_fallback,
        ),
    ]
    if not args.skip_codec:
        stages.append(tts.CodecCodesStage(model_path=_abs(args.codec_model)))
    stages.append(tts.SharValidationStage(require_codes=not args.skip_codec))
    return stages


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.command == "annotate":
        stages, name = _annotate_stages(args), "granary_hifi_annotate"
    else:
        stages, name = _build_shar_stages(args), "granary_hifi_build_shar"

    if args.executor == "ray_data":
        from nemo_curator.backends.ray_data import RayDataExecutor

        executor = RayDataExecutor()
    else:
        from nemo_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor(config={"execution_mode": args.execution_mode})

    from nemo_curator.core.client import RayClient

    logger.info(f"{name}: {[s.name for s in stages]}")
    ray_client = RayClient(**({"ray_temp_dir": args.ray_temp_dir} if args.ray_temp_dir else {}))
    ray_client.start()
    try:
        results = Pipeline(name=name, stages=stages).run(executor=executor)
    finally:
        ray_client.stop()

    if args.command == "build_shar" and args.publish_root:
        from nemo_curator.stages.audio.tts.shar_validation import PublishLayout, publish_shars

        registries = sorted({path for task in results or [] for path in task.data})
        summary = publish_shars(
            registries,
            _abs(args.publish_root),
            PublishLayout.for_dataset(args.preset),
            require_codes=not args.skip_codec,
        )
        logger.info(f"published: {summary}")
    logger.info(f"{name} complete.")


if __name__ == "__main__":
    main()
