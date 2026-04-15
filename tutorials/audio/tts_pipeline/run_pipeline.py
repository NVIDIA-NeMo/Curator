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

"""End-to-end TTS data curation pipeline.

Chains all stages::

    SED -> SED postprocess -> segment extract -> diarize ->
    transcribe (3-pass cascade) -> speaker embed -> speaker cluster ->
    utmos score

Usage::

    python run_pipeline.py \\
        --input_manifest /data/manifest.jsonl \\
        --stages sed,sed_post,segment,diarize,transcribe,embed,cluster,utmos \\
        --language Ru \\
        --sed_checkpoint /models/Cnn14_DecisionLevelMax.pth \\
        --vllm_host localhost --vllm_port 8200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import ray

try:
    from nemo_curator.backends.ray_data import RayDataExecutor
except ImportError:
    from nemo_curator.backends.experimental.ray_data import RayDataExecutor

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end TTS data curation pipeline.")

    p.add_argument("--input_manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="output/tts_pipeline/")

    p.add_argument(
        "--stages", type=str,
        default="sed,sed_post,segment,diarize,transcribe,embed,cluster,utmos",
        help="Comma-separated: sed,sed_post,segment,diarize,transcribe,embed,cluster,utmos",
    )

    # SED
    p.add_argument("--sed_checkpoint", type=str, default="")
    p.add_argument("--sed_model_type", type=str, default="Cnn14_DecisionLevelMax")
    p.add_argument("--sed_threshold", type=float, default=0.5)

    # Transcription
    p.add_argument("--language", type=str, default="Ru")
    p.add_argument("--vllm_host", type=str, default="localhost")
    p.add_argument("--vllm_port", type=int, default=8200)
    p.add_argument("--omni_model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--llm_model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct")
    p.add_argument("--api_key", type=str, default="dummy-key")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)

    # UTMOS
    p.add_argument("--utmos_batch_size", type=int, default=16, help="Batch size for UTMOSv2 scoring")

    # Speaker ID
    p.add_argument("--speaker_model", type=str, default="nvidia/speakerverification_en_titanet_large")
    p.add_argument("--cluster_threshold", type=float, default=0.292)
    p.add_argument("--cluster_batch_size", type=int, default=None,
                   help="Shards per clustering group (None=global, 1=per-shard, N=grouped)")

    p.add_argument("--no_ray_local", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)

    return p.parse_args()


def _run_audio_stage_on_manifest(
    manifest_path: str,
    stage,
    output_path: str,
    extra_keys: list[str] | None = None,
) -> None:
    """Run an AudioTask-based stage directly on manifest rows, bypassing Ray pipeline.

    Some stages (Sortformer, SpeakerEmbedding) expect AudioTask but the JSONL
    reader produces DocumentBatch. This helper reads the manifest, wraps each
    row as an AudioTask, runs the stage, and writes back to JSONL.
    """
    from nemo_curator.tasks import AudioTask

    rows = []
    if os.path.isdir(manifest_path):
        for fname in sorted(os.listdir(manifest_path)):
            if fname.endswith(".jsonl"):
                with open(os.path.join(manifest_path, fname)) as f:
                    rows.extend(json.loads(line) for line in f if line.strip())
    else:
        with open(manifest_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    if hasattr(stage, "setup_on_node"):
        stage.setup_on_node()
    stage.setup()
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "output.jsonl")
    with open(out_file, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            task = AudioTask(task_id=str(i), dataset_name="e2e", data=row)
            try:
                result = stage.process(task)
                out_row = dict(result.data)
            except Exception as e:
                print(f"  [{i}] Error: {e}")
                out_row = dict(row)
                out_row["stage_error"] = str(e)
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    stage.teardown()
    print(f"  Wrote {len(rows)} rows -> {out_file}")


def run_transcription_cascade(
    input_manifest: str,
    output_dir: str,
    args: argparse.Namespace,
) -> str:
    """Run 3-pass transcription cascade: ASR -> verify -> PnC.

    Returns path to the output manifest with all 3 text fields.
    """
    from nemo_curator.stages.audio.request.prompt_template import load_prompt_config
    from nemo_curator.stages.audio.request.transcription_cascade import (
        TranscriptionCascadeConfig,
        get_prompt_path,
        run_cascade_on_row,
    )

    cfg = TranscriptionCascadeConfig(language=args.language)
    p1_path, p2_path, p3_path = cfg.resolve_paths()
    p1_cfg = load_prompt_config(p1_path)
    p2_cfg = load_prompt_config(p2_path)
    p3_cfg = load_prompt_config(p3_path)

    base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"

    try:
        from openai import OpenAI
        omni_client = OpenAI(base_url=base_url, api_key=args.api_key)
    except ImportError:
        print("ERROR: pip install openai required for transcription cascade")
        sys.exit(1)

    def _convert_audio_parts(messages):
        """Convert YAML-style audio parts to OpenAI audio_url data URIs."""
        import base64 as b64mod
        import copy

        converted = copy.deepcopy(messages)
        for msg in converted:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_parts = []
            for part in content:
                if part.get("type") == "audio" and "audio" in part:
                    audio_path = part["audio"]
                    if os.path.exists(audio_path):
                        with open(audio_path, "rb") as af:
                            audio_b64 = b64mod.standard_b64encode(af.read()).decode("ascii")
                        ext = os.path.splitext(audio_path)[1].lower()
                        mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".opus": "audio/opus", ".flac": "audio/flac"}
                        mime = mime_map.get(ext, "audio/wav")
                        new_parts.append({
                            "type": "audio_url",
                            "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
                        })
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            msg["content"] = new_parts
        return converted

    def query_omni(messages):
        api_messages = _convert_audio_parts(messages)
        resp = omni_client.chat.completions.create(
            model=args.omni_model,
            messages=api_messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return resp.choices[0].message.content.strip()

    def query_llm(messages):
        resp = omni_client.chat.completions.create(
            model=args.omni_model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return resp.choices[0].message.content.strip()

    # Read input manifest (file or directory of JSONL shards)
    rows = []
    if os.path.isdir(input_manifest):
        for fname in sorted(os.listdir(input_manifest)):
            if fname.endswith(".jsonl"):
                with open(os.path.join(input_manifest, fname)) as f:
                    rows.extend(json.loads(line) for line in f if line.strip())
    else:
        with open(input_manifest) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    # Run cascade on each row
    cascade_output = os.path.join(output_dir, "cascade_output.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[cascade] Running 3-pass cascade on {len(rows)} rows ({args.language})...")
    with open(cascade_output, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            # For Passes 1 & 2, we need to prepare audio as data URI
            audio_path = row.get("audio_filepath", "")
            if audio_path and os.path.exists(audio_path):
                import base64
                with open(audio_path, "rb") as af:
                    audio_b64 = base64.standard_b64encode(af.read()).decode("ascii")
                # Determine mime from extension
                ext = os.path.splitext(audio_path)[1].lower()
                mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".opus": "audio/opus", ".flac": "audio/flac"}
                mime = mime_map.get(ext, "audio/wav")
                row["_audio_data_uri"] = f"data:{mime};base64,{audio_b64}"

            result = run_cascade_on_row(row, p1_cfg, p2_cfg, p3_cfg, query_omni, query_llm)

            # Remove large data URI from output
            result.pop("_audio_data_uri", None)

            p1_text = result.get(p1_cfg["output_field"], "ERROR")
            p2_text = result.get(p2_cfg["output_field"], "ERROR")
            p3_text = result.get(p3_cfg["output_field"], "ERROR")
            fname = os.path.basename(audio_path)
            print(f"  [{i+1}/{len(rows)}] {fname}")
            print(f"    P1: {p1_text[:80]}...")
            print(f"    P2: {p2_text[:80]}...")
            print(f"    P3: {p3_text[:80]}...")

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[cascade] Done -> {cascade_output}")
    return cascade_output


def main() -> None:
    args = parse_args()
    stages = set(args.stages.split(","))

    if not args.no_ray_local:
        ray.init(address="local", ignore_reinit_error=True)

    os.makedirs(args.output_dir, exist_ok=True)
    current_manifest = args.input_manifest

    # ---- Stage 1: SED Inference ----
    if "sed" in stages:
        from nemo_curator.stages.audio.inference.sed import SEDInferenceStage

        if not args.sed_checkpoint:
            print("ERROR: --sed_checkpoint required")
            sys.exit(1)

        print(f"[pipeline] Stage 1: SED Inference ({args.sed_model_type})")
        pipeline = Pipeline(name="sed")
        pipeline.add_stage(JsonlReader(file_paths=current_manifest))
        from nemo_curator.stages.resources import Resources

        pipeline.add_stage(
            SEDInferenceStage(
                checkpoint_path=args.sed_checkpoint,
                model_type=args.sed_model_type,
                output_dir=os.path.join(args.output_dir, "sed"),
                resources=Resources(cpus=1.0),  # CPU-only: avoids CUDA driver compat issues
            )
        )
        sed_out = os.path.join(args.output_dir, "sed_manifest")
        pipeline.add_stage(JsonlWriter(path=sed_out, write_kwargs={"force_ascii": False}).with_(batch_size=1))
        pipeline.run(executor=RayDataExecutor())
        current_manifest = sed_out
        print(f"[pipeline] SED done -> {sed_out}")

    # ---- Stage 2: SED Postprocessing ----
    if "sed_post" in stages:
        from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage

        print("[pipeline] Stage 2: SED Postprocessing")
        pipeline = Pipeline(name="sed_post")
        pipeline.add_stage(JsonlReader(file_paths=current_manifest))
        pipeline.add_stage(SEDPostprocessingStage(speech_threshold=args.sed_threshold, min_duration_sec=0.3))
        sed_post_out = os.path.join(args.output_dir, "sed_post_manifest")
        pipeline.add_stage(JsonlWriter(path=sed_post_out, write_kwargs={"force_ascii": False}).with_(batch_size=1))
        pipeline.run(executor=RayDataExecutor())
        current_manifest = sed_post_out
        print(f"[pipeline] SED postprocessing done -> {sed_post_out}")

    # ---- Stage 3: Segment Extraction ----
    if "segment" in stages:
        from nemo_curator.stages.audio.segmentation.segment_extractor import SegmentExtractorStage

        print("[pipeline] Stage 3: Segment Extraction (fan-out)")
        pipeline = Pipeline(name="segment")
        pipeline.add_stage(JsonlReader(file_paths=current_manifest))
        pipeline.add_stage(SegmentExtractorStage())
        seg_out = os.path.join(args.output_dir, "segment_manifest")
        pipeline.add_stage(JsonlWriter(path=seg_out, write_kwargs={"force_ascii": False}).with_(batch_size=1))
        pipeline.run(executor=RayDataExecutor())
        current_manifest = seg_out
        print(f"[pipeline] Segmentation done -> {seg_out}")

    # ---- Stage 4: Diarization ----
    if "diarize" in stages:
        from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage

        print("[pipeline] Stage 4: Diarization (Sortformer)")
        # Sortformer expects AudioTask, not DocumentBatch from JsonlReader.
        # Run directly on manifest rows.
        _run_audio_stage_on_manifest(
            current_manifest,
            InferenceSortformerStage(),
            os.path.join(args.output_dir, "diarize_manifest"),
            extra_keys=["diar_segments"],
        )
        current_manifest = os.path.join(args.output_dir, "diarize_manifest")
        print(f"[pipeline] Diarization done -> {current_manifest}")

    # ---- Stage 5: Transcription Cascade (3-pass) ----
    if "transcribe" in stages:
        print(f"[pipeline] Stage 5: Transcription Cascade ({args.language}, 3-pass)")
        current_manifest = run_transcription_cascade(current_manifest, args.output_dir, args)

    # ---- Stage 6: Speaker Embedding ----
    if "embed" in stages:
        from nemo_curator.stages.audio.speaker_id.speaker_embedding_request import SpeakerEmbeddingRequestStage

        print("[pipeline] Stage 6: Speaker Embedding (TitaNet)")
        pipeline = Pipeline(name="embed")
        pipeline.add_stage(JsonlReader(file_paths=current_manifest))
        pipeline.add_stage(SpeakerEmbeddingRequestStage(model_name=args.speaker_model))
        embed_out = os.path.join(args.output_dir, "embed_manifest")
        pipeline.add_stage(JsonlWriter(path=embed_out, write_kwargs={"force_ascii": False}).with_(batch_size=1))
        pipeline.run(executor=RayDataExecutor())
        current_manifest = embed_out
        print(f"[pipeline] Embedding done -> {embed_out}")

    # ---- Stage 7: Speaker Clustering ----
    if "cluster" in stages:
        from nemo_curator.stages.audio.speaker_id.speaker_clustering_and_scoring import SpeakerClusteringStage

        print("[pipeline] Stage 7: Speaker Clustering (AHC)")
        pipeline = Pipeline(name="cluster")
        pipeline.add_stage(JsonlReader(file_paths=current_manifest))
        pipeline.add_stage(SpeakerClusteringStage(
            threshold=args.cluster_threshold,
            batch_size=args.cluster_batch_size,
        ))
        cluster_out = os.path.join(args.output_dir, "cluster_manifest")
        pipeline.add_stage(JsonlWriter(path=cluster_out, write_kwargs={"force_ascii": False}).with_(batch_size=1))
        pipeline.run(executor=RayDataExecutor())
        current_manifest = cluster_out
        print(f"[pipeline] Clustering done -> {cluster_out}")

    # ---- Stage 8: UTMOS Scoring ----
    if "utmos" in stages:
        from nemo_curator.stages.audio.metrics.utmosv2_score import GetUtmosv2ScoreStage

        print("[pipeline] Stage 8: UTMOS Scoring (UTMOSv2)")
        pipeline = Pipeline(name="utmos")
        pipeline.add_stage(JsonlReader(file_paths=current_manifest))
        pipeline.add_stage(
            GetUtmosv2ScoreStage(inference_batch_size=args.utmos_batch_size)
        )
        utmos_out = os.path.join(args.output_dir, "utmos_manifest")
        pipeline.add_stage(JsonlWriter(path=utmos_out, write_kwargs={"force_ascii": False}).with_(batch_size=1))
        pipeline.run(executor=RayDataExecutor())
        current_manifest = utmos_out
        print(f"[pipeline] UTMOS scoring done -> {utmos_out}")

    # ---- Final output ----
    print(f"\n[pipeline] All stages complete. Final output: {current_manifest}")


if __name__ == "__main__":
    main()
